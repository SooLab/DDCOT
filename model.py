'''
Adapted from https://github.com/huggingface/transformers
'''

from transformers import T5Config, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack, __HEAD_MASK_WARNING_MSG, T5EncoderModel, T5LayerCrossAttention, T5LayerNorm
import copy
import math
import os
import warnings
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
)
import time

class T5ForMultimodalGeneration(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: T5Config, patch_size, padding_idx, save_dir):
        super().__init__(config)
        self.model_dim = config.d_model
        
        self.padding_idx = padding_idx
        self.out = open(os.path.join(save_dir, 'bboxes.txt'), 'w')

        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        
        self.patch_num, self.patch_dim = patch_size
        self.n_prompt = 3

        self.prompt_image_begin = nn.Parameter(torch.zeros((self.n_prompt, config.d_model)), requires_grad=True)
        self.prompt_image_end = nn.Parameter(torch.zeros((self.n_prompt, config.d_model)), requires_grad=True)

        self.image_emb = nn.Parameter(torch.zeros((self.patch_num, config.d_model)), requires_grad=True)
        self.image_dense = nn.Linear(self.patch_dim, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.n_aera = 16
        no_PE = False
        if no_PE:
            self.vis_embed = VisualEmbedding_NoPE(config)
        else:
            self.vis_embed = VisualEmbedding(config, self.n_aera)
        

        self.use_vis_embed = True
        
        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.max_len = 512

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_ids=None,
        forclip_input=None,
        exchange=None,
        qids=None,
        # input_text=None,
        image_embedding=None,
        source_len=None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        
        if image_ids is not None:
            if self.use_vis_embed:
                global_feat = image_ids[:, 0, :]
                image_feats = image_ids[:, 1:, :]
                B, N, C = image_feats.shape

                global_feat = global_feat.reshape(B, 1, C)

                global_feat = self.image_dense(global_feat)
                local_feat = self.image_dense(image_feats)

                inputs_embed = self.shared(input_ids)

                image_embedding, bboxes = self.vis_embed(global_feat, local_feat, inputs_embed)
                
            else:
                global_feat = image_ids[:, 1, :]
                image_feats = image_ids[:, 1:, :]
                B, N, C = image_feats.shape

                global_feat = global_feat.reshape(B, 1, C)

                global_feat_norm = global_feat / global_feat.norm(dim=-1, keepdim=True)
                image_feats_nrom = image_feats / image_feats.norm(dim=-1, keepdim=True)

                G2L = global_feat_norm @ image_feats_nrom.permute(0, 2, 1)
                L2L = image_feats_nrom @ image_feats_nrom.permute(0, 2, 1)
                L2L = L2L - torch.eye(N).cuda()*L2L

                L2L_reduce = L2L.mean(dim=-1)

                W_g = (L2L_reduce + G2L.squeeze(1)).softmax(dim=-1)
                W_g = W_g.unsqueeze(2).repeat(1, 1, 1024)
                W_l = 1 - W_g
                updated_image_feats = W_g * global_feat.repeat(1, N, 1) + W_l * image_feats
                image_embedding = self.image_dense(updated_image_feats)

            image_embedding = torch.cat((self.prompt_image_begin.unsqueeze(0).repeat(B, 1, 1), image_embedding, self.prompt_image_end.unsqueeze(0).repeat(B, 1, 1)), dim=1)

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            
            encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    image_embedding=image_embedding,
                    # exchange=exchange,
                    source_len=source_len,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        B, _, _ = hidden_states.shape

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        N = 2*self.n_prompt + self.n_aera
        new_attenton_mask = torch.ones((B, self.max_len+N)).cuda()
        new_attenton_mask[:, N:] = attention_mask

        attention_mask = new_attenton_mask

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        
        
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        image_ids=None,
        source_len=None,
        exchange=None,
        qids=None,
        forclip_input=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "image_ids": image_ids,
            "source_len": source_len,
            "exchange": exchange,
            "qids": qids,
            "forclip_input": forclip_input,
        }


class VisualEmbedding(nn.Module):
    def __init__(self, t5config, n_aera):
        super().__init__()
        feat_dim = t5config.d_model
        pos_dim = 5
        self.n_aera = n_aera
        self.global_ca = T5LayerCrossAttention(t5config)
        self.local_ca = T5LayerCrossAttention(t5config)
        self.box_head = Mlp_boxhead(feat_dim, feat_dim, 4*self.n_aera)

        absolute_vis_pos_embedding = [nn.Linear(pos_dim, feat_dim)]
        self.absolute_vis_pos_embedding = nn.Sequential(*absolute_vis_pos_embedding)

        self.layer_norm = T5LayerNorm(feat_dim, eps=t5config.layer_norm_epsilon)

    def get_area(self, pos):
        """
        Args
            pos: [B, N, 4]
                (x1, x2, y1, y2)
        Return
            area : [B, N]
        """
        # [B, N]
        height = pos[:, :, 3] - pos[:, :, 2]
        width = pos[:, :, 1] - pos[:, :, 0]
        area = height * width
        return area

    def forward(self, global_feats, local_feats, hidden_states):
        """
            global_feats: B 1 C
            local_feats: B N C
            hidden_states:  B N_h C
        """

        B, N, _ = local_feats.size()

        pos = self.box_head(self.global_ca(global_feats, key_value_states=hidden_states)[0])
        pos1 = pos.reshape(B,self.n_aera,4).sigmoid()
        pos1 = pos.reshape(B,self.n_aera,4)


        area = self.get_area(pos1).unsqueeze(2) # [B, N, 1]
        pos = torch.cat([pos1, area], dim=2) # [B, N, 5]

        # [B, N, d_model]
        absolute_vis_pos_embedding = self.absolute_vis_pos_embedding(pos)
        local_feats = self.local_ca(absolute_vis_pos_embedding, key_value_states=local_feats)[0]
        vis_embedding = local_feats + absolute_vis_pos_embedding

        vis_embedding = self.layer_norm(vis_embedding)
        return vis_embedding, pos1

class Mlp_boxhead(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VisualEmbedding_NoPE(nn.Module):
    def __init__(self, t5config):
        super().__init__()
        feat_dim = t5config.d_model
        self.global_ca = T5LayerCrossAttention(t5config)
        self.local_ca = T5LayerCrossAttention(t5config)
        self.sample_tokens = nn.Parameter(torch.zeros(1, 16, feat_dim))

        self.layer_norm = T5LayerNorm(t5config.d_model, eps=t5config.layer_norm_epsilon)

    def forward(self, _, local_feats, hidden_states):
        """
                global_feats: B 1 C
                local_feats: B N C
                hidden_states:   B N_h C
        """

        B, N, _ = local_feats.size()
        tokens = self.sample_tokens.expand(B,-1,-1)
        tokens = self.global_ca(tokens, key_value_states=hidden_states)[0]
        local_feats = self.local_ca(tokens, key_value_states=local_feats)[0]
        vis_embedding = local_feats + tokens

        vis_embedding = self.layer_norm(vis_embedding)
        return vis_embedding
