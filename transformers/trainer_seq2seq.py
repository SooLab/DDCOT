# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import Dataset

from .deepspeed import is_deepspeed_zero3_enabled
from .trainer import Trainer
from .trainer_utils import PredictionOutput
from .utils import logging


logger = logging.get_logger(__name__)


class Seq2SeqTrainer(Trainer):
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """

        gen_kwargs = gen_kwargs.copy()
        gen_kwargs["max_length"] = (
            gen_kwargs["max_length"] if gen_kwargs.get("max_length") is not None else self.args.generation_max_length
        )
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs

        return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        **gen_kwargs
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.

        <Tip>

        If your predictions or labels have different sequence lengths (for instance because you're doing dynamic
        padding in a token classification task) the predictions will be padded (on the right) to allow for
        concatenation into one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """

        gen_kwargs = gen_kwargs.copy()
        gen_kwargs["max_length"] = (
            gen_kwargs["max_length"] if gen_kwargs.get("max_length") is not None else self.args.generation_max_length
        )
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs

        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = self._gen_kwargs.copy()
        gen_kwargs["max_length"] = (
            gen_kwargs["max_length"] if gen_kwargs.get("max_length") is not None else self.model.config.max_length
        )
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        def xxyy2xyxy(box, h, w):
            box[:,:2] *=w
            box[:,2:] *=h
            return torch.stack((box[:,0], box[:,2], box[:,1], box[:,3])).transpose(0,1)
        from PIL import Image, ImageColor, ImageDraw
        def visualize(bbox, img_path, save_path):

            try:
                img = Image.open(img_path)
            except:
                return
            w, h = img.size
            formatted_bbox = xxyy2xyxy(bbox, h, w)
            img.putalpha(255)
            draw = ImageDraw.Draw(img)
            for box in formatted_bbox:
                draw.rectangle(
                    box.tolist(),
                    outline="#6600FF",
                    width=2,
                )
            save_img = img.convert('RGB')
            # save_img.show()
            save_img.save(save_path)
            
        ours = True 
        # gen_kwargs["image_ids"] = inputs["image_ids"]
        if "image_ids" in inputs.keys() and ours:
            image_ids = inputs["image_ids"]
            input_ids = inputs["input_ids"]
            qids = inputs["qids"]
            with torch.no_grad():
                # global_feat, image_feats = self.model.clip_model.visual(image_ids)

                if self.model.use_vis_embed:
                    global_feat = image_ids[:, 1, :]
                    image_feats = image_ids[:, 1:, :]
                    B, N, C = image_feats.shape

                    global_feat = global_feat.reshape(B, 1, C)

                    global_feat = self.model.image_dense(global_feat)
                    local_feat = self.model.image_dense(image_feats)

                    inputs_embed = self.model.shared(input_ids)

                    image_embedding, bboxes = self.model.vis_embed(global_feat, local_feat, inputs_embed)
                    # vis = True
                    # if vis:
                    #     for b in range(B):
                            
                    #         bbox = bboxes[b]
                    #         image_path = f"/public/home/yangbin/projects/COT-main/data/scienceqa/images/test/{qids[b]}/image.png"
                    #         save_path = f"/public/home/yangbin/projects/COT-main/vis/test1/{qids[b]}.png"
                    #         visualize(bbox, image_path, save_path)
                
                else:

                    global_feat = image_ids[:, 1, :]
                    image_feats = image_ids[:, 1:, :]

                    B, N, C = image_feats.shape

                    global_feat = global_feat.reshape(B, 1, C)

                    # N, B, C = image_feats.shape

                    # global_feat = global_feat.reshape(1, B, C).permute(1, 0, 2)
                    # image_feats = image_feats.permute(1, 0, 2)

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
                
                    image_embedding = self.model.image_dense(image_feats)
                # image_embedding = image_embedding + self.model.image_emb
                image_embedding = torch.cat((self.model.prompt_image_begin.unsqueeze(0).repeat(B, 1, 1), image_embedding, self.model.prompt_image_end.unsqueeze(0).repeat(B, 1, 1)), dim=1)
                # image_embedding = self.model.image_dense2(image_embedding)
            
            gen_kwargs["image_embedding"] = image_embedding
            gen_kwargs["source_len"] = inputs["source_len"]
            # gen_kwargs["exchange"] = inputs["exchange"]

        # else:
        #     gen_kwargs["part_len_begin"] = inputs["part_len_begin"]
        #     gen_kwargs["part_len_end"] = inputs["part_len_end"]

        generated_tokens = self.model.generate(
            generation_inputs,
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            with self.compute_loss_context_manager():
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor
