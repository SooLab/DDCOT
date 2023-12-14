import os
from torch.utils.data import Dataset
import os
import json
import numpy as np
import torch
from utils_prompt import *
import random

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image, ImageOps
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

img_shape = {
    "resnet": (512, 2048),
    "clip": (49, 1024),
    "detr": (100, 256),
}

def load_data_std(args):
    problems = json.load(open(os.path.join(args.data_root, 'scienceqa/problems.json')))
    
    if args.weak:
        pid_splits = json.load(open(os.path.join(args.data_root, 'scienceqa/pid_split_with_img.json')))
    else:
        pid_splits = json.load(open(os.path.join(args.data_root, 'scienceqa/pid_splits.json')))
    
    
    
    captions = json.load(open(args.caption_file))["captions"]

    for qid in problems:
        problems[qid]['caption'] = captions[qid] if qid in captions else ""

    train_qids = pid_splits['%s' % (args.train_split)]
    val_qids = pid_splits['%s' % (args.val_split)]
    test_qids = pid_splits['%s' % (args.test_split)]
    print(f"number of train problems: {len(train_qids)}\n")
    print(f"number of val problems: {len(val_qids)}\n")
    print(f"number of test problems: {len(test_qids)}\n")

    qids = {'train': train_qids, 'val':val_qids,'test':test_qids}
    return problems, qids,

def load_data_img(args):
    problems = json.load(open(os.path.join(args.data_root, 'scienceqa/problems.json')))
    
    
    if args.weak:
        pid_splits = json.load(open(os.path.join(args.data_root, 'scienceqa/pid_split_with_img.json')))
    else:
        pid_splits = json.load(open(os.path.join(args.data_root, 'scienceqa/pid_splits.json')))
    
    # captions = json.load(open(args.caption_file))["captions"]
    captions_train = json.load(open("data/scienceqa/new_caption_train.json"))
    captions_test = json.load(open("data/scienceqa/new_caption_test.json"))

    name_maps = json.load(open('name_map.json'))


    for qid in problems:
        if qid in pid_splits['train'] and qid in captions_train:
            problems[qid]['caption'] = captions_train[qid]
        elif qid in pid_splits['test'] and qid in captions_test:
            problems[qid]['caption'] = captions_test[qid]
        else:
            problems[qid]['caption'] = ""


    train_qids = pid_splits['%s' % (args.train_split)]


    val_qids = pid_splits['%s' % (args.val_split)]
    test_qids = pid_splits['%s' % (args.test_split)]

    print(f"number of train problems: {len(train_qids)}\n")
    print(f"number of val problems: {len(val_qids)}\n")
    print(f"number of test problems: {len(test_qids)}\n")

    qids = {'train': train_qids, 'val':val_qids,'test':test_qids}
    return problems, qids, name_maps

class ScienceQADatasetStd(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, problems, qids, tokenizer, source_len, target_len, args, test_le=None, weak=None
    ):
        self.tokenizer = tokenizer
        self.data = {qid : problems[qid] for qid in qids}
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = []
        self.source_text = []
        
        R_from_gpt = json.load(open("data/scienceqa/output_gpt_train_no_anwser.json"))
        R_from_gpt_test = json.load(open("data/scienceqa/output_gpt_test_no_anwser.json"))
        
        if test_le is not None:
            test_le_data =json.load(open(test_le))["preds"]
        else:
            test_le_data = None
        idx = 0
        for qid in self.data:
            if test_le_data is not None:
                curr_le_data = test_le_data[idx]
                idx += 1
            elif weak == "train":
                curr_le_data = R_from_gpt[qid]
            elif weak == "test":
                curr_le_data = R_from_gpt_test[qid]
            else:
                curr_le_data = None
                
            prompt, target = build_train_pair(problems, qid, args, curr_le_data)
            self.target_text.append(target)
            self.source_text.append(prompt)

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        ind1 = source_text.rfind('Context: ')
        ind2 = source_text.rfind('Caption: ') + 9
        ind3 = source_text.rfind('Options: ')
        ind4 = source_text.rfind('Solution: ')
        ind5 = source_text.rfind('Answer: ')

        source_text_begin = source_text[:ind1]
        source_text_context = source_text[ind1:ind2]
        source_text_caption = source_text[ind2:ind3]

        source_text_part_pre = source_text_begin + source_text_context 
        source_text_part_last = source_text_begin + source_text_context + source_text_caption

        source_part_pre = self.tokenizer.batch_encode_plus(
            [source_text_part_pre],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_part_last = self.tokenizer.batch_encode_plus(
            [source_text_part_last],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        part_len_begin = source_part_pre["attention_mask"].squeeze().sum() - 1
        part_len_end = source_part_last["attention_mask"].squeeze().sum() - 1

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze().tolist()
        
        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": target_ids,
            "part_len_begin": part_len_begin,
            "part_len_end": part_len_end
        }


class ScienceQADatasetImg(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, problems, qids, name_maps, tokenizer, source_len, target_len, args, test_le=None, use_mask=True, weak=""
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = {qid : problems[qid] for qid in qids}
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = []
        self.source_text = []
        
        self.image_ids = []
        self.use_mask = use_mask
        self.name_maps = name_maps
        self.image_root = f"data/scienceqa/images/{weak}"
        self.it = 0
        self.set = weak
        self.epoch = 30
        self.qids = []

        self.image_feats = np.load(f'clip_feat_{weak}.npy', allow_pickle=True).item()
        
        
        
        new_R_train = json.load(open("data/scienceqa/reform_Rv2_train_noR.json"))
        new_R_test = json.load(open("new_R.json"))

        if test_le is not None:
            test_le_data =json.load(open(test_le))["preds"]
        else:
            test_le_data = None
        idx = 0
        for qid in self.data:

            if test_le_data is not None:
                curr_le_data = test_le_data[idx]
                idx += 1
            elif weak == "train":
                curr_le_data = new_R_train[qid]
            elif weak == "test":
                curr_le_data = new_R_test[qid]

            else:
                curr_le_data = None
            prompt, target = build_train_pair(problems, qid, args, curr_le_data)

            self.target_text.append(target)
            
            self.source_text.append(prompt)
    
            self.image_ids.append(str(qid))

                
    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])
        len_end = 6
        image_ids = self.image_ids[index]

        image_tensor = self.image_feats[image_ids]

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())


        ind1 = source_text.rfind('Context: ')
        ind3 = source_text.rfind('Options: ')
        ind4 = source_text.rfind('Solution: ')
        ind5 = source_text.rfind('Answer: ')
        

        source_text_begin = source_text[:ind1]
        source_text_context = source_text[ind1:ind3]
        # source_text_caption = source_text[ind2:ind3]
        source_text_option = source_text[ind3:ind4]
        source_text_solution = source_text[ind4:ind5-len_end]
        # source_text_end = source_text[ind5-6:]

        source_text_part = source_text_begin + source_text_context
        source_R = source_text_solution

        source_part_R = self.tokenizer.batch_encode_plus(
            [source_R],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_part = self.tokenizer.batch_encode_plus(
            [source_text_part],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        
        part_len = source_part["attention_mask"].squeeze().sum() - 1
        R_len = source_part_R["attention_mask"].squeeze().sum() - 1
        
        
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze().tolist()
        
        
        return {
            "input_ids": source_ids,
            "forclip_input": source_ids,
            "attention_mask": source_mask,
            "image_ids": image_tensor,
            "source_len": [part_len, R_len],
            "labels": target_ids,
            "qids": int(image_ids),
            
        }
        
