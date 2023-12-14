# DDCOT
[NeurIPS 2023]DDCoT: Duty-Distinct Chain-of-Thought Prompting for Multimodal Reasoning in Language Models
This repository is the official implementation of [DDCoT](https://arxiv.org/abs/2310.16436).

**[DDCoT: Duty-Distinct Chain-of-Thought Prompting for Multimodal Reasoning in Language Models](https://arxiv.org/abs/2310.16436)**

[Ge Zheng∗]() , [Bin Yang∗](), [Jiajin Tang*](https://toneyaya.github.io/), [Hong-Yu Zhou](https://zhouhy.org/), [Sibei Yang†](https://faculty.sist.shanghaitech.edu.cn/yangsibei/)

*Equal contribution; †Corresponding Author
![image](https://github.com/YangBin55/DDCOT/blob/main/images/teaser.png)

## Requirements

Install all required python dependencies:

```
pip install -r requirements.txt
```

## Datasets

Download the dataset from the following repository:

```
https://github.com/lupantech/ScienceQA/tree/main/data
```

## Instructions

### Generate Rationale
To generate rationale for one sample，you can run the following code.
```
python rationale_generation.py
```

### Train

Model weights, generated rationales and extracted vision features will be released soon.

```
python main.py \
    --model  allenai/unifiedqa-t5-base \
    --user_msg answer --img_type clip \
    --bs 16 --eval_bs 4 --eval_acc 10 --output_len 64 --lr 1e-4 \
    --prompt_format QCMG-A \
    --output_dir model_path --use_generate --final_eval
```

### Inference 

```
python main.py \
    --model allenai/unifiedqa-t5-base \
    --user_msg answer --img_type clip \
    --bs 8 --eval_bs 1 --eval_acc 10 --output_len 64 \
    --final_eval --prompt_format QCMG-A \
    --evaluate_dir model_path --use_generate 
```


## Citing DDCoT

```
@article{zheng2023ddcot,
  title={Ddcot: Duty-distinct chain-of-thought prompting for multimodal reasoning in language models},
  author={Zheng, Ge and Yang, Bin and Tang, Jiajin and Zhou, Hong-Yu and Yang, Sibei},
  journal={arXiv preprint arXiv:2310.16436},
  year={2023}
}
```

## License

This project is licensed under the Apache-2.0 License.

## Acknowledgement

Part of our codes are adapted from [ScienceQA](https://github.com/lupantech/ScienceQA).
