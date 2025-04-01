# CL-MoE: Enhancing Multimodal Large Language Model with Dual Momentum Mixture-of-Experts for Continual Visual Question Answering

Tianyu Huai, Jie Zhou, Xingjiao Wu, Qin Chen, Qingchun Bai, Ze Zhou, Liang He.

<img src="./assets/architecture.png">

## Abstract

Multimodal large language models (MLLMs) have garnered widespread attention from researchers due to their remarkable understanding and generation capabilities in visual language tasks (e.g., visual question answering). However, the rapid pace of knowledge updates in the real world makes offline training of MLLMs costly, and when faced with non-stationary data streams, MLLMs suffer from catastrophic forgetting during learning. In this paper, we propose an MLLMs-based dual momentum Mixture-of-Experts (CL-MoE) framework for continual visual question answering (VQA). We integrate MLLMs with continual learning to utilize the rich commonsense knowledge in LLMs. We introduce a Dual-Router MoE (RMoE) strategy to select the global and local experts using task-level and instance-level routers, to robustly assign weights to the experts most appropriate for the task. Then, we design a dynamic Momentum MoE (MMoE) to update the parameters of experts dynamically based on the relationships between the experts and tasks/instances, so that the model can absorb new knowledge while maintaining existing knowledge. The extensive experimental results indicate that our method achieves state-of-the-art performance on 10 VQA tasks, proving the effectiveness of our approach.

## Install

1. Clone this repository and navigate to CLMoE folder

``` 
git clone https://github.com/ECNU-ICALK/CL-MoE.git
cd CL-MoE 
```

2. Install Package

```
conda create -n clmoe python=3.10 -y
conda activate clmoe
pip install --upgrade pip
pip install -e .
```

3. Install additional packages for training cases

```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

This repo is based on [LLaVA](https://github.com/haotian-liu/LLaVA). 
If you meet a problem, maybe you could find some solutions in issuses.

## Dataset

Please download the images from the COCO2014 dataset，include [train2014](http://images.cocodataset.org/zips/train2014.zip) and [val2014](http://images.cocodataset.org/zips/val2014.zip).

Please download the instruction from [CL4VQA](https://drive.google.com/drive/folders/1mcAjzmCU1UVW0TKvsHAy9Sr1hmwsJudo?usp=drive_link).

## Instruction Tuning

First, downloading the pretrained projectors in [LLaVA Model_Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md).

Setting `pretrain_mm_mlp_adapter` to the projector path.
You could modify the `deepspeed config` to change the deepspeed config.

We provide the scripts of our train order in `scripts/CLMoE/Train`.
Note, the `output_dir` of the previous script is the `previous_task_model_path` of the next training process.
Then, you could tune these datasets in your order.

## Evaluation

We have prepared the scripts to evaluate the trained model in `scripts/CLMoE/Eval`.

## Citation

```
@article{huai2025cl,
  title={CL-MoE: Enhancing Multimodal Large Language Model with Dual Momentum Mixture-of-Experts for Continual Visual Question Answering},
  author={Huai, Tianyu and Zhou, Jie and Wu, Xingjiao and Chen, Qin and Bai, Qingchun and Zhou, Ze and He, Liang},
  journal={arXiv preprint arXiv:2503.00413},
  year={2025}
}
```

## Acknowledgement

[LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon, and our base model LLaVA-1.5-7b that has the amazing vision-language capabilities!
