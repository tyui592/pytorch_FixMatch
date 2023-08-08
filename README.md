FixMatch
==

**Unofficial Pytorch Implementation "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence"**

# Usage
## Requirements
* pytorch (v2.0.1)
* torchvision (v0.15.2)
* PIL (v9.4.0)
* wandb (Optional)

## Arguments
* Augmentation Policy (`--augs`)
  - 0: no augmentation
  - 1: weak augmentation
  - 2: strong augmentation (based on RandAug)
* Check [`config.py`](./config.py) file for details. (Default parameters are set for cifar10)

## Example Scripts
```bash
# Model Training
$ python main.py --mode 'train' --data 'cifar10' --num_X 4000 --augs 1 2  --nesterov --amp --include_x_in_u --save_path ./model-store/001
>>>...
>>>Tue Aug  8 00:17:33 2023: Iteration: [1046528/1048576], Ls: 0.1149, Lu: 0.1146, Mask: 0.9892, Accuracy(train/test): [1.0000/0.9554]
>>>Tue Aug  8 00:31:01 2023: Iteration: [1047552/1048576], Ls: 0.1145, Lu: 0.1142, Mask: 0.9897, Accuracy(train/test): [0.9999/0.9556]
>>>Tue Aug  8 00:44:49 2023: Iteration: [1048576/1048576], Ls: 0.1153, Lu: 0.1149, Mask: 0.9897, Accuracy(train/test): [0.9999/0.9556]

# Model Evaluation
$ python main.py --mode 'eval' --load_path ./model-store/001/ckpt.pth
>>>...
>>>Model Performance: 0.9556
```

# Results

## CIFAR10
| Num Labaled Data | Top 1 Acc |
| --- | --- | 
| 4000 | 0.9556 | 
| 250 | 0.9473 |
| 40 | 0.9352 |

**Model weights (and training logs) of the above performance are on [the release page](https://github.com/tyui592/pytorch_FixMatch/releases/tag/v0.1).**

## References
- https://arxiv.org/abs/2001.07685
- https://www.zijianhu.com/post/pytorch/ema/
- https://github.com/kekmodel/FixMatch-pytorch
