# HRNET-MTL

**HRNet-Based Multi-Task Learning Model for Age, Segmentation and Pose Estimation Task**

## Adopted Learning Methodology

![Adopted Learning Methodology](Untitled(1).png)

---

## Dataset & DataLoader Architecture

![Dataset and DataLoader Architecture](Untitled(2).png)


## Prerequisites

#### Hardware Requirements

- CUDA GPU (as backbone HrNet is designed to be built on cuda only!)

#### Python Libraries

- Pytorch
- numpy
- yacs
- monai
- gdown

#### Setup & File Organization

```bash
# Clone HrNet
git clone https://github.com/leoxiaobin/deep-high-resolution-net.pytorch.git

# Clone PCGrad
git clone https://github.com/devmt04/PCGrad-PyTorch.git
cp -r PCGrad-PyTorch/pcgrad.py deep-high-resolution-net.pytorch/lib/

cd deep-high-resolution-net.pytorch/lib

# Download HrNet-Pose backbone weights (pose_hrnet_w48_384x288.pth)
gdown --id '1UoJhTtjHNByZSm96W3yFTfU5upJnsKiS' -O /content/deep-high-resolution-net.pytorch/models/pytorch/pose_coco/
```

## Usages

```python

model = Model_HRNetMTL()

# traning
model.train(age_inputs=[], age_targets=[], seg_inputs=[], seg_targets=[], pose_inputs=[], pose_targets=[],  epochs=10, lr=0.01)

# evaluation
model = model.eval()
(age, seg_mask, pose_heatmaps) = model(image_tensor) # image_tensor : (1, 3 384, 288)

```

### Input Data Dimensions for traning

- age_inputs : (1, 3, 384, 288) # a Image tensor
- age_targets : (1,) # a positive real value
- seg_inputs : (1, 3, 384, 288) # a Image tensor
- seg_targets : (1, 1, 384, 288) # a bitmap Segmented mask
- pose_inputs : (1, 3, 384, 288) # a Image tensor
- pose_targets : (1, 17, 96, 72) # pose heatmaps (same as of HrNet)


# Citations

```
@inproceedings{sun2019deep,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={CVPR},
  year={2019}
}

@inproceedings{xiao2018simple,
    author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
    title={Simple Baselines for Human Pose Estimation and Tracking},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2018}
}

@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and 
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and 
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal   = {TPAMI}
  year={2019}
}

@misc{yu2020gradientsurgerymultitasklearning,
      title={Gradient Surgery for Multi-Task Learning}, 
      author={Tianhe Yu and Saurabh Kumar and Abhishek Gupta and Sergey Levine and Karol Hausman and Chelsea Finn},
      year={2020},
      eprint={2001.06782},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2001.06782}, 
}
```
