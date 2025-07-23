from models.pose_hrnet import get_pose_net
from config.default import update_config, _C
from yacs.config import CfgNode as CN
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from monai.losses.dice import DiceLoss
from pcgrad import PCGrad

class TaskDataset(Dataset):
  def __init__(self, data, label, data_dt, label_dt):
    self.data = data
    self.label = label
    self.data_dt = data_dt;
    self.label_dt = label_dt

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    x = torch.tensor(self.data[idx], dtype=self.data_dt)
    y = torch.tensor(self.label[idx], dtype=self.label_dt)
    return x, y

class MTLDataset(Dataset):
  def __init__(self, age_dl, seg_dl, pose_dl):
    if(len(age_dl) != len(seg_dl) or len(age_dl) != len(pose_dl) or len(seg_dl) != len(pose_dl)):
      raise RuntimeError("All passed datasets must are of same length")

    self.age_dl = age_dl
    self.seg_dl = seg_dl
    self.pose_dl = pose_dl
    self.length = len(self.age_dl)


  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    return {
        "age": list(self.age_dl)[idx],
        "seg": list(self.seg_dl)[idx],
        "pose": list(self.pose_dl)[idx]
    }


class AgeHead(nn.Module):
  def __init__(self, in_channels = 48, num_classes = 5):
    super().__init__()

    self.gap = nn.AdaptiveAvgPool2d((1,1)) # num of output channels = num of input channels
    self.gmp = nn.AdaptiveMaxPool2d((1,1)) # will concat their result later

    self.classifier = nn.Sequential(
      nn.Linear(96, 48), # in = 96 as after concat input become 48+48=96
      nn.BatchNorm1d(48),
      nn.ReLU(),
      nn.Dropout(0.3),

      nn.Linear(48, 24),
      nn.ReLU(),

      nn.Linear(24, num_classes)
    )

  def forward(self, x):
    x1 = self.gap(x)
    x2 = self.gmp(x)
    x = torch.cat((x1, x2), dim=1)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x 

class SegHead(nn.Module):
  def __init__(self, in_channels = 48, num_classes=1):
    super().__init__()

    self.conv1 = nn.Sequential(
      nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), # 96x72 → 96x72
      nn.BatchNorm2d(64),
      nn.ReLU() # fused with x_conv2
    )
    self.up1 = nn.Sequential(
      nn.ConvTranspose2d(64+64, 32, kernel_size=4, stride=2, padding=1), # 96x72 → 192x144
      nn.BatchNorm2d(32),
      nn.ReLU() # fused with x_conv1
    )
    self.conv2 = nn.Sequential(
      nn.Conv2d(32 + 64, 32, kernel_size=3, padding=1),   # 192x144 → 192x144
      nn.BatchNorm2d(32),
      nn.ReLU()
    )
    self.up2 = nn.Sequential(
      nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # 192x144 → 384x288
      nn.BatchNorm2d(16),
      nn.ReLU()
    )
    self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)  # 384x288 → 384x288

  def forward(self, x, x_conv1, x_conv2):
    x = self.conv1(x)
    x = torch.cat([x, x_conv2], dim=1)

    x = self.up1(x)
    x = torch.cat([x, x_conv1], dim=1)

    x = self.conv2(x)
    x = self.up2(x)
    x = self.final_conv(x)

    return x  # Output shape: [1, 1, 384, 288]


class AdaptiveFeatureBlock(nn.Module):
  def __init__(self, in_channels=48, hidden_channels=96, dropout=0.3):
    super().__init__()

    self.block = nn.Sequential(
      nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),  # depthwise
      nn.Conv2d(in_channels, hidden_channels, kernel_size=1),  # pointwise expand
      nn.BatchNorm2d(hidden_channels),
      nn.ReLU(),
      nn.Dropout2d(dropout),

      nn.Conv2d(hidden_channels, in_channels, kernel_size=1),  # compress
      nn.BatchNorm2d(in_channels),
      nn.ReLU()
    )

  def forward(self, x):
    return self.block(x)

class HRNetMTL(nn.Module):
  def __init__(self, hrnet):
    super().__init__()
    self.backbone = hrnet

    # input : (1, 48, 96, 72)
    self.age_adapted_feature = AdaptiveFeatureBlock(in_channels=48).to("cuda")
    self.seg_adapted_feature = AdaptiveFeatureBlock(in_channels=48).to("cuda")
    self.pose_adapted_feature = AdaptiveFeatureBlock(in_channels=48).to("cuda")

    self.age_head = AgeHead(in_channels=48).to("cuda")
    self.seg_head = SegHead(in_channels=48).to("cuda")
    self.pose_head = self.backbone.final_layer


  def forward_backbone(self, x):
    x = self.backbone.conv1(x)
    x = self.backbone.bn1(x)
    x_conv1 = self.backbone.relu(x)

    x = self.backbone.conv2(x_conv1)
    x = self.backbone.bn2(x)
    x_conv2 = self.backbone.relu(x)

    x = self.backbone.layer1(x_conv2)

    # Stage 2
    x_list = []
    for i in range(self.backbone.stage2_cfg['NUM_BRANCHES']):
        if self.backbone.transition1[i] is not None:
            x_list.append(self.backbone.transition1[i](x))
        else:
            x_list.append(x)
    y_list = self.backbone.stage2(x_list)

    # Stage 3
    x_list = []
    for i in range(self.backbone.stage3_cfg['NUM_BRANCHES']):
        if self.backbone.transition2[i] is not None:
            x_list.append(self.backbone.transition2[i](y_list[-1]))
        else:
            x_list.append(y_list[i])
    y_list = self.backbone.stage3(x_list)

    # Stage 4
    x_list = []
    for i in range(self.backbone.stage4_cfg['NUM_BRANCHES']):
        if self.backbone.transition3[i] is not None:
            x_list.append(self.backbone.transition3[i](y_list[-1]))
        else:
            x_list.append(y_list[i])
    y_list = self.backbone.stage4(x_list)

    out_backbone = y_list[0]

    return out_backbone, x_conv1, x_conv2

  def forward_age(self, out_backbone):
    x_age_adapted = self.age_adapted_feature(out_backbone)
    age = self.age_head(x_age_adapted)
    return age

  def forward_seg(self, out_backbone, x_conv1, x_conv2):
    x_seg_adapted = self.seg_adapted_feature(out_backbone)
    seg = self.seg_head(x_seg_adapted, x_conv1, x_conv2)
    return seg

  def forward_pose(self, out_backbone):
    x_pose_adapted = self.pose_adapted_feature(out_backbone)
    pose = self.pose_head(x_pose_adapted)
    return pose

  #override forward method from original implimentation
  def forward(self, x):
    out_backbone, x_conv1, x_conv2 = self.forward_backbone(x)
    age = self.forward_age(out_backbone)
    seg = self.forward_seg(out_backbone, x_conv1, x_conv2)
    pose = self.forward_pose(out_backbone)
    return (age, seg, pose)

class BCEDiceLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.bce = nn.BCEWithLogitsLoss()
    self.dice = DiceLoss(reduction="mean")

  def forward(self, out, truth):
    return self.bce(out, truth) + self.dice(out, truth)

class Model_HRNetMTL():
  def __init__(self):
    self.model = self.get_model()

  def __call__(self, x):
    return self.model(x)

  @staticmethod
  def init_hrnet_model(cfg_path, weight_path, device="cuda"):
    args = CN()
    args.cfg = cfg_path
    args.opts = ['MODEL.PRETRAINED', weight_path]
    args.modelDir = ""
    args.logDir = ""
    args.dataDir = ""
    update_config(_C, args)
    hrnet = get_pose_net(cfg=_C, is_train=True)
    hrnet = hrnet.to(device)
    return hrnet

  def get_model(self):
    hrnet = self.init_hrnet_model(
        cfg_path="//content/deep-high-resolution-net.pytorch/experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml",
        weight_path="/content/deep-high-resolution-net.pytorch/models/pytorch/pose_coco/pose_hrnet_w48_384x288.pth"
    )
    return HRNetMTL(hrnet)

  def eval(self):
    self.model.eval()
    return self

  def train(self, age_inputs, age_targets, seg_inputs, seg_targets, pose_inputs, pose_targets,  epochs=None, lr=0.01):
    if(train_loader == None) : raise RuntimeError("train_loader is None!")
    if(epochs==None) : raise RuntimeError("epochs is None!")
    
    # all datasets must be of same length
    age_ds = TaskDataset(age_inputs, age_targets, data_dt = torch.float32, label_dt = torch.float32)
    seg_ds = TaskDataset(seg_inputs, seg_targets, data_dt = torch.float32, label_dt = torch.float32)
    pose_ds = TaskDataset(pose_inputs, pose_targets, data_dt = torch.float32, label_dt = torch.float32)

    age_dl = DataLoader(age_ds, batch_size=minibatch_size, shuffle=True)
    seg_dl = DataLoader(seg_ds, batch_size=minibatch_size, shuffle=True)
    pose_dl = DataLoader(pose_ds, batch_size=minibatch_size, shuffle=True)

    train_dataset = MTLDataset(age_dl, seg_dl, pose_dl)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(self.model.parameters(), lr=lr)
    criterion_age = nn.CrossEntropyLoss(reduction="mean")
    criterion_pose = nn.MSELoss(reduction="mean")
    criterion_seg = BCEDiceLoss()

    for epoch in range(epochs):
      self.model.train()
      for batch_idx, batch in enumerate(train_loader):
        shared_grad = []
        batch_loss = 0.0
        optimizer.zero_grad()
        for batch_label, mini_batch in batch.items():
          input, target = mini_batch
          out_backbone, x_conv1, x_conv2 = self.model.forward_backbone(input)

          if(batch_label == "age"):
            out = self.model.forward_age(out_backbone)
            batch_loss = criterion_age(out, target)

          elif(batch_label == "seg"):
            out = self.model.forward_seg(out_backbone, x_conv1, x_conv2)
            batch_loss = criterion_seg(out, target)

          elif(batch_label == "pose"):
            out = self.model.forward_pose(out_backbone)
            batch_loss = criterion_pose(out, target)

          else:
            raise ValueError(f"Unknown task: {batch_label}")

          batch_loss.backward(retain_graph=False)
          shared_grad.append([p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p, requires_grad=False) for p in self.model.backbone.parameters()])
          self.model.backbone.zero_grad()

        resolved_gradients = PCGrad(shared_grad).resolve_grads(verbose=True)
        # gradient clipping
        for param, grad in zip(self.model.backbone.parameters(), resolved_gradients):
              param.grad = grad
        optimizer.step()
    return self
