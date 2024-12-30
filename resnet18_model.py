from timm.models.nfnet import model_cfgs
from torch import nn as nn
from torch.nn import functional as F

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        model_cfgs = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True)
        )
    def forward (self, x):
        x = model_cfgs(x)
        return x
