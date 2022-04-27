"""
Video Face Manipulation Detection Through Ensemble of CNNs

Image and Sound Processing Lab - Politecnico di Milano

Nicolò Bonettini
Edoardo Daniele Cannas
Sara Mandelli
Luca Bondi
Paolo Bestagini
"""
from bdb import effective
from collections import OrderedDict
from textwrap import indent

import torch
from efficientnet_pytorch import EfficientNet
from torch import nn as nn
from torch.nn import functional as F
from torchvision import transforms

from . import externals

"""
Feature Extractor
"""


class FeatureExtractor(nn.Module):
    """
    Abstract class to be extended when supporting features extraction.
    It also provides standard normalized and parameters
    """

    def features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_trainable_parameters(self):
        return self.parameters()

    @staticmethod
    def get_normalizer():
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

"""
EfficientNet Block Attention
"""

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class FrontFeatureAttention(nn.Module):
    def __init__(self, in_dim):
        super(FrontFeatureAttention, self).__init__()
        # self.conv = nn.Conv2d(in_channels= in_dim, out_channels= 1, kernel_size= 1)
        # self.fm = nn.Conv2d(in_channels= 2, out_channels= 1, kernel_size= 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        # feature_reduct = self.conv(x)
        # x = torch.cat([feature_reduct, maxout], dim=1)
        # x = self.fm(x)
        x = maxout - avgout
        return self.sigmoid(x)

class EfficientNetAtt(EfficientNet):
    def init_att(self, model: str, width: int = 0):

        # self.attconv0 = nn.Conv2d(kernel_size=1, in_channels=24, out_channels=1)
        # self.attconv1 = nn.Conv2d(kernel_size=1, in_channels=32, out_channels=1)
        # self.attconv2 = nn.Conv2d(kernel_size=1, in_channels=56, out_channels=1)

        self.ffa0 = FrontFeatureAttention(24)
        self.ffa1 = FrontFeatureAttention(32)
        self.ffa2 = FrontFeatureAttention(56)

        self.ca0 = ChannelAttention(112)
        self.ca1 = ChannelAttention(160)
        self.ca2 = ChannelAttention(272)
        self.ca3 = ChannelAttention(448)

        self.sa = SpatialAttention()

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        EfficientNet-b4 

        torch.Size([1, 24, 112, 112])
        depth = 2

        torch.Size([1, 32, 56, 56])
        depth = 4
        
        torch.Size([1, 56, 28, 28])
        depth = 4

        torch.Size([1, 112, ?, ?])
        depth = 6

        torch.Size([1, 160, 14, 14])
        depth = 6

        torch.Size([1, 272, ?, ?])
        depth = 8

        torch.Size([1, 448, 7, 7])
        depth = 2

        """


        # Stem
        x = self._swish(self._bn0(self._conv_stem(x)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            # print(idx)
            if idx == 1:
                x = self.ffa0(x) * x
            elif idx == 5:
                x = self.ffa1(x) * x
            elif idx == 9:
                x = self.ffa2(x) * x
            elif idx == 15:
                x = self.ca0(x) * x  
                x = self.sa(x) * x
            elif idx == 21:
                x = self.ca1(x) * x  
                x = self.sa(x) * x
            elif idx == 29:
                x = self.ca2(x) * x  
                x = self.sa(x) * x
            elif idx == 30:
                x = self.ca3(x) * x  
                x = self.sa(x) * x

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

class EfficientNetCA_Gen(FeatureExtractor):
    def __init__(self, model: str, width: int):
        super(EfficientNetCA_Gen, self).__init__()

        self.efficientnet = EfficientNetAtt.from_pretrained(model)
        self.efficientnet.init_att(model, width)
        self.classifier = nn.Linear(self.efficientnet._conv_head.out_channels, 1)
        del self.efficientnet._fc

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.efficientnet.extract_features(x)
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.efficientnet._dropout(x)
        x = self.classifier(x)
        return x

class EfficientNetCA(EfficientNetCA_Gen):
    def __init__(self):
        super(EfficientNetCA, self).__init__(model='efficientnet-b4', width=0)
