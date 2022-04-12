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

class EfficientNetAtt(EfficientNet):
    def init_att(self, model: str, width: int = 0):
        """
        Initialize attention
        :param model: efficientnet-bx, x \in {0,..,7}
        :param depth: attention width
        :return:
        """
        self.input_channels = [24, 32, 56, 112, 160, 272, 448]

        if model == 'efficientnet-b4':
            self.att_block_idx = 9
            if width == 0:
                self.attconv0 = nn.Conv2d(kernel_size=1, in_channels=24, out_channels=1)
                self.attconv1 = nn.Conv2d(kernel_size=1, in_channels=32, out_channels=1)
                self.attconv2 = nn.Conv2d(kernel_size=1, in_channels=56, out_channels=1)
            else:
                attconv_layers = []
                for i in range(width):
                    attconv_layers.append(
                        ('conv{:d}'.format(i), nn.Conv2d(kernel_size=3, padding=1, in_channels=56, out_channels=56)))
                    attconv_layers.append(
                        ('relu{:d}'.format(i), nn.ReLU(inplace=True)))
                attconv_layers.append(('conv_out', nn.Conv2d(kernel_size=1, in_channels=56, out_channels=1)))
                self.attconv = nn.Sequential(OrderedDict(attconv_layers))
        else:
            raise ValueError('Model not valid: {}'.format(model))
        
        self.ca0 = ChannelAttention(self.input_channels[0])
        self.ca1 = ChannelAttention(self.input_channels[1])
        self.ca2 = ChannelAttention(self.input_channels[2])
        self.ca3 = ChannelAttention(self.input_channels[3])
        self.ca4 = ChannelAttention(self.input_channels[4])
        self.ca5 = ChannelAttention(self.input_channels[5])
        self.ca6 = ChannelAttention(self.input_channels[6])


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
                x = self.ca0(x) * x
                x = self.sa(x) * x
            elif idx == 5:
                x = self.ca1(x) * x
                x = self.sa(x) * x
            elif idx == 9:
                x = self.ca2(x) * x
                x = self.sa(x) * x
            elif idx == 15:
                x = self.ca3(x) * x
                x = self.sa(x) * x
            elif idx == 21:
                x = self.ca4(x) * x
                x = self.sa(x) * x
            elif idx == 29:
                x = self.ca5(x) * x
                x = self.sa(x) * x
            elif idx == 30:
                x = self.ca6(x) * x
                x = self.sa(x) * x

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

class EfficientNetBA_Gen(FeatureExtractor):
    def __init__(self, model: str, width: int):
        super(EfficientNetBA_Gen, self).__init__()

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

class EfficientNetBA(EfficientNetBA_Gen):
    def __init__(self):
        super(EfficientNetBA, self).__init__(model='efficientnet-b4', width=0)
