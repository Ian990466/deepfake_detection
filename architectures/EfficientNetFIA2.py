import torch
from efficientnet_pytorch import EfficientNet
from torch import nn as nn
from torch.nn import functional as F
from torchvision import transforms

"""
Feature Extractor
"""

class FeatureExtractor(nn.Module):
    def features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_trainable_parameters(self):
        return self.parameters()

    @staticmethod
    def get_normalizer():
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


"""
EfficientNetFIA
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
        return self.sigmoid(maxout + avgout)

class SpatialIntegrationAttention(nn.Module):
    def __init__(self, in_dim ,kernel_size=7):
        super(SpatialIntegrationAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.attconv = nn.Conv2d(in_channels= in_dim, out_channels=1, kernel_size= 1)
        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        fm = self.attconv(x)
        x = torch.cat([fm, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class EfficientNetAtt(EfficientNet):
    def init_att(self):
        # super(EfficientNetAtt, self).__init__()
        self.ca0 = ChannelAttention(112)
        self.ca1 = ChannelAttention(160)
        self.ca2 = ChannelAttention(272)
        self.ca3 = ChannelAttention(448)

        self.sa0 = SpatialIntegrationAttention(24)
        self.sa1 = SpatialIntegrationAttention(32)
        self.sa2 = SpatialIntegrationAttention(56)
        self.sa3 = SpatialIntegrationAttention(112)
        self.sa4 = SpatialIntegrationAttention(160)
        self.sa5 = SpatialIntegrationAttention(272)
        self.sa6 = SpatialIntegrationAttention(448)

    def extract_features(self, x):
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
                x = self.sa0(x) * x
            elif idx == 5:
                x = self.sa1(x) * x
            elif idx == 9:
                x = self.sa2(x) * x
            elif idx == 15:
                x = self.ca0(x) * x  
                x = self.sa3(x) * x
            elif idx == 21:
                x = self.ca1(x) * x  
                x = self.sa4(x) * x
            elif idx == 29:
                x = self.ca2(x) * x  
            # elif idx == 31:
            #     x = self.ca3(x) * x  
            #     x = self.sa6(x) * x

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        
        return x

class EfficientNetFIA2(FeatureExtractor):
    def __init__(self):
        super(EfficientNetFIA2, self).__init__()

        self.efficientnet = EfficientNetAtt.from_pretrained("efficientnet-b4")
        self.efficientnet.init_att()
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
