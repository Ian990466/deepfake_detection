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
EfficientNet
"""


class EfficientNetGen(FeatureExtractor):
    def __init__(self, model: str):
        super(EfficientNetGen, self).__init__()

        self.efficientnet = EfficientNet.from_pretrained(model)
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


class EfficientNetB4(EfficientNetGen):
    def __init__(self):
        super(EfficientNetB4, self).__init__(model='efficientnet-b4')


"""
EfficientNet CBAM 
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

class EfficientNetChannelAtt(EfficientNet):
    def init_att(self, model: str, width: int = 0):
        """
        Initialize attention
        :param model: efficientnet-bx, x \in {0,..,7}
        :param depth: attention width
        :return:
        """
        if model == 'efficientnet-b4':
            self.att_block_idx = 9
            if width == 0:
                self.attconv = nn.Conv2d(kernel_size=1, in_channels=56, out_channels=1)
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

        self.ca = ChannelAttention(56)
        self.sa = SpatialAttention()

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self._swish(self._bn0(self._conv_stem(x)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            # print(len(self._blocks))
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx == self.att_block_idx:
                cbam = self.ca(x) * x  
                cbam = self.sa(x) * cbam
                att = torch.sigmoid(self.attconv(cbam))
                x = x * att

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

class EfficientNetCBAM_Gen(FeatureExtractor):
    def __init__(self, model: str, width: int):
        super(EfficientNetCBAM_Gen, self).__init__()

        self.efficientnet = EfficientNetChannelAtt.from_pretrained(model)
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

class EfficientNetCBAM(EfficientNetCBAM_Gen):
    def __init__(self):
        super(EfficientNetCBAM, self).__init__(model='efficientnet-b4', width=0)



"""
EfficientNetAutoAtt
"""


class EfficientNetAutoAtt(EfficientNet):
    def init_att(self, model: str, width: int):
        """
        Initialize attention
        :param model: efficientnet-bx, x \in {0,..,7}
        :param depth: attention width
        :return:
        """
        if model == 'efficientnet-b4':
            self.att_block_idx = 9
            if width == 0:
                self.attconv = nn.Conv2d(kernel_size=1, in_channels=56, out_channels=1)
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

    def get_attention(self, x: torch.Tensor) -> torch.Tensor:

        # Placeholder
        att = None

        # Stem
        x = self._swish(self._bn0(self._conv_stem(x)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx == self.att_block_idx:
                att = torch.sigmoid(self.attconv(x))
                break

        return att

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self._swish(self._bn0(self._conv_stem(x)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx == self.att_block_idx:
                att = torch.sigmoid(self.attconv(x))
                x = x * att

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x


class EfficientNetGenAutoAtt(FeatureExtractor):
    def __init__(self, model: str, width: int):
        super(EfficientNetGenAutoAtt, self).__init__()

        self.efficientnet = EfficientNetAutoAtt.from_pretrained(model)
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

    def get_attention(self, x: torch.Tensor) -> torch.Tensor:
        return self.efficientnet.get_attention(x)


class EfficientNetAutoAttB4(EfficientNetGenAutoAtt):
    def __init__(self):
        super(EfficientNetAutoAttB4, self).__init__(model='efficientnet-b4', width=0)


"""
Xception
"""


class Xception(FeatureExtractor):
    def __init__(self):
        super(Xception, self).__init__()
        self.xception = externals.xception()
        self.xception.last_linear = nn.Linear(2048, 1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.xception.features(x)
        x = nn.ReLU(inplace=True)(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.xception.forward(x)


"""
Siamese tuning
"""


class SiameseTuning(FeatureExtractor):
    def __init__(self, feat_ext: FeatureExtractor, num_feat: int, lastonly: bool = True):
        super(SiameseTuning, self).__init__()
        self.feat_ext = feat_ext()
        if not hasattr(self.feat_ext, 'features'):
            raise NotImplementedError('The provided feature extractor needs to provide a features() method')
        self.lastonly = lastonly
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features=num_feat),
            nn.Linear(in_features=num_feat, out_features=1),
        )

    def features(self, x):
        x = self.feat_ext.features(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lastonly:
            with torch.no_grad():
                x = self.features(x)
        else:
            x = self.features(x)
        x = self.classifier(x)
        return x

    def get_trainable_parameters(self):
        if self.lastonly:
            return self.classifier.parameters()
        else:
            return self.parameters()


class EfficientNetB4ST(SiameseTuning):
    def __init__(self):
        super(EfficientNetB4ST, self).__init__(feat_ext=EfficientNetB4, num_feat=1792, lastonly=True)


class EfficientNetAutoAttB4ST(SiameseTuning):
    def __init__(self):
        super(EfficientNetAutoAttB4ST, self).__init__(feat_ext=EfficientNetAutoAttB4, num_feat=1792, lastonly=True)


class XceptionST(SiameseTuning):
    def __init__(self):
        super(XceptionST, self).__init__(feat_ext=Xception, num_feat=2048, lastonly=True)