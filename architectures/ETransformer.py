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

from einops import rearrange
from einops.layers.torch import Rearrange

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
EfficientNet Transformer
"""

class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=self.ih*self.iw, w=self.ih*self.iw)
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0.):
        super().__init__()
        hidden_dim = int(inp * 4)

        self.ih, self.iw = image_size
        self.downsample = downsample

        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.ff = FeedForward(oup, hidden_dim, dropout)

        self.attn = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(inp, self.attn, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

        self.ff = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(oup, self.ff, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

    def forward(self, x):
        if self.downsample:
            x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
        else:
            x = x + self.attn(x)
        x = x + self.ff(x)
        return x

# class EfficientNetChannelAtt(EfficientNet):
#     def init_att(self, model: str, width: int = 0):
#         """
#         Initialize attention
#         :param model: efficientnet-bx, x \in {0,..,7}
#         :param depth: attention width
#         :return:
#         """
#         if model == 'efficientnet-b4':
#             self.att_block_idx = 9
#             if width == 0:
#                 self.attconv0 = nn.Conv2d(kernel_size=1, in_channels=24, out_channels=1)
#                 self.attconv1 = nn.Conv2d(kernel_size=1, in_channels=32, out_channels=1)
#                 self.attconv2 = nn.Conv2d(kernel_size=1, in_channels=56, out_channels=1)
#             else:
#                 attconv_layers = []
#                 for i in range(width):
#                     attconv_layers.append(
#                         ('conv{:d}'.format(i), nn.Conv2d(kernel_size=3, padding=1, in_channels=56, out_channels=56)))
#                     attconv_layers.append(
#                         ('relu{:d}'.format(i), nn.ReLU(inplace=True)))
#                 attconv_layers.append(('conv_out', nn.Conv2d(kernel_size=1, in_channels=56, out_channels=1)))
#                 self.attconv = nn.Sequential(OrderedDict(attconv_layers))
#         else:
#             raise ValueError('Model not valid: {}'.format(model))

#     def extract_features(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         EfficientNet-b4 

#         torch.Size([1, 24, 112, 112])
#         depth = 2

#         torch.Size([1, 32, 56, 56])
#         depth = 4
        
#         torch.Size([1, 56, 28, 28])
#         depth = 3

#         torch.Size([1, 112, ?, ?])
#         depth = 6

#         torch.Size([1, 160, 14, 14])
#         depth = 6

#         torch.Size([1, 272, ?, ?])
#         depth = 8

#         torch.Size([1, 448, 7, 7])
#         depth = 2

#         """


#         # Stem
#         x = self._swish(self._bn0(self._conv_stem(x)))

#         # Blocks
#         for idx, block in enumerate(self._blocks):
#             drop_connect_rate = self._global_params.drop_connect_rate
#             if drop_connect_rate:
#                 drop_connect_rate *= float(idx) / len(self._blocks)
#             x = block(x, drop_connect_rate=drop_connect_rate)

#         # Head
#         x = self._swish(self._bn1(self._conv_head(x)))

#         return x

class EfficientNetTransformerGen(FeatureExtractor):
    def __init__(self, model: str):
        super(EfficientNetTransformerGen, self).__init__()

        self.efficientnet = EfficientNet.from_pretrained(model)
        self.transformer = Transformer(448, 768, (7, 7))
        self.classifier = nn.Linear(self.efficientnet._conv_head.out_channels, 1)
        del self.efficientnet._fc

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.efficientnet.extract_features(x)
        x = self.transformer(x)
        print(x.shape)
        # x = self.efficientnet._avg_pooling(x)
        # x = x.flatten(start_dim=1)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.efficientnet._dropout(x)
        x = self.classifier(x)
        return x


class EfficientNetTransformer(EfficientNetTransformerGen):
    def __init__(self):
        super(EfficientNetTransformer, self).__init__(model='efficientnet-b4')



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
