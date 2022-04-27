from efficientnet_pytorch import EfficientNet
from torch import nn as nn

class EfficientNetAtt(EfficientNet):
    def __init__(self):
        self.a = "a"

class EfficientNetCA_Gen():
    def __init__(self):
        #  super(EfficientNetCA_Gen, self).__init__()

        self.efficientnet = EfficientNetAtt.from_pretrained("efficientnet-b4")

# test = EfficientNetCA_Gen()
# print(test.efficientnet._blocks[-1])

e = EfficientNetAtt.from_pretrained("efficientnet-b4")

# a = EfficientNet.from_pretrained("efficientnet-b4")
# print(a._blocks[-1])