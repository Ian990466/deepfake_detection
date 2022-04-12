import torch
from efficientnet_pytorch import EfficientNet

inputs = torch.rand(1, 3, 224, 224)
model = EfficientNet.from_pretrained('efficientnet-b4')
endpoints = model.extract_endpoints(inputs)
print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])
print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])
print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])
print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])
print(endpoints['reduction_5'].shape)  # torch.Size([1, 320, 7, 7])
print(endpoints['reduction_6'].shape)  # torch.Size([1, 1280, 7, 7])