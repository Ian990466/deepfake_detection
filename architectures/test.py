import torch
from g_mlp_vision import gMLPVision

model = gMLPVision(
    image_size = 224,
    patch_size = 16,
    dim = 512,
    depth = 4,
    channels= 3,
    num_classes= 1
)

img = torch.randn(32, 3, 224, 224)
logits = model(img) # (1, 1000)
print(logits.shape)