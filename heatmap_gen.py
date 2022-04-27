from matplotlib.pyplot import get
from isplutils.heatmap import ConvHeatMap

import cv2
import torch
import numpy as np  
from PIL import Image
from collections import OrderedDict
from isplutils import utils

from architectures import fornet
from architectures.fornet import FeatureExtractor

# from architectures import EBAttentionv2
# from architectures.EBAttentionv2 import FeatureExtractor

# from architectures import EfficientNetFIA
# from architectures.EfficientNetFIA import FeatureExtractor

def save_imgs(cam, file_name: str, output_path= "./"):
    cv2.imwrite(f"{output_path}{file_name}.png", cam)

def main():
    net_class = getattr(fornet, "Xception")
    # net_class = getattr(EBAttentionv2, "EfficientNetCA")
    # net_class = getattr(EfficientNetFIA, "EfficientNetFIA")
    print(net_class)

    # load model
    print('Loading model...')
    state_tmp = torch.load("./weights/binclass/net-Xception_traindb-ff-c23-720-140-140_face-scale_size-224_seed-41/last.pth")
    if 'net' not in state_tmp.keys():
        state = OrderedDict({'net': OrderedDict()})
        [state['net'].update({'model.{}'.format(k): v}) for k, v in state_tmp.items()]
    else:
        state = state_tmp

    net: FeatureExtractor = net_class().eval().to(0)

    incomp_keys = net.load_state_dict(state['net'], strict=True)
    print(incomp_keys)
    print('Model loaded!')

    imgs = "/home/ian/Desktop/Heatmap_Result/face_extraction/face05.jpg"
    face_policy = 'scale'
    face_size = 224

    im_fake = Image.open(imgs)
    im_fake = np.array(im_fake)
    transf = utils.get_transformer(face_policy, face_size, net.get_normalizer(), train=False)
    faces_t = torch.stack( [ transf(image=im)['image'] for im in [im_fake] ] )
    
    with torch.no_grad():
        faces_pred = torch.sigmoid(net(faces_t.to(0))).cpu().numpy().flatten()

    print('Score for FAKE face: {:.4f}'.format(faces_pred[0]))

    chm = ConvHeatMap(
        imgs_path= imgs, method= "gradcam", 
        model= net,
        # target_layers= [net.efficientnet._blocks[-1]])
        target_layers= [net.xception.block12])
    cam = chm.get_cam()
    
    save_imgs(cam= cam, file_name= "Xception0531cam")

if __name__ == "__main__":
    main()