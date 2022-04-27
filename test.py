import cv2 
from PIL import Image, ImageChops
from blazeface import FaceExtractor, BlazeFace

facedet = BlazeFace().to(0)
facedet.load_weights("blazeface/blazeface.pth")
facedet.load_anchors("blazeface/anchors.npy")
face_extractor = FaceExtractor(facedet=facedet)

im_fake = Image.open('False07.jpg')
im_fake_faces = face_extractor.process_image(img=im_fake)

im_fake_face = im_fake_faces['faces'][0]

pil_image = Image.fromarray(im_fake_face)
pil_image.save("face07.jpg")