import cv2
import torch
from models.unet_model import UNet
import numpy as np

model=UNet()
model.eval()

def enhance_image(path):
    img=cv2.imread(path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(256,256))
    img=img/255.0
    img=np.transpose(img,(2,0,1))
    tensor=torch.from_numpy(img).float().unsqueeze(0)
    with torch.no_grad():
        output=model(tensor)

    output=output.squeeze().numpy()
    output=np.transpose(output,(1,2,0))
    output=(output*255).astype(np.uint8)

    original=cv2.imread(path)
    original=cv2.resize(original,(256,256))

    lab=cv2.cvtColor(original,cv2.COLOR_BGR2LAB)
    l,a,b=cv2.split(lab)
    clahe=cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
    l=clahe.apply(l)
    lab=cv2.merge((l,a,b))
    enhanced=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
    return enhanced
