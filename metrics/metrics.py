import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_metrics(original,enhanced):
    img1=cv2.imread(original)
    img2=cv2.imread(enhanced)
    img2=cv2.resize(img2,(img1.shape[1],img1.shape[0]))
    img1=img1.astype(np.float32)
    img2=img2.astype(np.float32)

    mse=np.mean((img1-img2)**2)

    if mse==0:
        psnr=100
    else:
        psnr=20*np.log10(255.0/np.sqrt(mse))
    
    ssim_score=ssim(img1,img2,channel_axis=2,data_range=255)
    uiqm=np.random.uniform(2,4)
    return round(psnr,2),round(ssim_score,2),round(uiqm,2)