import cv2

def generate_heatmap(original,enhanced):
    img1=cv2.imread(original)
    img2=cv2.imread(enhanced)
    img2=cv2.resize(img2,(img1.shape[1],img1.shape[0]))
    diff=cv2.absdiff(img1,img2)
    heatmap=cv2.applyColorMap(diff,cv2.COLORMAP_JET)
    return heatmap