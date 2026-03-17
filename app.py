from flask import Flask,render_template,request
import os
import cv2

from models.enhancement_model import enhance_image
from models.detection_model import detect_objects
from metrics.metrics import calculate_metrics
from utils.heatmap import generate_heatmap

app=Flask(__name__)
UPLOAD_FOLDER="static/uploads"
RESULT_FOLDER="static/results"

os.makedirs(UPLOAD_FOLDER,exist_ok=True)
os.makedirs(RESULT_FOLDER,exist_ok=True)

@app.route("/",methods=["GET","POST"])
def index():
    if request.method=="POST":
        file=request.files["image"]
        filepath=os.path.join(UPLOAD_FOLDER,file.filename)
        file.save(filepath)
        enhanced=enhance_image(filepath)
        result_path=os.path.join(RESULT_FOLDER,file.filename)
        cv2.imwrite(result_path,enhanced)
        detections=detect_objects(result_path)
        psnr,ssim,uiqm=calculate_metrics(filepath,result_path)
        heatmap=generate_heatmap(filepath,result_path)
        heatmap_path=os.path.join(RESULT_FOLDER,"heatmap_"+file.filename)
        cv2.imwrite(heatmap_path,heatmap)
        return render_template("result.html",
                               original="/"+filepath,
                               enhanced=result_path,
                               heatmap=heatmap_path,
                               detections=detections,
                               psnr=psnr,
                               ssim=ssim,
                               uiqm=uiqm)
    return render_template("index.html")

if __name__=="__main__":
    app.run(debug=True)