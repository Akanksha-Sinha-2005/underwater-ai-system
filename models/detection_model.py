from ultralytics import YOLO
import cv2

model=YOLO("runs/detect/train4/weights/best.pt")

def detect_objects(image_path):
    results=model(image_path)
    detections=[]

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls=int(box.cls[0])
            conf=float(box.conf[0])

            if conf<0.3:
                continue

            detections.append({
                "object":model.names[cls],
                "confidence":round(conf,2)
            })
        annotated=r.plot()
        cv2.imwrite("static/results/detection.jpg",annotated)
    return detections