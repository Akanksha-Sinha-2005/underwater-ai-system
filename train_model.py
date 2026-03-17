from ultralytics import YOLO
model=YOLO("runs/detect/train4/weights/best.pt")

def detect_objects(image_path):
    results=model(image_path)
    detections=[]
    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls=int(box.cls.item())
            conf=float(box.conf.item())

            detections.append({
                "object":model.names[cls],
                "confidence":round(conf,2)
            })
    return detections