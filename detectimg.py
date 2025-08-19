# image_detection.py
import cv2
from ultralytics import YOLO

#model = YOLO("yolov8n.pt")
models = {

    "BeeButterfly": YOLO("bee&butterfly.pt"),
    "AntInsect": YOLO("ant+in.pt"),
    #"YOLOv8n":YOLO("yolov8n.pt"),
"fanswitch":YOLO("switchfan.pt"),

}



def detect_objects_in_image(image_path):
    image = cv2.imread(image_path)
    image =cv2.resize(image, (600,600))

    for model_name, model in models.items():
        results = model(image)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[class_id]
            xyxy = box.xyxy[0].cpu().numpy().astype(int)

            cv2.rectangle(image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf * 100:.1f}%", (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Image Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
