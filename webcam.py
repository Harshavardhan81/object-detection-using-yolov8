# webcam_detection.py
import cv2
from ultralytics import YOLO

#model = YOLO("yolov8n.pt")
models = {
      "BeeButterfly": YOLO("bee&butterfly.pt"),
    "AntInsect": YOLO("ant+in.pt"),
    "YOLOv8n":YOLO("yolov8n.pt"),
    "fanswitch":YOLO("switchfan.pt")
}

def detect_objects_from_webcam():
    cap = cv2.VideoCapture(0)  # 0 for default webcam

    if not cap.isOpened():
        print("Error: Cannot access the webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for model_name, model in models.items():
            results = model(frame)

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[class_id]
                xyxy = box.xyxy[0].cpu().numpy().astype(int)

                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Webcam Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
