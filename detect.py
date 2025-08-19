# video_detection.py
import cv2
from ultralytics import YOLO

#model = YOLO("yolov8m.pt")
models = {

   # "YOLOv8m":YOLO("yolov8m.pt"),
    "BeeButterfly": YOLO("bee&butterfly.pt"),
    "AntInsect": YOLO("ant+in.pt"),
    "YOLOv8n": YOLO("yolov8n.pt"),
    "fanswitch":YOLO("switchfan.pt"),


}
def detect_objects_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video.")
        return

    output_path = "output_video.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
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
                cv2.putText(frame, f"{label} {conf * 100:.1f}%", (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)
        cv2.imshow("YOLOv8 Video Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
