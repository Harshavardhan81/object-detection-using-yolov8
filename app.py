# gui_main.py
import tkinter as tk
from tkinter import filedialog, messagebox
from threading import Thread
from detect import detect_objects_in_video
from detectimg import detect_objects_in_image
from webcam import detect_objects_from_webcam  # NEW ✅

def detect_video():
    file_path = filedialog.askopenfilename(
        title="Select a Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
    )
    if file_path:
        Thread(target=detect_objects_in_video, args=(file_path,), daemon=True).start()

def detect_image():
    file_path = filedialog.askopenfilename(
        title="Select an Image File",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if file_path:
        Thread(target=detect_objects_in_image, args=(file_path,), daemon=True).start()

def detect_webcam():
    Thread(target=detect_objects_from_webcam, daemon=True).start()

# Main GUI
app = tk.Tk()
app.title("YOLOv8 Object Detection")
app.geometry("1280x720")
app.resizable(False, False)

tk.Label(app, text="Select an option below:", font=("Arial", 14)).pack(pady=20)

tk.Button(app, text="Detect from Image", font=("Arial", 12), width=20, command=detect_image).pack(pady=10)
tk.Button(app, text="Detect from Video", font=("Arial", 12), width=20, command=detect_video).pack(pady=10)
tk.Button(app, text="Detect from Webcam", font=("Arial", 12), width=25, command=detect_webcam).pack(pady=10)
tk.Button(app, text="Exit", font=("Arial", 12), width=20, command=app.quit).pack(pady=20)

app.mainloop()
