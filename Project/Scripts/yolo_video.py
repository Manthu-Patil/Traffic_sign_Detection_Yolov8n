"""
Brief:
    Perform real-time traffic sign detection on a video using a
    custom-trained YOLOv8 model and visualize detected bounding boxes
    with class labels and confidence scores.

Purpose:
    - Verify trained YOLOv8 model performance on video input
    - Visually validate bounding box accuracy and class predictions
    - Save annotated output video for analysis and reporting

Model:
    YOLOv8 (Ultralytics) custom-trained traffic sign detection model

Input:
    - Video file (.mp4)
    - Trained YOLO model weights (best.pt)

Output:
    - Annotated video with bounding boxes and labels

Configurable Parameters:
    - Confidence threshold
    - Input / output video paths

Author:
    Mahantesh Patil

Date:
    2025-12-02
"""

import cv2
from ultralytics import YOLO
import os

# ---------------- CONFIG ----------------
VIDEO_PATH = r"C:\Users\manth\OneDrive\Desktop\Traffic_sign_Yolo\Video_Generation_with_Traffic_Signs.mp4"
MODEL_PATH = "runs/detect/train5/weights/best.pt"
CONF_THRESHOLD = 0.5
OUTPUT_PATH = "Traffic_sign_OP3.mp4"

# ---------------- LOAD MODEL ----------------
model = YOLO(MODEL_PATH)

# ---------------- LOAD VIDEO ----------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("[ERROR] Cannot open video")

print("[INFO] Video loaded")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"[INFO] FPS={fps}, Resolution={width}x{height}")

# ---------------- VIDEO WRITER ----------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Windows-safe
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

if not out.isOpened():
    raise RuntimeError("[ERROR] Cannot open VideoWriter")

print("[INFO] Output video writer initialized")

# ---------------- LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------- YOLO INFERENCE --------
    results = model(frame, verbose=False)

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # -------- SHOW & SAVE --------
    cv2.imshow("YOLO Video Check", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

# ---------------- CLEANUP ----------------
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"[INFO] Saved detected video to: {os.path.abspath(OUTPUT_PATH)}")
