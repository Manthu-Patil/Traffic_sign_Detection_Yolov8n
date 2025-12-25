# 🚦Traffic Sign Detection using YOLOv8 with CARLA Simulation

This project implements an end-to-end traffic sign detection pipeline using YOLOv8, starting from raw dataset preprocessing to real-time inference in the CARLA simulator.
The goal is to build and validate a practical perception module similar to those used in ADAS and autonomous driving systems.

# 📌 Project Highlights
- Custom dataset preparation from raw GTSDB format
- YOLO-format annotation conversion and validation
- YOLOv8n training on CPU
- Video-based inference and evaluation
- Real-time traffic sign detection in CARLA simulator

# 🧪 Step 1 – Dataset Preprocessing
Converted traffic sign images from .ppm format to .jpg using OpenCV to ensure compatibility with deep learning frameworks and faster image loading.

Details:
- Original dataset images were in .ppm format
- Converted .ppm images to .jpg using OpenCV
- Organized images into Train/ and Test/ folders
- Improved compatibility with YOLOv8 and computer vision pipelines

# 🏷️ Step 2 – Annotation Conversion (GTSDB → YOLO)
Converted GTSDB ground-truth annotations into YOLO format by normalizing bounding boxes and organizing images and labels into training and validation splits.

Details:
- Parsed gt.txt ground-truth annotation file
- Grouped multiple bounding boxes per image
- Split dataset into training and validation sets (80/20)
- Converted bounding boxes to YOLO normalized format
- Generated one .txt label file per image

<img width="1536" height="1024" alt="GTSDB to Yolo conversion format" src="https://github.com/user-attachments/assets/63bc7c75-d6c7-4639-89da-530b47ccc07b" />

# ✅ Step 3 – Annotation Verification

Before training, YOLO annotations were visually verified to ensure correctness.

Purpose:
- Ensure annotation conversion was correct
- Verify bounding boxes align with traffic signs
- Confirm class IDs are mapped properly
- This step avoids garbage-in → garbage-out problems.

Process:
- Randomly sampled images from training dataset
- Loaded corresponding YOLO label files
- Converted normalized YOLO coordinates back to pixel values
- Visualized bounding boxes and class IDs using OpenCV
- Verified annotation accuracy before model training

# 🧠 Step 4 – YOLOv8 Training

After annotation verification, a YOLOv8n model was trained using the Ultralytics CLI on CPU.
- Dataset Configuration (data.yaml)

Defined:
- Training image path
- Validation image path
- Number of classes
- Class names (traffic sign categories)

# Training Command Used In cmd prompt

yolo detect train \
model=yolov8n.pt \
data=data.yaml \
epochs=50 \
imgsz=640 \
batch=4 \
device=cpu \
workers=2

Training Design Choices

- YOLOv8n → lightweight, CPU-friendly, real-time capable
- Image size 640 → standard for traffic sign detection
- Batch size 4 → optimized for limited RAM (8 GB system)
- CPU training → hardware-agnostic and reproducible
- 50 epochs → balance between convergence and overfitting

![train_batch1](https://github.com/user-attachments/assets/8f3107f1-f038-4649-a1e6-d2858ca12a8a)
![val_batch2_pred](https://github.com/user-attachments/assets/5b9abc96-22e5-40bd-853a-9a690f8996b0)
![val_batch0_pred](https://github.com/user-attachments/assets/ca8f90b3-1425-4740-8d72-ef326a7dccc6)



# 📊 YOLOv8 Training Metrics Explained

During training, the following metrics were monitored to evaluate model learning, localization accuracy, and classification performance:

- Box Loss – accuracy of predicted bounding box locations
- Class Loss – accuracy of predicted class labels
- DFL Loss – precision of bounding box boundaries
- mAP@50 / mAP@50–95 – overall detection performance

Observations:

- Clear and frequent signs achieved higher performance
- Speed limit signs showed moderate performance, mainly due to:
    - Class imbalance
    - Small object size
    - Limited dataset diversity

 <img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/eb67a89d-4597-4f2a-83e6-1503e9da5f6d" />

<img width="1536" height="1024" alt="Yolov8 metrics" src="https://github.com/user-attachments/assets/d1c98f5d-02c9-4c45-8b77-185e33c82fde" />


# 🎥 Step 5 – Video Inference

Applied the trained YOLOv8 model to traffic sign video streams, generating annotated output videos with bounding boxes, labels, and confidence scores.

Details:
- Performed YOLOv8 inference on video input
- Processed frames sequentially using OpenCV
- Applied confidence-based filtering
- Visualized bounding boxes, class names, and confidence scores
- Saved annotated output videos for evaluation
Outputs
- Traffic_sign_OP1.mp4

https://github.com/user-attachments/assets/e9e1e8b7-23ee-41cf-b7c3-0788e28258a6


- Traffic_sign_OP2.mp4


https://github.com/user-attachments/assets/d73c9a43-c313-4f08-9625-e919eebe9352
- Traffic_sign_OP3.mp4


# 🚗 Step 6 – CARLA & YOLOv8 Integration

Integrated the trained YOLOv8 model with the CARLA simulator to perform real-time traffic sign detection on live camera streams from a simulated vehicle.

Details:
- Integrated YOLOv8 with CARLA simulator (v0.9.10)
- Captured RGB camera frames from simulated vehicle
- Performed real-time inference on CPU
- Visualized traffic sign detections in simulator view
- Enabled vehicle and camera control using keyboard inputs

<img width="996" height="788" alt="Carla_stopsign_detection" src="https://github.com/user-attachments/assets/f76732f8-b51d-49e7-a68e-663f091d0f78" />

This step validates the model in a realistic driving simulation environment, similar to industry ADAS workflows.

🔧 Technologies & Libraries Used

- CARLA Simulator (0.9.10) – realistic autonomous driving     simulation
- YOLOv8 (Ultralytics) – real-time object detection
- OpenCV (cv2) – image & bounding box visualization
- NumPy – image buffer and array processing
- Pygame – visualization window and vehicle control
- Python – end-to-end integration logic

# 🚀 Conclusion

This project demonstrates a complete traffic sign detection pipeline, covering:
- Dataset preparation
- Annotation conversion and validation
- YOLOv8 model training
- Video-based inference
- Simulation-based validation using CARLA
