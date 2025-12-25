"""
Brief:
    Convert GTSDB / IJCNN2013 dataset images from .ppm format to .jpg format
    for easier processing and compatibility with deep learning frameworks.

Author:
    Mahantesh Patil

Date:
    2025-11-20
"""
import cv2
import os
from glob import glob

TRAIN_SRC = r"C:\Users\manth\Downloads\archive (1)\TrainIJCNN2013\TrainIJCNN2013"
TEST_SRC  = r"C:\Users\manth\Downloads\archive (1)\TestIJCNN2013\TestIJCNN2013Download"

OUT_TRAIN = "Train"
OUT_TEST  = "Test"

os.makedirs(OUT_TRAIN, exist_ok=True)
os.makedirs(OUT_TEST, exist_ok=True)

for src, out in [(TRAIN_SRC, OUT_TRAIN), (TEST_SRC, OUT_TEST)]:
    ppm_files = glob(os.path.join(src, "*.ppm"))
    print(f"Found {len(ppm_files)} images in {src}")

    for ppm in ppm_files:
        img = cv2.imread(ppm)
        name = os.path.splitext(os.path.basename(ppm))[0]
        cv2.imwrite(os.path.join(out, name + ".jpg"), img)

print("✅ Conversion completed")
