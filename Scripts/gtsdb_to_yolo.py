"""
Brief:
    Convert GTSDB (German Traffic Sign Detection Benchmark) annotations
    into YOLO format (class_id, x_center, y_center, width, height).
    All bounding box values are normalized between 0 and 1.

Dataset:
    German Traffic Sign Detection Benchmark (GTSDB)

Input Format:
    left, top, right, bottom, class_id

Output Format (YOLO):
    class_id x_center y_center width height

Conversion Formula:
    x_center = (x1 + x2) / (2 * W)
    y_center = (y1 + y2) / (2 * H)
    width    = (x2 - x1) / W
    height   = (y2 - y1) / H

Author:
    Mahantesh Patil

Date:
    2025-11-20
"""

import os
import cv2
from collections import defaultdict
from sklearn.model_selection import train_test_split

GT_FILE = "C:/Users/manth/Downloads/archive (1)/gt.txt"
IMG_SRC = "Train"

OUT_IMG = "Dataset/images"
OUT_LBL = "Dataset/labels"

os.makedirs(f"{OUT_IMG}/train", exist_ok=True)
os.makedirs(f"{OUT_IMG}/val", exist_ok=True)
os.makedirs(f"{OUT_LBL}/train", exist_ok=True)
os.makedirs(f"{OUT_LBL}/val", exist_ok=True)

annotations = defaultdict(list)

with open(GT_FILE, "r") as f:
    for line in f:
        img_path, left, top, right, bottom, cls = line.strip().split(";")
        img_name = os.path.basename(img_path).replace(".ppm", ".jpg")
        annotations[img_name].append(
            (int(left), int(top), int(right), int(bottom), int(cls))
        )

images = list(annotations.keys())
train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

def process(split, img_list):
    for img_name in img_list:
        src_img = os.path.join(IMG_SRC, img_name)
        img = cv2.imread(src_img)

        if img is None:
            print(f"❌ Image not found: {src_img}")
            continue

        h, w, _ = img.shape

        dst_img = f"{OUT_IMG}/{split}/{img_name}"
        dst_lbl = f"{OUT_LBL}/{split}/{img_name.replace('.jpg', '.txt')}"

        cv2.imwrite(dst_img, img)

        with open(dst_lbl, "w") as f:
            for l, t, r, b, c in annotations[img_name]:
                xc = ((l + r) / 2) / w
                yc = ((t + b) / 2) / h
                bw = (r - l) / w
                bh = (b - t) / h
                f.write(f"{c} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

process("train", train_imgs)
process("val", val_imgs)

print("✅ GTSDB → YOLO annotation conversion completed")

