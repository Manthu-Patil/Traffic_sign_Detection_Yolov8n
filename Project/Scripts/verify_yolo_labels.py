"""
Brief:
    Randomly sample images and visualize YOLO-format bounding boxes
    to verify correctness of GTSDB-to-YOLO annotation conversion.

Input:
    - Image files (.jpg / .png)
    - YOLO annotation files (.txt)

YOLO Format:
    class_id x_center y_center width height

Author:
    Mahantesh Patil

Date:
    2025-11-22
"""

import cv2
import os
import random

IMG_DIR = "Dataset/images/train"
LBL_DIR = "Dataset/labels/train"

images = os.listdir(IMG_DIR)
sample_imgs = random.sample(images, 20)

for img_name in sample_imgs:
    img_path = os.path.join(IMG_DIR, img_name)
    lbl_path = os.path.join(LBL_DIR, img_name.replace(".jpg", ".txt"))

    img = cv2.imread(img_path)
    h, w, _ = img.shape

    if not os.path.exists(lbl_path):
        continue

    with open(lbl_path) as f:
        for line in f:
            cls, xc, yc, bw, bh = map(float, line.split())

            x1 = int((xc - bw/2) * w)
            y1 = int((yc - bh/2) * h)
            x2 = int((xc + bw/2) * w)
            y2 = int((yc + bh/2) * h)

            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, str(int(cls)), (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("YOLO Label Check", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
