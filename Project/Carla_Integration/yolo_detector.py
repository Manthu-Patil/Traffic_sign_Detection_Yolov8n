"""
Brief:
    Integrate a custom-trained YOLOv8 traffic sign detection model
    with the CARLA simulator for real-time perception and visualization.

Description:
    - Spawns a vehicle and RGB camera in CARLA
    - Streams camera frames to a YOLOv8 model
    - Performs real-time traffic sign detection
    - Visualizes bounding boxes and confidence scores using Pygame
    - Allows basic vehicle control and camera view switching

Simulation:
    CARLA Simulator v0.9.10 (Town05)

Model:
    YOLOv8 (Ultralytics) - custom-trained traffic sign detector

Controls:
    Arrow Keys : Throttle / Brake / Steering
    C          : Cycle camera views
    R          : Toggle reverse mode
    ESC        : Exit simulation

Author:
    Mahantesh Patil

Date:
    2025-12-19
"""

import sys
import time
import numpy as np
import pygame
import cv2
from ultralytics import YOLO

# =====================================================
# ---------------- CARLA PATH SETUP --------------------
# =====================================================
sys.path.append(
    r"C:\Users\manth\Downloads\CARLA_0.9.10\WindowsNoEditor\PythonAPI\carla"
)
sys.path.append(
    r"C:\Users\manth\Downloads\CARLA_0.9.10\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.10-py3.7-win-amd64.egg"
)
import carla

# =====================================================
# ---------------- YOLO SETUP --------------------------
# =====================================================
MODEL_PATH = r"runs/detect/train5/weights/best.pt"
CONF_THRESHOLD = 0.5
model = YOLO(MODEL_PATH)
print("[INFO] YOLO model loaded")

# =====================================================
# ---------------- GLOBAL FRAME ------------------------
# =====================================================
latest_frame = None

def camera_callback(image):
    global latest_frame
    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((image.height, image.width, 4))
    latest_frame = img[:, :, :3]

# =====================================================
# ---------------- MAIN FUNCTION -----------------------
# =====================================================
def main():

    pygame.init()
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("CARLA + YOLOv8")
    clock = pygame.time.Clock()

    # ---------- CARLA ----------
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.load_world("Town05")
    time.sleep(1)

    bp_lib = world.get_blueprint_library()

    # ---------- VEHICLE ----------
    vehicle_bp = bp_lib.find("vehicle.tesla.model3")
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print("[INFO] Vehicle spawned")

    # ---------- CAMERA ----------
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(WIDTH))
    cam_bp.set_attribute("image_size_y", str(HEIGHT))
    cam_bp.set_attribute("fov", "100")

    # CAMERA VIEWS
    camera_views = [
        carla.Transform(carla.Location(x=1.5, z=2.4)),
        carla.Transform(carla.Location(x=2.2, z=1.2), carla.Rotation(pitch=-5)),
        carla.Transform(carla.Location(x=-6, z=3), carla.Rotation(pitch=-15))
    ]
    view_idx = 0

    cam = world.spawn_actor(cam_bp, camera_views[view_idx], attach_to=vehicle)
    cam.listen(camera_callback)
    print("[INFO] Camera started")

    # ---------- CONTROL ----------
    control = carla.VehicleControl()
    running = True
    reverse_mode = False
    # =================================================
    # ---------------- MAIN LOOP ----------------------
    # =================================================
    while running:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                if event.key == pygame.K_c:
                    view_idx = (view_idx + 1) % len(camera_views)
                    cam.set_transform(camera_views[view_idx])

                if event.key == pygame.K_r:
                    reverse_mode = not reverse_mode
                    control.reverse = reverse_mode
                    print(f"[INFO] Reverse {'ON' if reverse_mode else 'OFF'}")

        keys = pygame.key.get_pressed()

        # ---------- ARROW CONTROLS ----------
        control.throttle = 1.0 if keys[pygame.K_UP] else 0.0
        control.brake    = 1.0 if keys[pygame.K_DOWN] else 0.0
        control.steer    = (
            -0.5 if keys[pygame.K_LEFT] else
             0.5 if keys[pygame.K_RIGHT] else 0.0
        )
        vehicle.apply_control(control)

        if latest_frame is None:
            continue

        frame = latest_frame.copy()

        # =================================================
        # ---------------- YOLO INFERENCE -----------------
        # =================================================
        results = model(frame, device="cpu", verbose=False)

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls[0])]

                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # ---------- DISPLAY ----------
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
        screen.blit(pygame.surfarray.make_surface(frame_rgb), (0, 0))
        pygame.display.flip()

    # =================================================
    # ---------------- CLEANUP -------------------------
    # =================================================
    cam.stop()
    cam.destroy()
    vehicle.destroy()
    pygame.quit()

# =====================================================
# ---------------- ENTRY POINT -------------------------
# =====================================================
if __name__ == "__main__":
    main()
