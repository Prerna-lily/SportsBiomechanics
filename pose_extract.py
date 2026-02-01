# pose_extract.py
print("[pose_extract] Starting (MediaPipe Tasks API)")

import cv2
import csv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

VIDEO_PATH = "data/bowlingvid.mp4"
OUTPUT_VIDEO = "outputs/overlay_video.mp4"
OUTPUT_CSV = "outputs/keypoints.csv"

print(f"[pose_extract] VIDEO_PATH={VIDEO_PATH}")

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/pose_landmarker_heavy.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1
)

landmarker = PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("❌ Could not open video")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

csv_file = open(OUTPUT_CSV, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["frame", "landmark", "x", "y", "visibility"])

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect_for_video(
        mp_image,
        int(frame_id * 1000 / fps)
    )

    if result.pose_landmarks:
        for lm_id, lm in enumerate(result.pose_landmarks[0]):
            writer.writerow([
                frame_id,
                lm_id,
                lm.x,
                lm.y,
                lm.visibility
            ])

            cx, cy = int(lm.x * width), int(lm.y * height)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

    out.write(frame)
    frame_id += 1

cap.release()
out.release()
csv_file.close()
landmarker.close()

print("✅ Pose extraction complete")
