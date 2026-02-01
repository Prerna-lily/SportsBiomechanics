import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

KEYPOINT_CSV = "outputs/keypoints.csv"

df = pd.read_csv(KEYPOINT_CSV)

def get_joint(frame, idx):
    row = df[(df.frame == frame) & (df.landmark == idx)]
    return np.array([row.x.values[0], row.y.values[0]])

def angle(a, b, c):
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

frames = sorted(df.frame.unique())

knee_angles = []
shoulder_hip_sep = []
wrist_speed = []

prev_wrist = None

for f in frames:
    try:
        # Front leg (assume LEFT leg)
        hip = get_joint(f, 23)
        knee = get_joint(f, 25)
        ankle = get_joint(f, 27)
        knee_angles.append(angle(hip, knee, ankle))

        # Shoulder–hip separation
        l_sh, r_sh = get_joint(f, 11), get_joint(f, 12)
        l_hp, r_hp = get_joint(f, 23), get_joint(f, 24)

        shoulder_vec = r_sh - l_sh
        hip_vec = r_hp - l_hp

        sep = angle(
            shoulder_vec,
            np.array([0, 0]),
            hip_vec
        )
        shoulder_hip_sep.append(sep)

        # Wrist velocity (bowling arm – RIGHT)
        wrist = get_joint(f, 16)
        if prev_wrist is not None:
            wrist_speed.append(np.linalg.norm(wrist - prev_wrist))
        prev_wrist = wrist

    except:
        continue

# Plot results
plt.figure()
plt.plot(knee_angles)
plt.title("Front Knee Flexion Angle")
plt.xlabel("Frame")
plt.ylabel("Degrees")
plt.show()

plt.figure()
plt.plot(shoulder_hip_sep)
plt.title("Shoulder–Hip Separation")
plt.xlabel("Frame")
plt.ylabel("Degrees")
plt.show()

plt.figure()
plt.plot(wrist_speed)
plt.title("Bowling Arm Wrist Speed")
plt.xlabel("Frame")
plt.ylabel("Pixel/frame")
plt.show()

print("Knee ROM:", max(knee_angles) - min(knee_angles))
print("Peak Shoulder–Hip Separation:", max(shoulder_hip_sep))
print("Wrist Smoothness (Variance):", np.var(wrist_speed))
