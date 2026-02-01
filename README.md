# 🏏 Pose-Based Cricket Bowling Analysis (Side-On Video)

## Objective
To simulate a real-world cricket biomechanics system that analyzes bowling technique
using a single phone-recorded side-on video and pose estimation.

The goal is to extract interpretable movement metrics rather than only visual overlays.

---

## Video Selection
- Cricket bowler delivering the ball
- Side-on / near side-on camera angle
- Phone-recorded
- Full body mostly visible

### Suitability
- Ideal for sagittal-plane motion analysis
- Clear delivery stride and follow-through
- Realistic consumer capture scenario

### Limitations
- Fast arm motion causes wrist jitter
- Occasional self-occlusion
- 2D video (no depth estimation)

---

## Pose Estimation Model
**MediaPipe Pose (BlazePose)**

### Why MediaPipe?
- Strong temporal stability
- Lightweight and fast
- Works well on mobile-quality videos
- Outputs 33 body landmarks

---

## Outputs
- Skeleton overlay video
- Frame-wise keypoints (CSV)

---

## Movement Metrics

### 1. Front Knee Flexion Angle
Measures stability and energy transfer during front-foot contact.

Why it matters:
- Excessive collapse increases injury risk
- Overly stiff leg reduces pace generation

---

### 2. Shoulder–Hip Separation
Proxy for trunk rotation and elastic energy storage.

Why it matters:
- Greater separation → higher bowling speed
- Early collapse → poor kinetic chain usage

---

### 3. Bowling Arm Smoothness
Measured via wrist velocity variance.

Why it matters:
- Smooth acceleration indicates control and repeatability
- High jitter suggests inefficiency or tracking instability

---

## Observations
- Wrist and elbow jitter during fast motion
- Partial occlusion during release phase
- Occasional left/right confusion

---

## Improvement Plan

### Model Improvements
- Temporal smoothing (LSTM / TCN)
- Bone-length consistency constraints
- Sports-specific fine-tuning

### Data Collection
- Multiple bowlers
- Different speeds and styles
- Annotated phases (foot contact, ball release)

### Dataset Split
- Train: 70% bowlers
- Validation: 15%
- Test: 15%
(Split by player, not video)

### Evaluation
- Keypoint accuracy (PCK)
- Temporal stability
- Metric repeatability
- Coach qualitative validation

---

## Future Scope
- Real-time feedback system
- Injury risk alerts
- Technique comparison with elite bowlers
- Mobile coaching assistant
