import cv2
import os
import numpy as np
import torch
import joblib

from src.pose import extract_pose
from src.ball import detect_ball, set_ball_hsv
from src.biomechanics import elbow_angle
from src.analysis import analyze_shot
from src.utils import draw_points
from config.config import MIN_BALL_POINTS
from src.trajectory import fit_trajectory, release_angle_from_points

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO = os.path.join(BASE_DIR, "data", "ball.mp4")

MODEL_PATH = os.path.join(BASE_DIR, "training", "shot_model.pt")
SCALER_PATH = os.path.join(BASE_DIR, "training", "scaler.pkl")

# ================= LOAD ML MODEL =================
class ShotNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(5, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU()
        )
        self.make_head = torch.nn.Linear(16, 1)
        self.angle_head = torch.nn.Linear(16, 1)

    def forward(self, x):
        f = self.fc(x)
        return self.make_head(f), self.angle_head(f)

model = ShotNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

scaler = joblib.load(SCALER_PATH)

# ================= TRACKING PARAMETERS =================
MIN_CONFIDENCE = 0.25
SOFT_CONFIDENCE = 0.18
BASE_MAX_JUMP = 35
MAX_LOST = 4

lost_counter = 0

# ================= VIDEO LOAD =================
cap = cv2.VideoCapture(VIDEO)
ret, first_frame = cap.read()
if not ret:
    print("❌ Failed to read video")
    exit()

# =====================================================
#                 HOOP CALIBRATION
# =====================================================
rim_clicks = []

def hoop_mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        rim_clicks.append((x, y))

cv2.namedWindow("Calibrate Hoop")
cv2.setMouseCallback("Calibrate Hoop", hoop_mouse_callback)

while True:
    temp = first_frame.copy()
    for c in rim_clicks:
        cv2.circle(temp, c, 5, (0, 0, 255), -1)
    cv2.imshow("Calibrate Hoop", temp)
    if cv2.waitKey(1) != -1 and len(rim_clicks) >= 2:
        break

cv2.destroyWindow("Calibrate Hoop")

rim_center = rim_clicks[0]
rim_radius = int(np.linalg.norm(np.array(rim_clicks[0]) - np.array(rim_clicks[1])))

# =====================================================
#               BALL COLOR CALIBRATION
# =====================================================
ball_click = None

def ball_mouse_callback(event, x, y, flags, param):
    global ball_click
    if event == cv2.EVENT_LBUTTONDOWN:
        ball_click = (x, y)

cv2.namedWindow("Calibrate Ball")
cv2.setMouseCallback("Calibrate Ball", ball_mouse_callback)

while True:
    temp = first_frame.copy()
    if ball_click:
        cv2.circle(temp, ball_click, 6, (0, 255, 0), -1)
    cv2.imshow("Calibrate Ball", temp)
    if cv2.waitKey(1) != -1 and ball_click:
        break

cv2.destroyWindow("Calibrate Ball")

x, y = ball_click
hsv = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)
patch = hsv[max(y-5,0):y+5, max(x-5,0):x+5]
h, s, v = np.mean(patch.reshape(-1, 3), axis=0)

lower = np.array([max(h-15,0), max(s-80,50), max(v-80,50)], dtype=np.uint8)
upper = np.array([min(h+15,179), 255, 255], dtype=np.uint8)
set_ball_hsv(lower, upper)

# =====================================================
#        FULL-FRAME PROJECTILE (PHYSICS)
# =====================================================
def compute_full_trajectory(p0, angle_deg, speed, frame_shape):
    h, w, _ = frame_shape
    g = 9.81
    angle = np.radians(angle_deg)

    vx = speed * np.cos(angle)
    vy = speed * np.sin(angle)

    traj = []
    for x in range(0, w, 5):
        t = (x - p0[0]) / (vx + 1e-6)
        if t < 0:
            continue
        y = p0[1] - (vy * t - 0.5 * g * t * t)
        if 0 <= y < h:
            traj.append((int(x), int(y)))
    return traj

# =====================================================
#                MAIN LOOP
# =====================================================
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

ball_points = []
elbow_angles = []
last_frame = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    last_frame = frame.copy()
    joints = extract_pose(frame)
    ball = detect_ball(frame)

    cv2.circle(frame, rim_center, rim_radius, (0, 255, 0), 2)

    if ball:
        bx, by, br, conf = ball
        max_jump = max(BASE_MAX_JUMP, 2.5 * br)

        if len(ball_points) > 0:
            px, py = ball_points[-1]
            jump = np.linalg.norm([bx - px, by - py])
        else:
            jump = 0

        if conf >= MIN_CONFIDENCE and jump <= max_jump:
            ball_points.append((bx, by))
            lost_counter = 0
        elif conf >= SOFT_CONFIDENCE and jump <= 0.6 * max_jump and lost_counter < 2:
            ball_points.append((bx, by))
            lost_counter = 0
        else:
            lost_counter += 1
    else:
        lost_counter += 1

    if len(ball_points) > 1:
        for i in range(1, len(ball_points)):
            cv2.line(frame, ball_points[i-1], ball_points[i], (0, 0, 255), 2)

    if joints:
        draw_points(frame, joints.values(), (255, 0, 0))
        elbow_angles.append(elbow_angle(joints))

    cv2.imshow("Sports Performance AI (ML)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# =====================================================
#                POST ANALYSIS
# =====================================================
a, b, c = fit_trajectory(ball_points)
shot_angle = release_angle_from_points(ball_points[:10], k=5)

shot_angle = (shot_angle + 360) % 360
if shot_angle > 180:
    shot_angle = 360 - shot_angle
if shot_angle > 90:
    shot_angle = 180 - shot_angle

avg_elbow = sum(elbow_angles) / len(elbow_angles)

apex_height = min(p[1] for p in ball_points)
distance = abs(rim_center[0] - ball_points[0][0])
speed_est = np.mean([
    np.linalg.norm(np.array(ball_points[i+1]) - np.array(ball_points[i]))
    for i in range(len(ball_points)-1)
])

# ---- NORMALIZE FEATURES ----
features = np.array([[
    np.clip(shot_angle, 30, 70),
    np.clip(speed_est / 10.0, 6, 12),
    2.1,
    np.clip(distance / 100.0, 4.5, 7.5),
    np.clip(apex_height / 100.0, 3.0, 5.0)
]])

features = scaler.transform(features)
features = torch.tensor(features, dtype=torch.float32)

with torch.no_grad():
    make_logit, angle_delta = model(features)
    make_prob = torch.sigmoid(make_logit).item()
    angle_delta = angle_delta.item()

ML_MADE = make_prob >= 0.35

recent = ball_points[-7:]
entered_physics = any(
    np.linalg.norm(np.array(p) - np.array(rim_center)) <= 1.3 * rim_radius
    for p in recent
)

MADE = entered_physics or ML_MADE

print("\n================ SHOT RESULT ================")
print("🏀 SHOT MADE ✅" if MADE else "❌ SHOT MISSED")
print(f"ML confidence: {make_prob:.2f}")
print(f"Physics detected make: {entered_physics}")

# ================= MISS ANALYSIS =================
if not MADE:
    x_rim = rim_center[0]
    y_pred = a * x_rim**2 + b * x_rim + c
    vertical_error = y_pred - rim_center[1]
    angle_error = angle_delta

    print("\n============ MISS ANALYSIS =============")
    print(f"Angle error   : {angle_error:+.2f}°")
    print(f"Vertical miss: {vertical_error:+.1f} px")
    print("Miss type    :", "LONG" if vertical_error > 0 else "SHORT")

# ================= FINAL VISUALIZATION =================
final_frame = last_frame.copy()

for i in range(1, len(ball_points)):
    cv2.line(final_frame, ball_points[i-1], ball_points[i], (0, 0, 255), 2)

ideal_traj = compute_full_trajectory(
    ball_points[0],
    shot_angle + angle_delta,
    speed_est / 10,
    final_frame.shape
)

for i in range(1, len(ideal_traj)):
    cv2.line(final_frame, ideal_traj[i-1], ideal_traj[i], (255, 0, 0), 2)

cv2.imshow("Final Shot Analysis (RED=Actual, BLUE=Ideal)", final_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()