import cv2
import os
import numpy as np
import torch
import joblib
import sys

from src.pose import extract_pose
from src.ball import detect_ball, set_ball_hsv
from src.biomechanics import elbow_angle
from src.utils import draw_elbow_skeleton
from src.trajectory import fit_trajectory, release_angle_from_points
from config.config import MIN_BALL_POINTS

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ================= VIDEO PATH =================
if len(sys.argv) > 1:
    VIDEO = sys.argv[1]
else:
    VIDEO = os.path.join(BASE_DIR, "data", "second vid.mp4")


MODEL_PATH = os.path.join(BASE_DIR, "training", "shot_model.pt")
SCALER_PATH = os.path.join(BASE_DIR, "training", "scaler.pkl")

# ================= LOAD ML MODEL (ANGLE ONLY) =================
class ShotNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(5, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU()
        )
        self.make_head = torch.nn.Linear(16, 1)   # compatibility
        self.angle_head = torch.nn.Linear(16, 1)

    def forward(self, x):
        f = self.fc(x)
        return self.make_head(f), self.angle_head(f)

model = ShotNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
scaler = joblib.load(SCALER_PATH)

# ================= VIDEO LOAD =================
cap = cv2.VideoCapture(VIDEO)
ret, first_frame = cap.read()
if not ret:
    print("❌ Failed to read video")
    exit()

# ================= HOOP CALIBRATION =================
rim_clicks = []

def hoop_mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        rim_clicks.append((x, y))

cv2.namedWindow("Calibrate Hoop")
cv2.setMouseCallback("Calibrate Hoop", hoop_mouse_callback)

while True:
    tmp = first_frame.copy()
    for c in rim_clicks:
        cv2.circle(tmp, c, 5, (0, 0, 255), -1)
    cv2.imshow("Calibrate Hoop", tmp)
    if cv2.waitKey(1) != -1 and len(rim_clicks) >= 2:
        break

cv2.destroyWindow("Calibrate Hoop")
rim_center = rim_clicks[0]
rim_radius = int(np.linalg.norm(
    np.array(rim_clicks[0]) - np.array(rim_clicks[1])
))

# ================= BALL COLOR CALIBRATION =================
ball_click = None

def ball_mouse_callback(event, x, y, flags, param):
    global ball_click
    if event == cv2.EVENT_LBUTTONDOWN:
        ball_click = (x, y)

cv2.namedWindow("Calibrate Ball")
cv2.setMouseCallback("Calibrate Ball", ball_mouse_callback)

while True:
    tmp = first_frame.copy()
    if ball_click:
        cv2.circle(tmp, ball_click, 6, (0, 255, 0), -1)
    cv2.imshow("Calibrate Ball", tmp)
    if cv2.waitKey(1) != -1 and ball_click:
        break

cv2.destroyWindow("Calibrate Ball")

x, y = ball_click
hsv = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)
patch = hsv[max(y-5,0):y+5, max(x-5,0):x+5]
h, s, v = np.mean(patch.reshape(-1,3), axis=0)

lower = np.array([max(h-20,0), max(s-90,40), max(v-90,40)], dtype=np.uint8)
upper = np.array([min(h+20,179), 255, 255], dtype=np.uint8)
set_ball_hsv(lower, upper)

# ================= MAIN LOOP =================
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
        ball_points.append((bx, by))

    if len(ball_points) > 1:
        for i in range(1, len(ball_points)):
            cv2.line(frame, ball_points[i-1], ball_points[i], (0, 0, 255), 2)

    if joints:
        draw_elbow_skeleton(frame, joints)
        elbow_angles.append(elbow_angle(joints))

    cv2.imshow("Sports Performance AI", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# ================= POST ANALYSIS =================
if len(ball_points) < MIN_BALL_POINTS:
    print("❌ Not enough trajectory data")
    exit()

a, b, c = fit_trajectory(ball_points)
shot_angle = abs(release_angle_from_points(ball_points[:10], k=5))
avg_elbow = sum(elbow_angles) / len(elbow_angles)

# ================= MAKE / MISS (TEMPORAL) =================
rim_x, rim_y = rim_center
above, below = False, False

for (x, y) in ball_points:
    if abs(x - rim_x) <= 1.5 * rim_radius:
        if y < rim_y:
            above = True
        elif y > rim_y:
            below = True

MADE = above and below

# ================= ML ANGLE CORRECTION =================
apex_height = min(p[1] for p in ball_points)
distance = abs(rim_x - ball_points[0][0])
speed_est = np.mean([
    np.linalg.norm(np.array(ball_points[i+1]) - np.array(ball_points[i]))
    for i in range(len(ball_points)-1)
])

features = np.array([[
    np.clip(shot_angle, 30, 70),
    np.clip(speed_est / 10, 6, 12),
    2.1,
    np.clip(distance / 100, 4.5, 7.5),
    np.clip(apex_height / 100, 3, 5)
]])

features = scaler.transform(features)
features = torch.tensor(features, dtype=torch.float32)

with torch.no_grad():
    _, angle_delta = model(features)
    angle_delta = angle_delta.item()

# ================= COACHING FEEDBACK =================
feedback = []

if angle_delta > 3:
    feedback.append("Reduce shot arc slightly")
elif angle_delta < -3:
    feedback.append("Increase shot arc")

if avg_elbow < 135:
    feedback.append("Extend elbow more during release")
elif avg_elbow > 165:
    feedback.append("Avoid overextending elbow")

if MADE:
    feedback.append("Good mechanics — maintain this form")

feedback_text = " | ".join(feedback) if feedback else "Balanced shot mechanics"

# ================= FINAL VISUALIZATION =================
final_frame = last_frame.copy()

for i in range(1, len(ball_points)):
    cv2.line(final_frame, ball_points[i-1], ball_points[i], (0, 0, 255), 2)

overlay_text = [
    f"RESULT: {'MADE' if MADE else 'MISSED'}",
    f"Release angle: {shot_angle:.1f} deg",
    f"Optimal angle: {shot_angle + angle_delta:.1f} deg",
    f"Angle error: {angle_delta:+.1f} deg",
    f"Elbow angle: {avg_elbow:.1f} deg",
    f"COACHING: {feedback_text}"
]

# ---------- SEMI-TRANSPARENT BOX ----------
box_x, box_y = 10, 10
box_w = 460
box_h = 30 + len(overlay_text) * 28

overlay = final_frame.copy()
cv2.rectangle(
    overlay,
    (box_x, box_y),
    (box_x + box_w, box_y + box_h),
    (180, 255, 255),  # light yellow
    -1
)

final_frame = cv2.addWeighted(overlay, 0.55, final_frame, 0.45, 0)

# ---------- TEXT ----------
for i, text in enumerate(overlay_text):
    cv2.putText(
        final_frame,
        text,
        (box_x + 12, box_y + 28 + i * 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2
    )

cv2.imshow("Final Shot Analysis", final_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()