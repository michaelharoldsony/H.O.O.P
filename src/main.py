import cv2
import os
import numpy as np

from src.pose import extract_pose
from src.ball import detect_ball, set_ball_hsv
from src.biomechanics import elbow_angle
from src.analysis import analyze_shot
from src.utils import draw_points
from config.config import MIN_BALL_POINTS
from src.trajectory import fit_trajectory, release_angle_from_points

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO = os.path.join(BASE_DIR, "data", "second vid.mp4")

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
        print(f"Hoop click: {(x, y)}")

cv2.namedWindow("Calibrate Hoop")
cv2.setMouseCallback("Calibrate Hoop", hoop_mouse_callback)

print("\nHOOP CALIBRATION")
print("1️⃣ Click CENTER of hoop")
print("2️⃣ Click EDGE of hoop")
print("Press any key after 2 clicks\n")

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

print(f"✅ Rim center: {rim_center}, Rim radius: {rim_radius}")

# =====================================================
#               BALL COLOR CALIBRATION
# =====================================================
ball_click = None

def ball_mouse_callback(event, x, y, flags, param):
    global ball_click
    if event == cv2.EVENT_LBUTTONDOWN:
        ball_click = (x, y)
        print(f"Ball click: {ball_click}")

cv2.namedWindow("Calibrate Ball")
cv2.setMouseCallback("Calibrate Ball", ball_mouse_callback)

print("\nBALL CALIBRATION")
print("🎯 Click ON the basketball")
print("Press any key after clicking\n")

while True:
    temp = first_frame.copy()
    if ball_click:
        cv2.circle(temp, ball_click, 6, (0, 255, 0), -1)
    cv2.imshow("Calibrate Ball", temp)

    if cv2.waitKey(1) != -1 and ball_click:
        break

cv2.destroyWindow("Calibrate Ball")

# --- HSV sampling around click ---
x, y = ball_click
hsv = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)
patch = hsv[max(y-5,0):y+5, max(x-5,0):x+5]
h_mean, s_mean, v_mean = np.mean(patch.reshape(-1, 3), axis=0)

lower = np.array([
    max(h_mean - 10, 0),
    max(s_mean - 60, 50),
    max(v_mean - 60, 50)
], dtype=np.uint8)

upper = np.array([
    min(h_mean + 10, 179),
    255,
    255
], dtype=np.uint8)

set_ball_hsv(lower, upper)

print(f"✅ Ball HSV calibrated")
print(f"LOWER = {lower}")
print(f"UPPER = {upper}")

# =====================================================
#                MAIN PROCESSING LOOP
# =====================================================
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

ball_points = []
elbow_angles = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    joints = extract_pose(frame)
    ball = detect_ball(frame, debug=True)

    # Draw hoop
    cv2.circle(frame, rim_center, rim_radius, (0, 255, 0), 2)

    # Ball tracking
    if ball:
        x, y, r = ball
        ball_points.append((x, y))

        # Draw detected circle
        cv2.circle(frame, (x, y), r, (0, 0, 255), 2)

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x - r, y - r),
            (x + r, y + r),
            (255, 0, 0),
            2
        )
        cv2.putText(
            frame,
            "BALL",
            (x - r, y - r - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    # Label
    cv2.putText(
        frame,
        "BALL",
        (x - r, y - r - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )

    # Pose + biomechanics
    if joints:
        draw_points(frame, joints.values(), (255, 0, 0))
        elbow_angles.append(elbow_angle(joints))

    cv2.imshow("Sports Performance AI", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# =====================================================
#                  POST ANALYSIS
# =====================================================
if len(ball_points) < MIN_BALL_POINTS:
    print("❌ Not enough ball data for analysis.")
    exit()

# --- Trajectory ---
a, b, c = fit_trajectory(ball_points)
shot_angle = release_angle_from_points(ball_points[:10], k=5)
# -------- ANGLE NORMALIZATION --------
# Convert angle to [0, 180)
shot_angle = (shot_angle + 360) % 360

# Mirror angles > 180
if shot_angle > 180:
    shot_angle = 360 - shot_angle

# Final physical constraint: only upward shots
if shot_angle > 90:
    shot_angle = 180 - shot_angle
avg_elbow = sum(elbow_angles) / len(elbow_angles)

# --- Hoop intersection ---
x_rim = rim_center[0]
y_pred = a * x_rim**2 + b * x_rim + c
miss_distance = y_pred - rim_center[1]

# ---------------- ROBUST MAKE / MISS LOGIC ----------------

# Condition 1: trajectory intersects rim (original)
traj_hit = abs(miss_distance) <= rim_radius

# Condition 2: last detected ball position is inside rim
last_x, last_y = ball_points[-1]
dist_last = np.linalg.norm(
    np.array([last_x, last_y]) - np.array(rim_center)
)
last_inside_rim = dist_last <= rim_radius

# Condition 3: ball disappeared above rim (fell through)
ball_above_rim_then_gone = (
    last_y < rim_center[1] and   # last seen ABOVE rim center
    abs(last_x - rim_center[0]) <= rim_radius
)

# Final decision
MADE = traj_hit or last_inside_rim or ball_above_rim_then_gone

print("\n================ SHOT RESULT ================")
print("🏀 SHOT MADE ✅" if MADE else "❌ SHOT MISSED")

if not MADE:
    miss_type = "LONG" if miss_distance > 0 else "SHORT"
    angle_correction = np.degrees(np.arctan(abs(miss_distance) / 100))
    print(f"Miss type: {miss_type}")
    print(f"Suggested angle correction: ~{angle_correction:.2f}°")
else:
    print("Reason: Ball entered rim or disappeared through hoop")

# --- Biomechanics + physics report ---
report = analyze_shot(shot_angle, avg_elbow)

print("\n============ PERFORMANCE REPORT =============")
for k, v in report.items():
    if k == "feedback":
        print("Feedback:")
        for f in v:
            print("•", f)
    else:
        print(f"{k}: {v}")