import cv2
from src.pose import extract_pose
from src.ball import detect_ball
from src.trajectory import fit_trajectory, release_angle
from src.biomechanics import elbow_angle
from src.analysis import analyze_shot
from src.utils import draw_points
from config.config import MIN_BALL_POINTS

VIDEO = "data/basketball_shot.mp4"

cap = cv2.VideoCapture(VIDEO)
ball_points = []
elbow_angles = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    joints = extract_pose(frame)
    ball = detect_ball(frame)

    if ball:
        ball_points.append(ball)
        cv2.circle(frame, ball, 5, (0, 0, 255), -1)

    if joints:
        draw_points(frame, joints.values(), (255, 0, 0))
        elbow_angles.append(elbow_angle(joints))

    cv2.imshow("Sports Performance AI", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# ====== POST ANALYSIS ======

if len(ball_points) >= MIN_BALL_POINTS:
    a, b, c = fit_trajectory(ball_points)
    shot_angle = release_angle(a, b)
    avg_elbow = sum(elbow_angles) / len(elbow_angles)

    report = analyze_shot(shot_angle, avg_elbow)

    print("\n=== PERFORMANCE REPORT ===")
    for k, v in report.items():
        if k == "feedback":
            print("Feedback:")
            for f in v:
                print("•", f)
        else:
            print(f"{k}: {v}")
else:
    print("Not enough data for analysis.")