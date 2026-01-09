from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

KEYPOINTS = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
}

def extract_pose(frame):
    result = model(frame, verbose=False)[0]
    if result.keypoints is None:
        return None

    kp = result.keypoints.xy[0].cpu().numpy()
    joints = {name: kp[idx] for name, idx in KEYPOINTS.items()}
    return joints