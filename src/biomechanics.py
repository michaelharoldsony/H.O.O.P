import numpy as np

def angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def elbow_angle(joints, side="right"):
    shoulder = joints[f"{side}_shoulder"]
    elbow = joints[f"{side}_elbow"]
    wrist = joints[f"{side}_wrist"]
    return angle(shoulder, elbow, wrist)