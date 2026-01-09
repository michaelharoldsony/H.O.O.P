import cv2

def draw_elbow_skeleton(frame, joints, color=(255, 0, 0), thickness=3):
    """
    Draws a skeleton for the shooting arm:
    Shoulder -> Elbow -> Wrist
    """

    required = ["right_shoulder", "right_elbow", "right_wrist"]

    for k in required:
        if k not in joints:
            return  # Do nothing if pose is incomplete

    shoulder = tuple(map(int, joints["right_shoulder"]))
    elbow = tuple(map(int, joints["right_elbow"]))
    wrist = tuple(map(int, joints["right_wrist"]))

    # Draw bones
    cv2.line(frame, shoulder, elbow, color, thickness)
    cv2.line(frame, elbow, wrist, color, thickness)

    # Draw joints (filled circles)
    cv2.circle(frame, shoulder, 6, color, -1)
    cv2.circle(frame, elbow, 6, color, -1)
    cv2.circle(frame, wrist, 6, color, -1)