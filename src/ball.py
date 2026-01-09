import cv2
import numpy as np

# These will be set by calibration
BALL_HSV_LOWER = None
BALL_HSV_UPPER = None

def set_ball_hsv(lower, upper):
    global BALL_HSV_LOWER, BALL_HSV_UPPER
    BALL_HSV_LOWER = lower
    BALL_HSV_UPPER = upper


def detect_ball(frame, debug=False):
    """
    Detect basketball using HSV + contour filtering.
    Returns (x, y, r, confidence) or None
    """

    if BALL_HSV_LOWER is None or BALL_HSV_UPPER is None:
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BALL_HSV_LOWER, BALL_HSV_UPPER)

    # Clean mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    # Pick largest contour (most likely ball)
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)

    if area < 50:
        return None

    ((x, y), radius) = cv2.minEnclosingCircle(c)

    if radius < 5 or radius > 40:
        return None

    # --- CONFIDENCE MEASURE ---
    circle_area = np.pi * radius * radius
    confidence = min(1.0, area / (circle_area + 1e-6))

    if debug:
        print(f"[BALL] r={radius:.1f}, area={area:.1f}, conf={confidence:.2f}")

    return int(x), int(y), int(radius), confidence