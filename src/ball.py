import cv2
import numpy as np

BALL_HSV_LOWER = None
BALL_HSV_UPPER = None

def set_ball_hsv(lower, upper):
    global BALL_HSV_LOWER, BALL_HSV_UPPER
    BALL_HSV_LOWER = lower
    BALL_HSV_UPPER = upper

def detect_ball(frame, debug=False):
    if BALL_HSV_LOWER is None or BALL_HSV_UPPER is None:
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BALL_HSV_LOWER, BALL_HSV_UPPER)
    mask = cv2.GaussianBlur(mask, (7, 7), 1.5)

    if debug:
        cv2.imshow("Ball Mask", mask)

    circles = cv2.HoughCircles(
        mask,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=50,
        param2=15,
        minRadius=5,
        maxRadius=40
    )

    if circles is not None:
        x, y, r = circles[0][0]
        return int(x), int(y), int(r)

    return None