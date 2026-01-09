import cv2
import numpy as np
from config.config import BALL_HSV_LOWER, BALL_HSV_UPPER

def detect_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BALL_HSV_LOWER, BALL_HSV_UPPER)
    mask = cv2.GaussianBlur(mask, (7, 7), 1.5)

    circles = cv2.HoughCircles(
        mask, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=40,
        param1=50, param2=18,
        minRadius=5, maxRadius=30
    )

    if circles is not None:
        x, y, _ = circles[0][0]
        return int(x), int(y)

    return None