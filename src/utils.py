import cv2

def draw_points(frame, points, color):
    for p in points:
        cv2.circle(frame, tuple(p.astype(int)), 4, color, -1)