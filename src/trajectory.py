import numpy as np
from scipy.optimize import curve_fit

def parabola(x, a, b, c):
    return a * x**2 + b * x + c

def fit_trajectory(points):
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    params, _ = curve_fit(parabola, xs, ys)
    return params

def release_angle_from_points(ball_points, k=5):
    if len(ball_points) < k:
        return None
    p1 = np.array(ball_points[0])
    p2 = np.array(ball_points[k])
    dx = p2[0] - p1[0]
    dy = p1[1] - p2[1]
    return np.degrees(np.arctan2(dy, dx))