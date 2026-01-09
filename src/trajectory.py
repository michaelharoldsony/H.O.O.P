import numpy as np
from scipy.optimize import curve_fit

def parabola(x, a, b, c):
    return a * x**2 + b * x + c

def fit_trajectory(points):
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    params, _ = curve_fit(parabola, xs, ys)
    return params  # a, b, c

def release_angle(a, b):
    return np.degrees(np.arctan(b))