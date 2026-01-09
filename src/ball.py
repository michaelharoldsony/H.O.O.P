import cv2
import numpy as np

# Optional ML fallback
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ---------------- HSV STATE ----------------
BALL_HSV_LOWER = None
BALL_HSV_UPPER = None

last_ball_pos = None
last_radius = None

# ---------------- YOLO MODEL ----------------
if YOLO_AVAILABLE:
    yolo_model = YOLO("yolov8n.pt")
else:
    yolo_model = None

# -------------------------------------------------
# CALIBRATION
# -------------------------------------------------
def set_ball_hsv(lower, upper):
    global BALL_HSV_LOWER, BALL_HSV_UPPER
    BALL_HSV_LOWER = lower
    BALL_HSV_UPPER = upper

# -------------------------------------------------
# ADAPTIVE HSV UPDATE
# -------------------------------------------------
def adapt_hsv(frame, x, y):
    global BALL_HSV_LOWER, BALL_HSV_UPPER

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w, _ = frame.shape

    x1 = max(0, x - 6)
    y1 = max(0, y - 6)
    x2 = min(w, x + 6)
    y2 = min(h, y + 6)

    patch = hsv[y1:y2, x1:x2]
    if patch.size == 0:
        return

    h_mean, s_mean, v_mean = np.mean(patch.reshape(-1, 3), axis=0)

    BALL_HSV_LOWER = np.array([
        max(h_mean - 25, 0),
        max(s_mean - 100, 40),
        max(v_mean - 100, 40)
    ], dtype=np.uint8)

    BALL_HSV_UPPER = np.array([
        min(h_mean + 25, 179),
        255,
        255
    ], dtype=np.uint8)

# -------------------------------------------------
# HSV DETECTOR
# -------------------------------------------------
def hsv_detect(frame):
    global last_ball_pos, last_radius

    if BALL_HSV_LOWER is None:
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BALL_HSV_LOWER, BALL_HSV_UPPER)

    mask = cv2.medianBlur(mask, 5)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return None

    best = None
    best_score = 0

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 50:
            continue

        (x, y), r = cv2.minEnclosingCircle(c)
        x, y, r = int(x), int(y), int(r)

        if r < 4 or r > 60:
            continue

        # Circularity
        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue

        circularity = 4 * np.pi * area / (peri * peri)
        if circularity < 0.45:
            continue

        # Motion consistency
        if last_ball_pos is not None:
            jump = np.linalg.norm([x - last_ball_pos[0], y - last_ball_pos[1]])
            if jump > 80:
                continue

        score = circularity * area
        if score > best_score:
            best_score = score
            best = (x, y, r, 0.9)

    if best:
        last_ball_pos = (best[0], best[1])
        last_radius = best[2]
        adapt_hsv(frame, best[0], best[1])
        return best

    return None

# -------------------------------------------------
# YOLO FALLBACK
# -------------------------------------------------
def yolo_detect(frame):
    if not YOLO_AVAILABLE:
        return None

    results = yolo_model(frame, conf=0.2, iou=0.4, verbose=False)
    if not results:
        return None

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # Sports ball class (YOLO COCO = 32)
            if cls == 32 and conf > 0.25:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                r = max(x2 - x1, y2 - y1) // 2
                return (cx, cy, r, conf)

    return None

# -------------------------------------------------
# HYBRID ENTRY POINT
# -------------------------------------------------
def detect_ball(frame, debug=False):
    """
    Returns: (x, y, r, confidence) or None
    """

    # 1️⃣ Try HSV first
    hsv_result = hsv_detect(frame)
    if hsv_result:
        return hsv_result

    # 2️⃣ Fallback to YOLO
    yolo_result = yolo_detect(frame)
    if yolo_result:
        return yolo_result

    return None