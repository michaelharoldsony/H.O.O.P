import numpy as np
import pandas as pd

GRAVITY = 9.81
RIM_HEIGHT = 3.05  # meters
BALL_RADIUS = 0.12

def simulate_shot(angle_deg, speed, release_height, distance):
    angle = np.radians(angle_deg)

    t = distance / (speed * np.cos(angle))
    y = release_height + speed * np.sin(angle) * t - 0.5 * GRAVITY * t**2

    made = abs(y - RIM_HEIGHT) <= BALL_RADIUS
    miss_type = "MADE" if made else ("SHORT" if y < RIM_HEIGHT else "LONG")

    apex_height = release_height + (speed**2 * np.sin(angle)**2) / (2 * GRAVITY)

    return made, miss_type, y, apex_height


def generate_dataset(n_samples=20000):
    data = []

    for _ in range(n_samples):
        angle = np.random.uniform(30, 70)
        speed = np.random.uniform(6, 12)
        release_height = np.random.uniform(1.8, 2.3)
        distance = np.random.uniform(4.5, 7.5)

        made, miss_type, y_at_rim, apex = simulate_shot(
            angle, speed, release_height, distance
        )

        noise = np.random.normal(0, 0.8)

        data.append({
            "release_angle": angle + noise,
            "release_speed": speed + noise * 0.1,
            "release_height": release_height,
            "distance": distance,
            "apex_height": apex + noise,
            "made": int(made),
            "miss_type": miss_type
        })

    return pd.DataFrame(data)


if __name__ == "__main__":
    df = generate_dataset(30000)
    df.to_csv("synthetic_basketball_shots.csv", index=False)
    print("✅ Synthetic dataset generated:", df.shape)