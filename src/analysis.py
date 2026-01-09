from config.config import OPTIMAL_RELEASE_ANGLE, ANGLE_TOLERANCE

def analyze_shot(player_angle, elbow_angle_val):
    feedback = []

    diff = player_angle - OPTIMAL_RELEASE_ANGLE

    if diff < -ANGLE_TOLERANCE:
        feedback.append("Shot too flat → increase release angle.")
    elif diff > ANGLE_TOLERANCE:
        feedback.append("Shot too high → reduce arc.")
    else:
        feedback.append("Release angle near optimal.")

    if elbow_angle_val < 145:
        feedback.append("Elbow extension insufficient.")

    return {
        "player_angle": player_angle,
        "optimal_angle": OPTIMAL_RELEASE_ANGLE,
        "error": diff,
        "elbow_angle": elbow_angle_val,
        "feedback": feedback
    }