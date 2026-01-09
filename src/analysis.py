def analyze_shot(player_angle, elbow_angle_val, reference_angle):
    feedback = []

    diff = player_angle - reference_angle

    if abs(diff) < 4:
        feedback.append("Release angle is well controlled.")
    elif diff < 0:
        feedback.append("Shot slightly flat → add more arc.")
    else:
        feedback.append("Shot slightly high → flatten release.")

    if elbow_angle_val < 135:
        feedback.append("Elbow extension limited → extend arm more at release.")
    elif elbow_angle_val < 150:
        feedback.append("Elbow extension acceptable but can improve.")
    else:
        feedback.append("Good elbow extension at release.")

    return {
        "player_angle": round(player_angle, 2),
        "reference_angle": round(reference_angle, 2),
        "elbow_angle": round(elbow_angle_val, 2),
        "feedback": feedback
    }