def calculate_priority(confidence, area_percent, drift_distance_km):
    """
    Transparent scoring model for prototype prioritisation.
    It is not a scientifically validated cleanup decision model.
    """

    confidence_score = confidence * 60.0
    area_score = min(area_percent * 10.0, 25.0)
    movement_score = min(drift_distance_km / 10.0, 15.0)

    score = min(100.0, confidence_score + area_score + movement_score)

    if score >= 70:
        level = "HIGH PRIORITY — field verification recommended"
    elif score >= 40:
        level = "MEDIUM PRIORITY — monitor and verify"
    else:
        level = "LOW PRIORITY — continue monitoring"

    return {
        "score": score,
        "level": level
    }
