import math

EARTH_RADIUS_KM = 6371.0

def _destination(lat, lon, distance_km, bearing_deg):
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    bearing = math.radians(bearing_deg)

    angular_distance = distance_km / EARTH_RADIUS_KM

    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_distance)
        + math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing)
    )

    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat1),
        math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2)
    )

    return math.degrees(lat2), math.degrees(lon2)


def predict_drift(
    lat,
    lon,
    current_speed,
    current_direction,
    wind_speed,
    wind_direction,
    hours
):
    """
    Simple vector-combination drift model.

    This is intentionally transparent for a hackathon prototype.
    Replace it with validated INCOIS/ocean-model data for a real system.
    """

    current_distance = current_speed * 3.6 * hours
    wind_distance = wind_speed * 0.03 * 3.6 * hours

    # Convert the two vectors to x/y components.
    c = math.radians(current_direction)
    w = math.radians(wind_direction)

    east = current_distance * math.sin(c) + wind_distance * math.sin(w)
    north = current_distance * math.cos(c) + wind_distance * math.cos(w)

    distance = math.sqrt(east**2 + north**2)

    bearing = math.degrees(math.atan2(east, north)) % 360

    new_lat, new_lon = _destination(lat, lon, distance, bearing)

    return {
        "latitude": new_lat,
        "longitude": new_lon,
        "distance_km": distance,
        "bearing": bearing
    }
