import folium

def make_map(
    lat,
    lon,
    predicted_lat,
    predicted_lon,
    confidence,
    priority_score
):
    m = folium.Map(
        location=[lat, lon],
        zoom_start=8,
        control_scale=True
    )

    folium.Marker(
        [lat, lon],
        tooltip="Detected debris",
        popup=f"Confidence: {confidence:.1%}",
        icon=folium.Icon(icon="warning-sign")
    ).add_to(m)

    folium.Marker(
        [predicted_lat, predicted_lon],
        tooltip="Predicted future location",
        popup=f"Priority score: {priority_score:.1f}/100",
        icon=folium.Icon(icon="flag")
    ).add_to(m)

    folium.PolyLine(
        [[lat, lon], [predicted_lat, predicted_lon]],
        tooltip="Predicted drift path",
        weight=4
    ).add_to(m)

    folium.Circle(
        [lat, lon],
        radius=5000,
        tooltip="Candidate debris zone"
    ).add_to(m)

    return m._repr_html_()
