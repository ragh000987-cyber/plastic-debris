# Marine Plastic AI/ML Detection and Mapping

This is a hackathon prototype for:

Satellite image
-> AI plastic detection
-> geolocation
-> confidence
-> ocean/wind drift prediction
-> cleanup priority
-> interactive GIS map

## Run

Install:

pip install -r requirements.txt

Start:

streamlit run app.py

Open the Streamlit URL shown in the terminal.

## Real AI detection

The app looks for:

weights/best.pt

If it exists, Ultralytics YOLO is used.

The model must be trained on floating marine plastic/debris imagery.

If best.pt is missing, the app runs a clearly-labelled DEMO fallback so the dashboard can still be demonstrated. The fallback is not a real plastic classifier.

## Training

Prepare a labelled dataset:

dataset/
  images/
    train/
    val/
  labels/
    train/
    val/

Create:

dataset/data.yaml

using the example file.

Then run:

python train.py

After training, copy:

runs/marine_plastic/weights/best.pt

to:

weights/best.pt

## Important geolocation note

The sample code uses a simple demo conversion from image pixels to latitude/longitude. For the actual project, satellite images must be properly georeferenced using their metadata/geospatial transform.

## Ocean data

The current prototype accepts current and wind values manually.

For the final hackathon version, connect the drift module to suitable INCOIS/oceanographic and meteorological datasets. A validated particle-drift model can then replace the simplified vector model.

## Suggested final architecture

Satellite imagery
  -> preprocessing
  -> YOLO/segmentation model
  -> detections
  -> georeferencing
  -> INCOIS/current/wind data
  -> drift model
  -> hotspot prediction
  -> priority layer
  -> Streamlit GIS dashboard
