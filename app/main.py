from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from ultralytics import YOLO
import mlflow

# Initialize FastAPI app
app = FastAPI(
    title="YOLO11 Object Detection API",
    description="Real-time object detection using the latest YOLO11 model",
    version="1.0.0"
)

# Load the model globally so it only loads once when the server starts
# Using the latest YOLO11 nano model ('yolo11n.pt')
model = YOLO("yolo11n.pt")
results = model.train(
    data="your_dataset.yaml",
    epochs=50,
    imgsz=640,
    project="YOLOv8_API_Project",
    name="run_1"
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read the uploaded image file into memory
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. Run inference directly on the OpenCV image (No need to read from disk!)
    results = model(img)

    # 3. Parse the results into a JSON-friendly format
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        detections.append({
            "class": label,
            "confidence": round(conf, 2),
            "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        })

    return {
        "filename": file.filename,
        "object_count": len(detections),
        "detections": detections
    }

# To run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload