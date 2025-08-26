"""
FastAPI server for anomaly detection API
"""
from src.api.routes import router as api_router
from ast import Invert
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from typing import List
import io
from PIL import Image

from src.core.gan_model import GANAnomalyDetector
from src.core.video_processor import VideoProcessor

app = FastAPI(title="GAN Surveillance API", version="1.0.0")

# Initialize model (in production, load from trained weights)
model = GANAnomalyDetector()

@app.get("/")
async def root():
    return {"message": "GAN Surveillance API", "status": "active"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/detect/frame")
async def detect_frame_anomaly(file: UploadFile = File(...)):
    """Process single image frame for anomalies"""
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        # Invert to grayscale
        frame = np.array(image.convert('L'))
        frame = cv2.resize(frame, (64, 64))
        frame = frame.astype(np.float32) / 255.0
        frame = frame[np.newaxis, ..., np.newaxis]  # Add batch and channel dims
        
        # Detect anomaly
        anomaly_score = model.detect_anomalies(frame)[0]
        is_anomaly = anomaly_score > 0.1
        
        return JSONResponse({
            "anomaly_score": float(anomaly_score),
            "is_anomaly": bool(is_anomaly),
            "message": "Anomaly detected" if is_anomaly else "Normal frame",
            "threshold": 0.1
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/detect/video")
async def detect_video_anomalies(file: UploadFile = File(...)):
    """Process video file for anomalies"""
    try:
        # Save uploaded video temporarily
        video_path = f"temp_{file.filename}"
        with open(video_path, "wb") as f:
            f.write(await file.read())
        
        # Extract frames
        output_dir = "temp_frames"
        frame_count = VideoProcessor.extract_frames(video_path, output_dir)
        frames = VideoProcessor.load_frames(output_dir)
        
        # Detect anomalies
        anomalies = model.detect_anomalies(frames)
        results = []
        
        for i, score in enumerate(anomalies):
            results.append({
                "frame_number": i,
                "anomaly_score": float(score),
                "is_anomaly": bool(score > 0.1)
            })
        
        # Cleanup
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)
        import os
        os.remove(video_path)
        
        return JSONResponse({
            "total_frames": frame_count,
            "anomalous_frames": sum(1 for r in results if r['is_anomaly']),
            "results": results
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)