# test_anomalies.py
import cv2
import numpy as np
import os
from src.core.gan_model import GANAnomalyDetector
from src.core.video_processor import VideoProcessor

print("Testing anomaly detection with artificial anomalies...")

# Load your trained model
model = GANAnomalyDetector()
model.load_model("models")
print("Model loaded successfully")

# Create test frames with artificial anomalies
def create_test_frames():
    # Create a simple normal frame (gray background)
    normal_frame = np.full((64, 64), 0.5, dtype=np.float32)  # 50% gray
    
    # Create anomalous frames
    anomalies = []
    
    # 1. Bright spot anomaly
    bright_spot = normal_frame.copy()
    bright_spot[20:30, 20:30] = 1.0  # White square
    anomalies.append(("Bright Spot", bright_spot))
    
    # 2. Dark spot anomaly  
    dark_spot = normal_frame.copy()
    dark_spot[40:50, 40:50] = 0.0  # Black square
    anomalies.append(("Dark Spot", dark_spot))
    
    # 3. Line anomaly
    line_anomaly = normal_frame.copy()
    line_anomaly[30:35, 10:54] = 0.9  # White line
    anomalies.append(("Line", line_anomaly))
    
    # 4. Checkerboard pattern (very anomalous)
    checkerboard = normal_frame.copy()
    for i in range(0, 64, 8):
        for j in range(0, 64, 8):
            if (i//8 + j//8) % 2 == 0:
                checkerboard[i:i+8, j:j+8] = 0.8
    anomalies.append(("Checkerboard", checkerboard))
    
    return normal_frame, anomalies

# Create test data
normal_frame, test_anomalies = create_test_frames()

# Test normal frame
normal_score = model.detect_anomalies(normal_frame[np.newaxis, ..., np.newaxis])[0]
print(f"\nNormal frame score: {normal_score:.6f}")

# Test each anomaly
print("\nTesting artificial anomalies:")
for name, anomaly_frame in test_anomalies:
    score = model.detect_anomalies(anomaly_frame[np.newaxis, ..., np.newaxis])[0]
    difference = score - normal_score
    status = "ANOMALY" if difference > 0.02 else "Normal"
    print(f"{name:15}: {score:.6f} (Diff: {difference:+.6f}) {status}")

# Test with actual video frames (if available)
print("\nTesting with actual video frames...")
try:
    if os.path.exists("training_frames"):
        frames = VideoProcessor.load_frames("training_frames")[:5]  # First 5 frames
        scores = model.detect_anomalies(frames)
        print(f"Video frame scores: {[f'{s:.6f}' for s in scores]}")
        print(f"Average score: {np.mean(scores):.6f}")
except Exception as e:
    print(f"Could not load video frames: {e}")

print("\nTest completed! Your model should detect significant score differences for anomalies.")