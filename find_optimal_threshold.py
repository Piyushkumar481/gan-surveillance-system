# find_optimal_threshold.py
import numpy as np
from src.core.gan_model import GANAnomalyDetector
from src.core.video_processor import VideoProcessor

print("Finding Optimal Threshold")
print("=" * 50)

# Load model and frames
model = GANAnomalyDetector()
model.load_model("models")

# Load training frames to understand normal distribution
try:
    frames = VideoProcessor.load_frames("training_frames")
    errors = model.detect_anomalies(frames)
    print(f"Analyzed {len(errors)} training frames")
except:
    # If no training frames, create synthetic normal data
    print(" No training frames found, using synthetic data")
    normal_frames = np.full((20, 64, 64, 1), 0.5, dtype=np.float32)
    errors = model.detect_anomalies(normal_frames)

# Calculate statistics
mean_error = np.mean(errors)
std_error = np.std(errors)
max_error = np.max(errors)
min_error = np.min(errors)

print(f"\nError Statistics:")
print(f"Mean: {mean_error:.6f}")
print(f"Std: {std_error:.6f}")
print(f"Min: {min_error:.6f}")
print(f"Max: {max_error:.6f}")
print(f"95th percentile: {np.percentile(errors, 95):.6f}")

print(f"\nRecommended Thresholds:")
print(f"Sensitive (mean + 1σ): {mean_error + std_error:.6f}")
print(f"Balanced (mean + 2σ): {mean_error + 2*std_error:.6f}")
print(f"Strict (mean + 3σ): {mean_error + 3*std_error:.6f}")
print(f"Very Strict (95th %ile): {np.percentile(errors, 95):.6f}")

print(f"\nBased on your earlier results (~0.042 avg), try:")
print(f"python main.py detect --video-path data/sample_video.avi --model-dir models --threshold 0.055")
print(f"python main.py detect --video-path data/sample_video.avi --model-dir models --threshold 0.06")