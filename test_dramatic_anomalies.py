# test_dramatic_anomalies.py
import numpy as np
from src.core.gan_model import GANAnomalyDetector

print("Testing Dramatic Anomalies")
print("=" * 50)

# Load model
model = GANAnomalyDetector()
model.load_model("models")
print("Model loaded successfully")

# Create normal frame
normal = np.full((64, 64), 0.5, dtype=np.float32)
normal_score = model.detect_anomalies(normal[np.newaxis, ..., np.newaxis])[0]
print(f"Normal frame score: {normal_score:.6f}")

# Create dramatic anomalies
anomalies = []

# 1. Half white
half_white = normal.copy()
half_white[:, 32:] = 1.0  # Right half white
anomalies.append(("HALF WHITE", half_white))

# 2. Cross pattern
cross = normal.copy()
cross[30:34, :] = 0.9  # Horizontal line
cross[:, 30:34] = 0.9  # Vertical line
anomalies.append(("CROSS", cross))

# 3. Border anomaly
border = normal.copy()
border[:5, :] = 0.8    # Top border
border[-5:, :] = 0.8   # Bottom border
border[:, :5] = 0.8    # Left border
border[:, -5:] = 0.8   # Right border
anomalies.append(("BORDER", border))

# 4. Random noise
random_noise = np.random.rand(64, 64).astype(np.float32)
anomalies.append(("RANDOM NOISE", random_noise))

# 5. Completely white
all_white = np.ones((64, 64), dtype=np.float32)
anomalies.append(("ALL WHITE", all_white))

# 6. Completely black
all_black = np.zeros((64, 64), dtype=np.float32)
anomalies.append(("ALL BLACK", all_black))

print("\nTesting dramatic anomalies:")
print("-" * 60)

for name, anomaly in anomalies:
    score = model.detect_anomalies(anomaly[np.newaxis, ..., np.newaxis])[0]
    diff = score - normal_score
    status = "ANOMALY" if diff > 0.01 else "Normal"
    print(f"{name:15}: {score:.6f} (Diff: {diff:+.6f}) {status}")

print("\n" + "=" * 50)
print("Test completed! Look for significant score differences.")
print("If scores are similar, your model needs more training!")