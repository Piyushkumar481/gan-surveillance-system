# test_train.py
import numpy as np
import tensorflow as tf
from src.core.gan_model import GANAnomalyDetector

print("Starting GAN model test...")
print("=" * 50)

# Create simple test data (8 fake frames)
print("1. Creating test data...")
test_data = np.random.rand(8, 64, 64, 1).astype(np.float32)
print(f"   Test data shape: {test_data.shape}")

# Initialize model
print("2. Initializing GAN model...")
model = GANAnomalyDetector()
print("Model initialized successfully")

# Test single training step
print("3. Testing training step...")
try:
    d_loss, g_loss, recon_loss = model.train_step(test_data)
    print(f"Training step successful!")
    print(f"   D_loss: {d_loss:.4f}, G_loss: {g_loss:.4f}, Recon: {recon_loss:.4f}")
except Exception as e:
    print(f"Training failed: {e}")
    exit()

# Test anomaly detection
print("4. Testing anomaly detection...")
try:
    anomaly_scores = model.detect_anomalies(test_data)
    print(f"Anomaly detection successful!")
    print(f"   Anomaly scores: {anomaly_scores}")
except Exception as e:
    print(f"Anomaly detection failed: {e}")
    exit()

# Test model saving
print("5. Testing model saving...")
try:
    model.save_model("test_models")
    print("Models saved successfully!")
    print("   Check 'test_models' folder")
except Exception as e:
    print(f"Model saving failed: {e}")
    exit()

print("=" * 50)
print("All tests passed! Your GAN model is working correctly!")
print("Now you can run the full training with:")
print("python main.py train --video-path data/sample_video.avi --epochs 3 --batch-size 4")