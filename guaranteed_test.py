# guaranteed_test.py
import os
import numpy as np
from src.core.gan_model import GANAnomalyDetector

print("=== GUARANTEED MODEL TEST ===")

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)
print("Models folder created/verified")

# Create simple dummy data
print("Creating test data...")
dummy_data = np.random.rand(8, 64, 64, 1).astype(np.float32)
print(f"Test data shape: {dummy_data.shape}")

# Initialize model
print("Initializing GAN model...")
model = GANAnomalyDetector()
print("Model initialized")

# Test one training step
print("Testing training step...")
d_loss, g_loss, recon_loss = model.train_step(dummy_data[:4])
print(f"Training works! D_loss: {d_loss:.4f}, G_loss: {g_loss:.4f}")

# Save model
print("Saving model...")
model.save_model("models")
print("Models saved successfully!")

# Verify files were created
print("Checking saved files...")
if os.path.exists("models/generator.keras"):
    print("generator.keras found!")
else:
    print("generator.keras missing!")

if os.path.exists("models/discriminator.keras"):
    print("discriminator.keras found!")
else:
    print("discriminator.keras missing!")

print("Test completed! Check models/ folder now:")