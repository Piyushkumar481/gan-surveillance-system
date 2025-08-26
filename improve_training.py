# improve_training.py
import numpy as np
from src.core.gan_model import GANAnomalyDetector
from src.core.video_processor import VideoProcessor

print("Improving model training...")

# Load frames
frames = VideoProcessor.load_frames("training_frames")
print(f"Loaded {len(frames)} training frames")

# Create improved model with better architecture
class ImprovedGANAnomalyDetector(GANAnomalyDetector):
    def _build_generator(self):
        """Improved U-Net generator with more capacity"""
        inputs = layers.Input(shape=self.img_shape)
        
        # Encoder
        x = layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Conv2D(256, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # Decoder
        x = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2DTranspose(1, 4, strides=2, padding='same')(x)
        outputs = layers.Activation('sigmoid')(x)
        
        return Model(inputs, outputs, name='generator')

# Train improved model
print("Training improved model...")
improved_model = ImprovedGANAnomalyDetector()

# Train for a few more epochs
for epoch in range(3):
    epoch_loss = 0
    for i in range(0, len(frames), 8):
        batch = frames[i:i+8]
        d_loss, g_loss, recon_loss = improved_model.train_step(batch)
        epoch_loss += recon_loss.numpy()
    
    avg_loss = epoch_loss / (len(frames) / 8)
    print(f"Epoch {epoch+1}: Avg Recon Loss: {avg_loss:.6f}")

# Save improved model
improved_model.save_model("improved_models")
print("Improved model saved!")

# Test improved model
normal_frame = np.full((64, 64), 0.5, dtype=np.float32)
test_anomaly = normal_frame.copy()
test_anomaly[20:30, 20:30] = 1.0  # Bright spot

old_score = model.detect_anomalies(normal_frame[np.newaxis, ..., np.newaxis])[0]
new_score = improved_model.detect_anomalies(normal_frame[np.newaxis, ..., np.newaxis])[0]
anomaly_score = improved_model.detect_anomalies(test_anomaly[np.newaxis, ..., np.newaxis])[0]

print(f"\nComparison:")
print(f"Old model normal: {old_score:.6f}")
print(f"New model normal: {new_score:.6f}")
print(f"New model anomaly: {anomaly_score:.6f}")
print(f"Improvement: {((old_score - new_score) / old_score * 100):.1f}%")