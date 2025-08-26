"""
GAN model architecture and training logic
Core anomaly detection engine
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from typing import Tuple, Dict, List

class GANAnomalyDetector:
    def __init__(self, img_shape: Tuple[int, int, int] = (64, 64, 1)):
        self.img_shape = img_shape
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.g_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.d_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.mse_loss = tf.keras.losses.MeanSquaredError()

    def _build_generator(self) -> Model:
        """Build U-Net style generator for frame reconstruction"""
        inputs = layers.Input(shape=self.img_shape)
        
        # Encoder
        x = layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # Decoder
        x = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2DTranspose(1, 4, strides=2, padding='same')(x)
        outputs = layers.Activation('sigmoid')(x)
        
        return Model(inputs, outputs, name='generator')

    def _build_discriminator(self) -> Model:
        """Build CNN discriminator to distinguish real/fake frames"""
        inputs = layers.Input(shape=self.img_shape)
        
        x = layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(1)(x)
        
        return Model(inputs, x, name='discriminator')

    @tf.function
    def train_step(self, real_images: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Single training step for both generator and discriminator"""
        # Train discriminator
        with tf.GradientTape() as d_tape:
            generated_images = self.generator(real_images, training=True)
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            
            d_loss_real = self.loss_fn(tf.ones_like(real_output), real_output)
            d_loss_fake = self.loss_fn(tf.zeros_like(fake_output), fake_output)
            d_loss = d_loss_real + d_loss_fake
        
        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        
        # Train generator
        with tf.GradientTape() as g_tape:
            generated_images = self.generator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            
            g_adv_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
            g_recon_loss = self.mse_loss(real_images, generated_images)
            g_loss = g_adv_loss + 10.0 * g_recon_loss
        
        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        
        return d_loss, g_loss, g_recon_loss

    def train(self, dataset: tf.data.Dataset, epochs: int = 50) -> Dict[str, List[float]]:
        """Complete training loop with progress tracking"""
        history = {'d_loss': [], 'g_loss': [], 'recon_loss': []}
        
        for epoch in range(epochs):
            epoch_d_loss = tf.keras.metrics.Mean()
            epoch_g_loss = tf.keras.metrics.Mean()
            epoch_recon_loss = tf.keras.metrics.Mean()
            
            for batch in dataset:
                d_loss, g_loss, recon_loss = self.train_step(batch)
                epoch_d_loss.update_state(d_loss)
                epoch_g_loss.update_state(g_loss)
                epoch_recon_loss.update_state(recon_loss)
            
            # Store history
            history['d_loss'].append(float(epoch_d_loss.result()))
            history['g_loss'].append(float(epoch_g_loss.result()))
            history['recon_loss'].append(float(epoch_recon_loss.result()))
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"D_loss: {epoch_d_loss.result():.4f}, "
                  f"G_loss: {epoch_g_loss.result():.4f}, "
                  f"Recon: {epoch_recon_loss.result():.4f}")
        
        return history

    def detect_anomalies(self, frames: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """Detect anomalies using reconstruction error"""
        # Ensure frames are in correct format
        if frames.dtype != np.float32:
            frames = frames.astype(np.float32) / 255.0
        
        if len(frames.shape) == 3:
            frames = frames[..., np.newaxis]
        
        # Generate reconstructions
        reconstructions = self.generator.predict(frames, verbose=0)
        
        # Calculate reconstruction errors
        errors = np.mean((frames - reconstructions) ** 2, axis=(1, 2, 3))
        return errors

    def save_model(self, save_path: str):
        """Save trained models"""
        import os
        os.makedirs(save_path, exist_ok=True)
        self.generator.save(f"{save_path}/generator.keras")
        self.discriminator.save(f"{save_path}/discriminator.keras")
        print(f"Models saved to {save_path}")

    def load_model(self, load_path: str):
        """Load pre-trained models"""
        self.generator = tf.keras.models.load_model(f"{load_path}/generator.keras")
        self.discriminator = tf.keras.models.load_model(f"{load_path}/discriminator.keras")
        print(f"Models loaded from {load_path}")

    def summary(self):
        """Print model summaries"""
        print("Generator Summary:")
        self.generator.summary()
        print("\nDiscriminator Summary:")
        self.discriminator.summary()

# Simple anomaly detector for basic use cases
class SimpleAnomalyDetector:
    @staticmethod
    def detect_anomalies(frames: np.ndarray, 
                        reconstructions: np.ndarray, 
                        threshold: float = 0.1) -> np.ndarray:
        """
        Simple anomaly detection using reconstruction error
        
        Args:
            frames: Original frames
            reconstructions: Reconstructed frames
            threshold: Anomaly detection threshold
            
        Returns:
            Array of anomaly scores
        """
        errors = np.mean((frames - reconstructions) ** 2, axis=(1, 2, 3))
        return errors
    