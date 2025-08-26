"""
Unit tests for GAN model functionality
"""

import pytest
import numpy as np
import tensorflow as tf
from src.core.gan_model import GANAnomalyDetector

class TestGANModel:
    @pytest.fixture
    def gan_model(self):
        """Create GAN model instance for testing"""
        return GANAnomalyDetector(img_shape=(64, 64, 1))
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample test data"""
        return np.random.rand(5, 64, 64, 1).astype(np.float32)
    
    def test_model_initialization(self, gan_model):
        """Test that model initializes correctly"""
        assert gan_model.generator is not None
        assert gan_model.discriminator is not None
        assert gan_model.generator.input_shape == (None, 64, 64, 1)
        assert gan_model.discriminator.input_shape == (None, 64, 64, 1)
    
    def test_anomaly_detection_shape(self, gan_model, sample_data):
        """Test anomaly detection returns correct shape"""
        anomalies = gan_model.detect_anomalies(sample_data)
        assert anomalies.shape == (5,)  # One score per sample
        assert np.all(anomalies >= 0)  # Scores should be non-negative
    
    def test_anomaly_detection_range(self, gan_model, sample_data):
        """Test anomaly scores are within expected range"""
        anomalies = gan_model.detect_anomalies(sample_data)
        assert np.all(anomalies <= 1.0)  # Scores should be <= 1.0
    
    def test_training_step_execution(self, gan_model, sample_data):
        """Test that training step runs without errors"""
        # Convert to TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices(sample_data).batch(2)
        
        # Should not raise exceptions
        for batch in dataset.take(1):
            losses = gan_model.train_step(batch)
            assert 'd_loss' in losses
            assert 'g_loss' in losses
            assert 'recon_loss' in losses
            assert all(isinstance(loss, tf.Tensor) for loss in losses.values())
    
    def test_model_save_load(self, gan_model, tmp_path):
        """Test model saving and loading functionality"""
        save_path = tmp_path / "test_models"
        
        # Save models
        gan_model.save_model(str(save_path))
        
        # Check files were created
        assert (save_path / "generator.keras").exists()
        assert (save_path / "discriminator.keras").exists()
        
        # Create new model and load
        new_model = GANAnomalyDetector()
        new_model.load_model(str(save_path))
        
        # Verify models were loaded
        assert new_model.generator is not None
        assert new_model.discriminator is not None