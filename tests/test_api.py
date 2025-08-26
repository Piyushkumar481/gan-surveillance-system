"""
Unit tests for API endpoints
"""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app
import numpy as np
from PIL import Image
import io

class TestAPI:
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["model_loaded"] == True
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        
        assert response.status_code == 200
        assert "GAN Surveillance API" in response.json()["message"]
    
    def test_detect_frame_anomaly(self, client):
        """Test frame anomaly detection endpoint"""
        # Create test image
        test_image = np.random.rand(100, 100, 3) * 255
        test_image = test_image.astype(np.uint8)
        pil_image = Image.fromarray(test_image)
        
        # Save to bytes
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Make request
        files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
        response = client.post("/detect/frame", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "anomaly_score" in data
        assert "is_anomaly" in data
        assert "message" in data
        assert "threshold" in data
        assert 0 <= data["anomaly_score"] <= 1
    
    def test_detect_frame_invalid_file(self, client):
        """Test frame detection with invalid file"""
        files = {'file': ('test.txt', b'invalid content', 'text/plain')}
        response = client.post("/detect/frame", files=files)
        
        assert response.status_code == 500  # Should handle error gracefully
    
    def test_detect_video_anomaly_missing_file(self, client):
        """Test video detection with missing file"""
        response = client.post("/detect/video")
        
        assert response.status_code != 200  # Should return error