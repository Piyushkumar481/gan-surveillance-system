"""
Unit tests for database operations
"""

import pytest
import os
from sqlalchemy.orm import Session
from src.database.models import AnomalyEvent, get_engine, init_db, get_session
from src.database.crud import create_anomaly_event, get_anomaly_events, get_anomaly_stats

class TestDatabase:
    @pytest.fixture(autouse=True)
    def setup_db(self):
        """Set up fresh database for each test"""
        self.engine = get_engine(":memory:")  # Use in-memory database for tests
        init_db(self.engine)
        self.session = get_session(self.engine)
        yield
        self.session.close()
    
    def test_create_anomaly_event(self):
        """Test creating a new anomaly event"""
        event = create_anomaly_event(
            session=self.session,
            anomaly_score=0.85,
            camera_id="test_cam_1",
            location="Entrance",
            confidence=0.9
        )
        
        assert event.id is not None
        assert event.anomaly_score == 0.85
        assert event.is_anomaly == True  # Should be True since score > 0.1
        assert event.camera_id == "test_cam_1"
    
    def test_get_anomaly_events(self):
        """Test retrieving anomaly events"""
        # Create test events
        create_anomaly_event(self.session, 0.05, "test_cam_1")  # Normal
        create_anomaly_event(self.session, 0.15, "test_cam_1")  # Anomaly
        create_anomaly_event(self.session, 0.25, "test_cam_2")  # Anomaly
        
        # Get all events
        all_events = get_anomaly_events(self.session)
        assert len(all_events) == 3
        
        # Get only anomalies
        anomaly_events = get_anomaly_events(self.session, only_anomalies=True)
        assert len(anomaly_events) == 2
        assert all(event.is_anomaly for event in anomaly_events)
    
    def test_get_anomaly_stats(self):
        """Test getting anomaly statistics"""
        # Create test events
        create_anomaly_event(self.session, 0.05)  # Normal
        create_anomaly_event(self.session, 0.15)  # Anomaly
        create_anomaly_event(self.session, 0.25)  # Anomaly
        
        stats = get_anomaly_stats(self.session)
        
        assert stats["total_events"] == 3
        assert stats["anomaly_events"] == 2
        assert stats["anomaly_rate"] == 2/3
        assert 0.1 < stats["avg_anomaly_score"] < 0.2
    
    def test_anomaly_event_to_dict(self):
        """Test converting event to dictionary"""
        event = create_anomaly_event(
            session=self.session,
            anomaly_score=0.3,
            camera_id="test_cam",
            location="Test Location"
        )
        
        event_dict = event.to_dict()
        
        assert "id" in event_dict
        assert "timestamp" in event_dict
        assert event_dict["anomaly_score"] == 0.3
        assert event_dict["camera_id"] == "test_cam"
        assert event_dict["is_anomaly"] == True