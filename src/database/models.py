"""
SQLAlchemy database models for anomaly events
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import base64

Base = declarative_base()

class AnomalyEvent(Base):
    __tablename__ = "anomaly_events"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    camera_id = Column(String(50), index=True, default="default")
    anomaly_score = Column(Float, nullable=False)
    frame_path = Column(String(255))
    is_anomaly = Column(Boolean, default=False)
    location = Column(String(100))
    resolved = Column(Boolean, default=False)
    resolution_notes = Column(Text)
    confidence = Column(Float, default=0.0)
    processed_time = Column(Float)  # Processing time in seconds
    
    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "camera_id": self.camera_id,
            "anomaly_score": self.anomaly_score,
            "is_anomaly": self.is_anomaly,
            "location": self.location,
            "resolved": self.resolved,
            "confidence": self.confidence,
            "processed_time": self.processed_time
        }

def get_engine(db_path: str = "anomalies.db"):
    """Create SQLite database engine"""
    return create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})

def get_session(engine):
    """Create database session"""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()

def init_db(engine):
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)