"""
Pydantic schemas for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class AnomalyDetectionResponse(BaseModel):
    anomaly_score: float = Field(..., ge=0, le=1, description="Anomaly score between 0 and 1")
    is_anomaly: bool = Field(..., description="Whether the frame is anomalous")
    message: str = Field(..., description="Description of the result")
    threshold: float = Field(0.1, description="Detection threshold used")

class VideoDetectionResponse(BaseModel):
    total_frames: int = Field(..., description="Total frames processed")
    anomalous_frames: int = Field(..., description="Number of anomalous frames")
    results: List[dict] = Field(..., description="Detailed results per frame")

class AnomalyEventResponse(BaseModel):
    id: int
    timestamp: datetime
    camera_id: str
    anomaly_score: float
    is_anomaly: bool
    location: Optional[str] = None
    resolved: bool
    confidence: Optional[float] = None
    processed_time: Optional[float] = None

class StatsResponse(BaseModel):
    time_range_hours: int
    stats: dict
    timestamp: datetime

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: datetime

class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)