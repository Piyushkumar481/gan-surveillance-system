"""
Database operations for anomaly events
"""

from sqlalchemy.orm import Session
from .models import AnomalyEvent
from datetime import datetime, timedelta
from typing import List, Optional

def create_anomaly_event(
    session: Session,
    anomaly_score: float,
    frame_path: Optional[str] = None,
    camera_id: str = "default",
    location: Optional[str] = None,
    processed_time: Optional[float] = None,
    confidence: float = 0.0
) -> AnomalyEvent:
    """Create new anomaly event record"""
    event = AnomalyEvent(
        anomaly_score=anomaly_score,
        frame_path=frame_path,
        camera_id=camera_id,
        location=location,
        is_anomaly=anomaly_score > 0.1,  # Default threshold
        processed_time=processed_time,
        confidence=confidence
    )
    
    session.add(event)
    session.commit()
    session.refresh(event)
    return event

def get_anomaly_events(
    session: Session,
    skip: int = 0,
    limit: int = 100,
    only_anomalies: bool = False,
    time_range: Optional[timedelta] = None
) -> List[AnomalyEvent]:
    """Retrieve anomaly events with filtering"""
    query = session.query(AnomalyEvent)
    
    if only_anomalies:
        query = query.filter(AnomalyEvent.is_anomaly == True)
    
    if time_range:
        time_threshold = datetime.utcnow() - time_range
        query = query.filter(AnomalyEvent.timestamp >= time_threshold)
    
    return query.order_by(AnomalyEvent.timestamp.desc()).offset(skip).limit(limit).all()

def get_anomaly_stats(session: Session, time_range: timedelta = timedelta(hours=24)) -> dict:
    """Get statistics about anomaly events"""
    time_threshold = datetime.utcnow() - time_range
    
    total = session.query(AnomalyEvent).filter(AnomalyEvent.timestamp >= time_threshold).count()
    anomalies = session.query(AnomalyEvent).filter(
        AnomalyEvent.timestamp >= time_threshold,
        AnomalyEvent.is_anomaly == True
    ).count()
    
    avg_score = session.query(
        sqlalchemy.func.avg(AnomalyEvent.anomaly_score)
    ).filter(AnomalyEvent.timestamp >= time_threshold).scalar() or 0
    
    return {
        "total_events": total,
        "anomaly_events": anomalies,
        "anomaly_rate": anomalies / total if total > 0 else 0,
        "avg_anomaly_score": float(avg_score)
    }

def update_anomaly_event(
    session: Session,
    event_id: int,
    resolved: Optional[bool] = None,
    notes: Optional[str] = None
) -> Optional[AnomalyEvent]:
    """Update anomaly event resolution status"""
    event = session.query(AnomalyEvent).filter(AnomalyEvent.id == event_id).first()
    
    if event:
        if resolved is not None:
            event.resolved = resolved
        if notes is not None:
            event.resolution_notes = notes
        
        session.commit()
        session.refresh(event)
    
    return event