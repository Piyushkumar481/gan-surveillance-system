"""
Additional API routes for extended functionality
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime, timedelta
from src.database.models import get_engine, get_session
from src.database.crud import get_anomaly_events, get_anomaly_stats, update_anomaly_event
from src.utils.visualizer import ResultVisualizer
import matplotlib.pyplot as plt
import io
import base64

router = APIRouter()

@router.get("/events")
async def get_events(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    only_anomalies: bool = Query(False),
    hours: int = Query(24, ge=1, le=720)
):
    """Get anomaly events with filtering"""
    try:
        engine = get_engine()
        session = get_session(engine)
        
        time_range = timedelta(hours=hours) if hours > 0 else None
        events = get_anomaly_events(session, skip, limit, only_anomalies, time_range)
        
        return {
            "events": [event.to_dict() for event in events],
            "total_retrieved": len(events),
            "filters": {
                "skip": skip,
                "limit": limit,
                "only_anomalies": only_anomalies,
                "time_range_hours": hours
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/stats")
async def get_stats(hours: int = Query(24, ge=1, le=720)):
    """Get anomaly statistics"""
    try:
        engine = get_engine()
        session = get_session(engine)
        
        time_range = timedelta(hours=hours)
        stats = get_anomaly_stats(session, time_range)
        
        return {
            "time_range_hours": hours,
            "stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.patch("/events/{event_id}")
async def update_event(
    event_id: int,
    resolved: Optional[bool] = None,
    notes: Optional[str] = None
):
    """Update anomaly event resolution status"""
    try:
        engine = get_engine()
        session = get_session(engine)
        
        event = update_anomaly_event(session, event_id, resolved, notes)
        
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        
        return {"message": "Event updated successfully", "event": event.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/visualization")
async def get_visualization(
    anomaly_scores: List[float] = Query(...),
    threshold: float = Query(0.1)
):
    """Generate anomaly visualization plot"""
    try:
        # Create plot
        fig = ResultVisualizer.plot_anomaly_scores(anomaly_scores, threshold)
        
        # Convert to base64 for API response
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return {
            "image_type": "image/png",
            "image_data": image_base64,
            "anomaly_scores": anomaly_scores,
            "threshold": threshold
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization error: {str(e)}")