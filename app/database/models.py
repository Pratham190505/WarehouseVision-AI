"""
Pydantic models for WarehouseVision AI.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class ObjectCounts(BaseModel):
    """Counts of detected warehouse objects."""
    workers: int = 0
    forklifts: int = 0
    boxes: int = 0
    pallets: int = 0


class TrackingData(BaseModel):
    """Tracking data for video analysis."""
    unique_workers: int = 0
    unique_forklifts: int = 0
    unique_boxes: int = 0
    unique_pallets: int = 0
    total_frames: int = 0
    processed_frames: int = 0
    frame_data: List[Dict[str, Any]] = Field(default_factory=list)


class AnalyticsData(BaseModel):
    """Analytics computed from analysis."""
    occupancy: float = 0.0
    space_utilization: float = 0.0
    restricted_zone_violations: int = 0
    insights: List[str] = Field(default_factory=list)


class AnalysisRecord(BaseModel):
    """Complete analysis record stored in MongoDB."""
    analysis_id: str
    user_id: Optional[str] = None
    filename: str
    original_filename: str
    media_type: str  # "image" or "video"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "pending"  # pending, uploading, processing, analyzing, completed, failed
    processing_time: float = 0.0
    object_counts: ObjectCounts = Field(default_factory=ObjectCounts)
    analytics: AnalyticsData = Field(default_factory=AnalyticsData)
    tracking: Optional[TrackingData] = None
    result_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    error_message: Optional[str] = None
    file_size: int = 0
    media_path: str = ""

    safety_alerts_count: int = 0
    safety_incidents: List[Dict[str, Any]] = Field(default_factory=list)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def to_mongo(self) -> dict:
        """Convert to MongoDB document."""
        data = self.model_dump()
        data["timestamp"] = self.timestamp
        data["created_at"] = self.created_at
        return data

    @classmethod
    def from_mongo(cls, doc: dict) -> "AnalysisRecord":
        """Create from MongoDB document."""
        if "_id" in doc:
            doc.pop("_id")
        return cls(**doc)


class SafetyEventRecord(BaseModel):
    """Safety proximity event record stored in MongoDB safety_events collection."""
    event_id: str
    user_id: Optional[str] = None
    analysis_id: str
    event_type: str = "machine_proximity"  # "machine_proximity" or "restricted_zone_violation"
    worker_track_id: Optional[str] = None
    machine_track_id: Optional[str] = None
    machine_type: Optional[str] = None
    zone_name: Optional[str] = None
    proximity_method: str = "image_space"
    estimated_distance_m: Optional[float] = None
    configured_threshold_m: Optional[float] = None
    severity: str = "HIGH"  # CRITICAL, HIGH, WARNING
    status: str = "ACTIVE"  # ACTIVE, RESOLVED
    is_acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    first_detected_at: datetime = Field(default_factory=datetime.utcnow)
    last_detected_at: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def to_mongo(self) -> dict:
        """Convert to MongoDB document."""
        data = self.model_dump()
        data["first_detected_at"] = self.first_detected_at
        data["last_detected_at"] = self.last_detected_at
        data["created_at"] = self.created_at
        if self.acknowledged_at:
            data["acknowledged_at"] = self.acknowledged_at
        if self.resolved_at:
            data["resolved_at"] = self.resolved_at
        return data

    @classmethod
    def from_mongo(cls, doc: dict) -> "SafetyEventRecord":
        """Create from MongoDB document."""
        if "_id" in doc:
            doc.pop("_id")
        return cls(**doc)


class UserCreate(BaseModel):
    """Payload for user registration."""
    name: str = Field(..., min_length=2, max_length=100)
    email: str = Field(..., min_length=5, max_length=150)
    password: str = Field(..., min_length=6, max_length=128)
    confirm_password: str = Field(..., min_length=6, max_length=128)


class UserLogin(BaseModel):
    """Payload for user login."""
    email: str = Field(..., min_length=5, max_length=150)
    password: str = Field(..., min_length=1, max_length=128)


class UserProfileUpdate(BaseModel):
    """Payload for updating user profile."""
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    organization: Optional[str] = Field(None, max_length=150)
    department: Optional[str] = Field(None, max_length=100)


class UserResponse(BaseModel):
    """Sanitized user response model."""
    id: str
    name: str
    email: str
    role: str = "user"
    organization: Optional[str] = ""
    department: Optional[str] = ""
    created_at: str


class UserRecord(BaseModel):
    """User record stored in MongoDB."""
    name: str
    email: str
    password_hash: str
    role: str = "user"
    organization: Optional[str] = ""
    department: Optional[str] = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def to_mongo(self) -> dict:
        data = self.model_dump()
        data["created_at"] = self.created_at
        return data


class AnalysisResponse(BaseModel):
    """API response for analysis results."""
    analysis_id: str
    status: str
    filename: str
    media_type: str
    timestamp: str
    processing_time: float = 0.0
    object_counts: ObjectCounts = Field(default_factory=ObjectCounts)
    analytics: AnalyticsData = Field(default_factory=AnalyticsData)
    tracking: Optional[TrackingData] = None
    result_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    error_message: Optional[str] = None
    safety_alerts_count: int = 0
    safety_incidents: List[Dict[str, Any]] = Field(default_factory=list)


class DashboardStats(BaseModel):
    """Aggregated stats for the dashboard."""
    total_analyses: int = 0
    latest_analysis: Optional[Dict[str, Any]] = None
    total_workers: int = 0
    total_forklifts: int = 0
    total_boxes: int = 0
    total_pallets: int = 0
    avg_occupancy: float = 0.0
    avg_space_utilization: float = 0.0
    total_violations: int = 0
    active_safety_alerts: int = 0
    total_safety_events: int = 0
    resolved_safety_events: int = 0
