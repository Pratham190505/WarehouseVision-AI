"""
Processing service that orchestrates image and video analysis.

Acts as the main entry point for analysis requests,
delegating to ImageService or VideoService based on media type.
"""

import os
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from app.services.image_service import ImageService
from app.services.video_service import VideoService
from app.database.mongodb import get_collection
from app.database.models import AnalysisRecord, ObjectCounts, AnalyticsData

logger = logging.getLogger(__name__)

# Supported file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}

BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = BASE_DIR / os.getenv("UPLOAD_DIR", "uploads")
RESULT_DIR = BASE_DIR / os.getenv("RESULT_DIR", "results")


def get_media_type(filename: str) -> Optional[str]:
    """Determine media type from filename extension."""
    ext = Path(filename).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    elif ext in VIDEO_EXTENSIONS:
        return "video"
    return None


def serialize_mongo_doc(doc: Any) -> Any:
    """Recursively convert MongoDB document fields (datetime, ObjectId) to JSON-serializable primitives."""
    if isinstance(doc, dict):
        cleaned = {}
        for k, v in doc.items():
            if k == "_id":
                continue
            cleaned[k] = serialize_mongo_doc(v)
        return cleaned
    elif isinstance(doc, list):
        return [serialize_mongo_doc(item) for item in doc]
    elif hasattr(doc, "isoformat"):
        return doc.isoformat()
    elif isinstance(doc, (int, float, str, bool)) or doc is None:
        return doc
    else:
        return str(doc)


_shared_processing_service: Optional["ProcessingService"] = None


def get_processing_service() -> "ProcessingService":
    """Get the shared singleton ProcessingService instance."""
    global _shared_processing_service
    if _shared_processing_service is None:
        _shared_processing_service = ProcessingService()
    return _shared_processing_service


class ProcessingService:
    """Orchestrates media processing and database storage."""

    def __init__(self):
        self.image_service = ImageService()
        self.video_service = VideoService()
        self._in_memory_analyses: Dict[str, Dict[str, Any]] = {}

    def create_analysis(self, filename: str, original_filename: str,
                        media_type: str, file_size: int,
                        media_path: str, user_id: Optional[str] = None) -> str:
        """
        Create an initial analysis record in MongoDB.

        Returns the analysis_id.
        """
        analysis_id = str(uuid.uuid4())
        now = datetime.utcnow()

        record = AnalysisRecord(
            analysis_id=analysis_id,
            user_id=user_id,
            filename=filename,
            original_filename=original_filename,
            media_type=media_type,
            timestamp=now,
            created_at=now,
            status="pending",
            file_size=file_size,
            media_path=media_path,
        )

        collection = get_collection("analyses")
        if collection is not None:
            try:
                collection.insert_one(record.to_mongo())
                logger.info(f"Created analysis record: {analysis_id} for user: {user_id}")
            except Exception as e:
                logger.error(f"Failed to create analysis record: {e}")

        # In-memory record as fallback
        self._in_memory_analyses[analysis_id] = record.model_dump()
        return analysis_id

    def process_media(self, analysis_id: str, media_path: str,
                      media_type: str) -> Dict[str, Any]:
        """
        Process media through the appropriate pipeline.

        Args:
            analysis_id: The analysis ID.
            media_path: Path to the media file.
            media_type: 'image' or 'video'.

        Returns:
            Analysis result dict.
        """
        # Update status to processing
        self._update_status(analysis_id, "processing")

        if media_type == "image":
            result_dir = str(RESULT_DIR / "images")
            os.makedirs(result_dir, exist_ok=True)
            result = self.image_service.process(media_path, result_dir, analysis_id)
        elif media_type == "video":
            result_dir = str(RESULT_DIR / "videos")
            os.makedirs(result_dir, exist_ok=True)
            result = self.video_service.process(media_path, result_dir, analysis_id)
        else:
            result = {
                "analysis_id": analysis_id,
                "status": "failed",
                "error_message": f"Unsupported media type: {media_type}",
            }

        # Save result to MongoDB
        self._save_result(analysis_id, result, media_type)

        return result

    def _update_status(self, analysis_id: str, status: str):
        """Update analysis status in MongoDB."""
        collection = get_collection("analyses")
        if collection is not None:
            try:
                collection.update_one(
                    {"analysis_id": analysis_id},
                    {"$set": {"status": status}},
                )
            except Exception as e:
                logger.error(f"Failed to update status: {e}")

    def _save_result(self, analysis_id: str, result: Dict[str, Any],
                     media_type: str):
        """Save analysis result to MongoDB and in-memory cache."""
        update_data = {
            "status": result.get("status", "completed"),
            "processing_time": result.get("processing_time", 0),
            "result_path": result.get("result_path"),
            "thumbnail_path": result.get("thumbnail_path"),
            "error_message": result.get("error_message"),
            "object_counts": result.get("object_counts", {}),
            "analytics": {
                "occupancy": result.get("analytics", {}).get("occupancy", 0),
                "space_utilization": result.get("analytics", {}).get("space_utilization", 0),
                "restricted_zone_violations": result.get("analytics", {}).get("restricted_zone_violations", 0),
                "insights": result.get("analytics", {}).get("insights", []),
            },
        }

        # Add tracking data for videos
        if media_type == "video" and "tracking" in result:
            tracking = result["tracking"]
            update_data["tracking"] = {
                "unique_workers": tracking.get("unique_workers", 0),
                "unique_forklifts": tracking.get("unique_forklifts", 0),
                "unique_boxes": tracking.get("unique_boxes", 0),
                "unique_pallets": tracking.get("unique_pallets", 0),
                "total_frames": tracking.get("total_frames", 0),
                "processed_frames": tracking.get("processed_frames", 0),
                "frame_data": tracking.get("frame_data", [])[:100],
            }

        # Update in-memory fallback
        if analysis_id in self._in_memory_analyses:
            self._in_memory_analyses[analysis_id].update(update_data)

        collection = get_collection("analyses")
        if collection is None:
            return

        try:
            collection.update_one(
                {"analysis_id": analysis_id},
                {"$set": update_data},
            )
            logger.info(f"Saved analysis result: {analysis_id}")
        except Exception as e:
            logger.error(f"Failed to save result: {e}")

    def get_analysis(self, analysis_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve analysis result.
        If user_id is provided, ensures the record belongs to that user.
        """
        collection = get_collection("analyses")
        if collection is not None:
            try:
                query = {"analysis_id": analysis_id}
                if user_id:
                    query["user_id"] = user_id

                doc = collection.find_one(query)
                if doc:
                    return serialize_mongo_doc(doc)
                return None
            except Exception as e:
                logger.error(f"Failed to get analysis: {e}")

        # Fallback to in-memory store
        doc = self._in_memory_analyses.get(analysis_id)
        if doc:
            if user_id and doc.get("user_id") != user_id:
                return None
            return serialize_mongo_doc(doc)
        return None

    def get_history(self, user_id: Optional[str] = None, limit: int = 50) -> list:
        """
        Get analysis history filtered strictly by user_id.
        """
        query: Dict[str, Any] = {}
        if user_id:
            query["user_id"] = user_id

        collection = get_collection("analyses")
        if collection is not None:
            try:
                cursor = collection.find(
                    query,
                    {
                        "_id": 0,
                        "analysis_id": 1,
                        "user_id": 1,
                        "original_filename": 1,
                        "media_type": 1,
                        "timestamp": 1,
                        "created_at": 1,
                        "status": 1,
                        "processing_time": 1,
                        "object_counts": 1,
                        "analytics": 1,
                        "result_path": 1,
                        "thumbnail_path": 1,
                    }
                ).sort("timestamp", -1).limit(limit)

                return [serialize_mongo_doc(doc) for doc in cursor]
            except Exception as e:
                logger.error(f"Failed to get history: {e}")

        # Fallback in-memory
        records = []
        for doc in sorted(self._in_memory_analyses.values(), key=lambda x: str(x.get("timestamp", "")), reverse=True):
            if user_id and doc.get("user_id") != user_id:
                continue
            records.append(serialize_mongo_doc(doc))
            if len(records) >= limit:
                break
        return records

    def get_dashboard_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get aggregated dashboard statistics filtered strictly by user_id.
        """
        default_stats = {
            "total_analyses": 0,
            "latest_analysis": None,
            "total_workers": 0,
            "total_forklifts": 0,
            "total_boxes": 0,
            "total_pallets": 0,
            "avg_occupancy": 0,
            "avg_space_utilization": 0,
            "total_violations": 0,
        }

        collection = get_collection("analyses")
        if collection is not None:
            try:
                base_query = {"user_id": user_id} if user_id else {}
                completed_query = {"status": "completed"}
                if user_id:
                    completed_query["user_id"] = user_id

                total = collection.count_documents(base_query)
                latest = collection.find_one(
                    completed_query,
                    {"_id": 0},
                    sort=[("timestamp", -1)],
                )
                latest_clean = serialize_mongo_doc(latest) if latest else None

                match_stage: Dict[str, Any] = {"status": "completed"}
                if user_id:
                    match_stage["user_id"] = user_id

                pipeline = [
                    {"$match": match_stage},
                    {"$group": {
                        "_id": None,
                        "total_workers": {"$sum": "$object_counts.workers"},
                        "total_forklifts": {"$sum": "$object_counts.forklifts"},
                        "total_boxes": {"$sum": "$object_counts.boxes"},
                        "total_pallets": {"$sum": "$object_counts.pallets"},
                        "avg_occupancy": {"$avg": "$analytics.occupancy"},
                        "avg_space_utilization": {"$avg": "$analytics.space_utilization"},
                        "total_violations": {"$sum": "$analytics.restricted_zone_violations"},
                    }},
                ]

                agg_result = list(collection.aggregate(pipeline))
                stats = agg_result[0] if agg_result else {}

                return {
                    "total_analyses": total,
                    "latest_analysis": latest_clean,
                    "total_workers": stats.get("total_workers", 0),
                    "total_forklifts": stats.get("total_forklifts", 0),
                    "total_boxes": stats.get("total_boxes", 0),
                    "total_pallets": stats.get("total_pallets", 0),
                    "avg_occupancy": round(stats.get("avg_occupancy", 0), 1),
                    "avg_space_utilization": round(stats.get("avg_space_utilization", 0), 1),
                    "total_violations": stats.get("total_violations", 0),
                }
            except Exception as e:
                logger.error(f"Failed to get dashboard stats: {e}")

        # In-memory fallback
        user_analyses = [
            a for a in self._in_memory_analyses.values()
            if not user_id or a.get("user_id") == user_id
        ]
        completed = [a for a in user_analyses if a.get("status") == "completed"]
        latest = completed[-1] if completed else None

        total_workers = sum(a.get("object_counts", {}).get("workers", 0) for a in completed)
        total_forklifts = sum(a.get("object_counts", {}).get("forklifts", 0) for a in completed)
        total_boxes = sum(a.get("object_counts", {}).get("boxes", 0) for a in completed)
        total_pallets = sum(a.get("object_counts", {}).get("pallets", 0) for a in completed)
        total_violations = sum(a.get("analytics", {}).get("restricted_zone_violations", 0) for a in completed)
        avg_occupancy = round(sum(a.get("analytics", {}).get("occupancy", 0) for a in completed) / len(completed), 1) if completed else 0
        avg_util = round(sum(a.get("analytics", {}).get("space_utilization", 0) for a in completed) / len(completed), 1) if completed else 0

        return {
            "total_analyses": len(user_analyses),
            "latest_analysis": serialize_mongo_doc(latest) if latest else None,
            "total_workers": total_workers,
            "total_forklifts": total_forklifts,
            "total_boxes": total_boxes,
            "total_pallets": total_pallets,
            "avg_occupancy": avg_occupancy,
            "avg_space_utilization": avg_util,
            "total_violations": total_violations,
        }
