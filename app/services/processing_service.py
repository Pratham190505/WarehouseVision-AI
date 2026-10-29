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
from typing import Dict, Any, Optional, List

from app.services.image_service import ImageService
from app.services.video_service import VideoService
from app.database.mongodb import get_collection
from app.database.models import AnalysisRecord, ObjectCounts, AnalyticsData, SafetyEventRecord

logger = logging.getLogger(__name__)

# Supported file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}

BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = BASE_DIR / os.getenv("UPLOAD_DIR", "uploads")
RESULT_DIR = BASE_DIR / os.getenv("RESULT_DIR", "results")
BOX_TRACKING_RESULT_DIR = BASE_DIR / os.getenv("BOX_TRACKING_RESULT_DIR", "private_box_results")


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
        self._in_memory_safety_events: Dict[str, Dict[str, Any]] = {}

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
                      media_type: str, box_tracking_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
            result_dir = str(BOX_TRACKING_RESULT_DIR if box_tracking_config is not None else RESULT_DIR / "videos")
            os.makedirs(result_dir, exist_ok=True)
            result = self.video_service.process(
                media_path, result_dir, analysis_id, box_tracking_config=box_tracking_config
            )
        else:
            result = {
                "analysis_id": analysis_id,
                "status": "failed",
                "error_message": f"Unsupported media type: {media_type}",
            }

        result["media_type"] = media_type

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
        safety_incidents = result.get("safety_incidents", []) or result.get("safety_alerts", [])
        safety_alerts_count = len(safety_incidents)

        update_data = {
            "status": result.get("status", "completed"),
            "processing_time": result.get("processing_time", 0),
            "result_path": result.get("result_path"),
            "video_filename": result.get("video_filename"),
            "thumbnail_path": result.get("thumbnail_path"),
            "error_message": result.get("error_message"),
            "object_counts": result.get("object_counts", {}),
            "safety_alerts_count": safety_alerts_count,
            "safety_incidents": [serialize_mongo_doc(inc) for inc in safety_incidents],
            "analytics": {
                "occupancy": result.get("analytics", {}).get("occupancy", 0),
                "space_utilization": result.get("analytics", {}).get("space_utilization", 0),
                "restricted_zone_violations": result.get("analytics", {}).get("restricted_zone_violations", 0),
                "insights": result.get("analytics", {}).get("insights", []),
            },
        }

        # Only videos have a processed video URL. Keeping this field absent for
        # images prevents clients from rendering an annotated image as a video.
        if media_type == "video":
            update_data["processed_video_url"] = (
                result.get("processed_video_url") or result.get("result_path")
            )

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

        if "box_tracking" in result:
            update_data["box_tracking"] = serialize_mongo_doc(result["box_tracking"])
            update_data["unique_boxes_tracked"] = int(result["box_tracking"].get("total_unique_boxes", 0))

        # Retrieve user_id from existing analysis
        user_id = None
        if analysis_id in self._in_memory_analyses:
            user_id = self._in_memory_analyses[analysis_id].get("user_id")
        else:
            coll = get_collection("analyses")
            if coll is not None:
                existing = coll.find_one({"analysis_id": analysis_id}, {"user_id": 1})
                if existing:
                    user_id = existing.get("user_id")

        # Persist safety events
        self._persist_safety_events(analysis_id, user_id, safety_incidents)

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
                        "processed_video_url": 1,
                        "thumbnail_path": 1,
                        "box_tracking": 1,
                        "unique_boxes_tracked": 1,
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

    def delete_analysis(self, analysis_id: str, user_id: Optional[str] = None) -> bool:
        """
        Delete an analysis record, its generated output files, and associated safety events.
        If user_id is provided, ensures the record belongs to that user.
        Returns True if deleted, False if not found or unauthorized.
        """
        target_doc = None
        collection = get_collection("analyses")
        if collection is not None:
            try:
                query: Dict[str, Any] = {"analysis_id": analysis_id}
                if user_id:
                    query["user_id"] = user_id
                target_doc = collection.find_one(query)
            except Exception as e:
                logger.error(f"Failed to find analysis for deletion: {e}")

        if target_doc is None and analysis_id in self._in_memory_analyses:
            mem_doc = self._in_memory_analyses[analysis_id]
            if not user_id or mem_doc.get("user_id") == user_id:
                target_doc = mem_doc

        if target_doc is None:
            return False

        # Attempt to clean up associated files if they exist
        paths_to_clean = []
        media_path = target_doc.get("media_path")
        if media_path and isinstance(media_path, str) and os.path.isfile(media_path):
            paths_to_clean.append(media_path)

        result_path = target_doc.get("result_path")
        if result_path and isinstance(result_path, str):
            if result_path.startswith("/results/"):
                rel_part = result_path.replace("/results/", "", 1)
                full_result_path = RESULT_DIR / rel_part
                if full_result_path.is_file():
                    paths_to_clean.append(str(full_result_path))

        thumbnail_path = target_doc.get("thumbnail_path")
        if thumbnail_path and isinstance(thumbnail_path, str):
            if thumbnail_path.startswith("/results/"):
                rel_part = thumbnail_path.replace("/results/", "", 1)
                full_thumb_path = RESULT_DIR / rel_part
                if full_thumb_path.is_file():
                    paths_to_clean.append(str(full_thumb_path))

        box_tracking = target_doc.get("box_tracking")
        if isinstance(box_tracking, dict):
            for key in ("output_filename", "thumbnail_filename"):
                bt_fname = box_tracking.get(key)
                if bt_fname and isinstance(bt_fname, str):
                    bt_path = BOX_TRACKING_RESULT_DIR / bt_fname
                    if bt_path.is_file():
                        paths_to_clean.append(str(bt_path))

        video_filename = target_doc.get("video_filename")
        if video_filename and isinstance(video_filename, str):
            v_path = RESULT_DIR / "videos" / video_filename
            if v_path.is_file():
                paths_to_clean.append(str(v_path))
        for vid_cand in (RESULT_DIR / "videos").glob(f"{analysis_id}*"):
            if vid_cand.is_file():
                paths_to_clean.append(str(vid_cand))

        for p in paths_to_clean:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception as e:
                logger.warning(f"Could not remove file {p}: {e}")

        # Delete from MongoDB analyses
        if collection is not None:
            try:
                query = {"analysis_id": analysis_id}
                if user_id:
                    query["user_id"] = user_id
                collection.delete_one(query)
            except Exception as e:
                logger.error(f"Failed to delete analysis from MongoDB: {e}")

        # Delete from in-memory analyses
        self._in_memory_analyses.pop(analysis_id, None)

        # Delete associated safety events
        safety_col = get_collection("safety_events")
        if safety_col is not None:
            try:
                safety_query: Dict[str, Any] = {"analysis_id": analysis_id}
                if user_id:
                    safety_query["user_id"] = user_id
                safety_col.delete_many(safety_query)
            except Exception as e:
                logger.error(f"Failed to delete safety events for analysis {analysis_id}: {e}")

        # In-memory safety events cleanup
        keys_to_delete = [
            k for k, v in self._in_memory_safety_events.items()
            if v.get("analysis_id") == analysis_id and (not user_id or v.get("user_id") == user_id)
        ]
        for k in keys_to_delete:
            self._in_memory_safety_events.pop(k, None)

        logger.info(f"Successfully deleted analysis {analysis_id} for user {user_id}")
        return True

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

                # Safety events stats derived from actual persisted safety events
                safety_summary = self.get_safety_summary(user_id=user_id)
                active_safety_alerts = safety_summary["active"]
                total_safety_events = safety_summary["total"]
                resolved_safety_events = safety_summary["resolved"]

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
                    "active_safety_alerts": active_safety_alerts,
                    "total_safety_events": total_safety_events,
                    "resolved_safety_events": resolved_safety_events,
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

        safety_summary = self.get_safety_summary(user_id=user_id)

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
            "active_safety_alerts": safety_summary["active"],
            "total_safety_events": safety_summary["total"],
            "resolved_safety_events": safety_summary["resolved"],
        }

    def get_safety_summary(self, user_id: Optional[str] = None) -> Dict[str, int]:
        """
        Derive safety KPI counts from actual persisted safety events for the authenticated user.
        - active: Count of incidents with status 'ACTIVE'
        - total: Count of all persisted safety incidents for the user
        - resolved: Count of incidents with status 'RESOLVED'
        """
        collection = get_collection("safety_events")
        if collection is not None:
            try:
                sq: Dict[str, Any] = {}
                if user_id:
                    sq["user_id"] = user_id
                total = collection.count_documents(sq)
                active = collection.count_documents({**sq, "status": "ACTIVE"})
                resolved = collection.count_documents({**sq, "status": "RESOLVED"})
                return {
                    "active": active,
                    "total": total,
                    "resolved": resolved,
                }
            except Exception as e:
                logger.error(f"Failed to fetch safety summary from MongoDB: {e}")

        # In-memory fallback
        user_events = [
            e for e in self._in_memory_safety_events.values()
            if not user_id or e.get("user_id") == user_id
        ]
        return {
            "active": len([e for e in user_events if e.get("status") == "ACTIVE"]),
            "total": len(user_events),
            "resolved": len([e for e in user_events if e.get("status") == "RESOLVED"]),
        }

    def _persist_safety_events(
        self,
        analysis_id: str,
        user_id: Optional[str],
        incidents: List[Dict[str, Any]],
    ):
        """
        Persist safety events into MongoDB 'safety_events' collection and in-memory store.
        Avoids creating duplicate events for the same incident; updates existing active incidents.
        Handles both restricted zone violations and machine proximity incidents.
        """
        now = datetime.utcnow()
        collection = get_collection("safety_events")

        for inc in incidents:
            event_id = inc.get("event_id") or inc.get("incident_id")
            event_type = inc.get("event_type") or ("restricted_zone_violation" if inc.get("zone_name") else "machine_proximity")
            zone_name = inc.get("zone_name")

            raw_wid = inc.get("worker_track_id") or inc.get("worker_id") or "worker"
            wid = str(raw_wid).replace("Worker #", "").strip()

            raw_mid = inc.get("machine_track_id") or inc.get("machine_id")
            mid = None
            if raw_mid is not None:
                mid = str(raw_mid)
                if "#" in mid:
                    mid = mid.split("#")[-1].strip()

            if not event_id:
                if event_type == "restricted_zone_violation" and zone_name:
                    clean_zone = zone_name.lower().replace(" ", "_")
                    event_id = f"{analysis_id}_{wid}_{clean_zone}"
                else:
                    event_id = f"{analysis_id}_{wid}_{mid or 'machine'}"

            machine_type = inc.get("machine_type") or ("forklift" if event_type == "machine_proximity" else None)
            proximity_method = inc.get("proximity_method", "image_space")
            dist_m = inc.get("estimated_distance_m")
            thresh_m = inc.get("configured_threshold_m")
            severity = inc.get("severity", "HIGH")
            status = inc.get("status", "ACTIVE")

            first_detected = inc.get("first_detected_at")
            if isinstance(first_detected, str):
                try:
                    first_detected = datetime.fromisoformat(first_detected.replace("Z", "+00:00"))
                except Exception:
                    first_detected = now
            elif not isinstance(first_detected, datetime):
                first_detected = now

            last_detected = inc.get("last_detected_at")
            if isinstance(last_detected, str):
                try:
                    last_detected = datetime.fromisoformat(last_detected.replace("Z", "+00:00"))
                except Exception:
                    last_detected = now
            elif not isinstance(last_detected, datetime):
                last_detected = now

            resolved_at = inc.get("resolved_at")
            if isinstance(resolved_at, str):
                try:
                    resolved_at = datetime.fromisoformat(resolved_at.replace("Z", "+00:00"))
                except Exception:
                    resolved_at = now
            elif not isinstance(resolved_at, datetime) and status == "RESOLVED":
                resolved_at = now

            event_doc = {
                "event_id": event_id,
                "user_id": user_id,
                "analysis_id": analysis_id,
                "event_type": event_type,
                "zone_name": zone_name,
                "worker_track_id": wid,
                "machine_track_id": mid,
                "machine_type": machine_type,
                "proximity_method": proximity_method,
                "estimated_distance_m": dist_m,
                "configured_threshold_m": thresh_m,
                "severity": severity,
                "status": status,
                "is_acknowledged": inc.get("is_acknowledged", False),
                "acknowledged_at": inc.get("acknowledged_at"),
                "acknowledged_by": inc.get("acknowledged_by"),
                "resolved_at": resolved_at,
                "first_detected_at": first_detected,
                "last_detected_at": last_detected,
                "created_at": now,
            }

            # Update or insert in in-memory store
            if event_id in self._in_memory_safety_events:
                existing = self._in_memory_safety_events[event_id]
                existing["last_detected_at"] = last_detected
                existing["status"] = status
                if user_id:
                    existing["user_id"] = user_id
                if dist_m is not None:
                    existing["estimated_distance_m"] = dist_m
                if resolved_at:
                    existing["resolved_at"] = resolved_at
            else:
                self._in_memory_safety_events[event_id] = event_doc.copy()

            # Update or insert in MongoDB
            if collection is not None:
                try:
                    set_dict: Dict[str, Any] = {
                        "user_id": user_id,
                        "last_detected_at": last_detected,
                        "status": status,
                        "estimated_distance_m": dist_m,
                        "configured_threshold_m": thresh_m,
                        "severity": severity,
                    }
                    if resolved_at:
                        set_dict["resolved_at"] = resolved_at

                    collection.update_one(
                        {"event_id": event_id},
                        {
                            "$set": set_dict,
                            "$setOnInsert": {
                                "event_id": event_id,
                                "analysis_id": analysis_id,
                                "event_type": event_type,
                                "zone_name": zone_name,
                                "worker_track_id": wid,
                                "machine_track_id": mid,
                                "machine_type": machine_type,
                                "proximity_method": proximity_method,
                                "is_acknowledged": False,
                                "acknowledged_at": None,
                                "first_detected_at": first_detected,
                                "created_at": now,
                            }
                        },
                        upsert=True
                    )
                except Exception as e:
                    logger.error(f"Failed to persist safety event {event_id}: {e}")

    def get_safety_events(
        self,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        analysis_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get safety events filtered strictly by user_id and optional filters."""
        query: Dict[str, Any] = {}
        if user_id:
            query["user_id"] = user_id
        if status:
            query["status"] = status
        if analysis_id:
            query["analysis_id"] = analysis_id

        collection = get_collection("safety_events")
        if collection is not None:
            try:
                cursor = collection.find(query).sort("last_detected_at", -1).limit(limit)
                return [serialize_mongo_doc(doc) for doc in cursor]
            except Exception as e:
                logger.error(f"Failed to fetch safety events from MongoDB: {e}")

        # In-memory fallback
        results = []
        for doc in sorted(self._in_memory_safety_events.values(), key=lambda x: str(x.get("last_detected_at", "")), reverse=True):
            if user_id and doc.get("user_id") != user_id:
                continue
            if status and doc.get("status") != status:
                continue
            if analysis_id and doc.get("analysis_id") != analysis_id:
                continue
            results.append(serialize_mongo_doc(doc))
            if len(results) >= limit:
                break
        return results

    def get_active_safety_alerts(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get currently active safety alerts for user."""
        return self.get_safety_events(user_id=user_id, status="ACTIVE", limit=20)

    def get_safety_event(self, event_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get single safety event ensuring user ownership."""
        collection = get_collection("safety_events")
        if collection is not None:
            try:
                query = {"event_id": event_id}
                if user_id:
                    query["user_id"] = user_id
                doc = collection.find_one(query)
                if doc:
                    return serialize_mongo_doc(doc)
                return None
            except Exception as e:
                logger.error(f"Failed to fetch safety event {event_id}: {e}")

        doc = self._in_memory_safety_events.get(event_id)
        if doc:
            if user_id and doc.get("user_id") != user_id:
                return None
            return serialize_mongo_doc(doc)
        return None

    def acknowledge_safety_event(
        self,
        event_id: str,
        user_id: Optional[str] = None,
        acknowledged_by: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Acknowledge a safety alert.
        IMPORTANT: Acknowledging does NOT make an active hazard safe!
        The event's physical status remains ACTIVE until the worker leaves the zone.
        """
        event = self.get_safety_event(event_id, user_id=user_id)
        if not event:
            return None

        now = datetime.utcnow()
        ack_data = {
            "is_acknowledged": True,
            "acknowledged_at": now.isoformat(),
            "acknowledged_by": acknowledged_by or user_id or "user",
        }

        # Update in-memory
        if event_id in self._in_memory_safety_events:
            self._in_memory_safety_events[event_id].update(ack_data)

        # Update MongoDB
        collection = get_collection("safety_events")
        if collection is not None:
            try:
                collection.update_one(
                    {"event_id": event_id},
                    {"$set": {
                        "is_acknowledged": True,
                        "acknowledged_at": now,
                        "acknowledged_by": acknowledged_by or user_id or "user",
                    }},
                )
            except Exception as e:
                logger.error(f"Failed to update safety event acknowledgement: {e}")

        # Return updated event
        return self.get_safety_event(event_id, user_id=user_id)
