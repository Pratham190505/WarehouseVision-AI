"""
Video processing service for WarehouseVision AI.

Handles the complete pipeline for video analysis:
Video → Frame Extraction → Detection + Tracking → Analytics → Processed Video

Processes frames sequentially without loading the entire video into memory.
"""

import os
import time
import uuid
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np

from app.vision.detector import WarehouseDetector
from app.vision.tracker import WarehouseTracker
from app.vision.zone_detection import ZoneDetector
from app.vision.occupancy import OccupancyAnalyzer
from app.vision.inventory import InventoryAnalyzer
from app.vision.analytics import WarehouseAnalytics
from app.vision.safety_zone import SafetyZoneEvaluator

logger = logging.getLogger(__name__)

# Process every Nth frame to balance speed and accuracy
FRAME_SKIP = 2  # Process every 2nd frame by default


class VideoService:
    """Service for processing warehouse videos."""

    def __init__(self):
        self.detector = WarehouseDetector()
        self.tracker = WarehouseTracker(self.detector)
        self.zone_detector = ZoneDetector()
        self.occupancy_analyzer = OccupancyAnalyzer()
        self.inventory_analyzer = InventoryAnalyzer()
        self.analytics_engine = WarehouseAnalytics()
        self.safety_evaluator = SafetyZoneEvaluator()

    def process(self, video_path: str, result_dir: str,
                analysis_id: str = None, frame_skip: int = FRAME_SKIP) -> Dict[str, Any]:
        """
        Process a warehouse video through the full analysis pipeline.

        Args:
            video_path: Path to the uploaded video.
            result_dir: Directory to save processed video.
            analysis_id: Optional pre-assigned analysis ID.
            frame_skip: Process every Nth frame.

        Returns:
            Complete analysis result dict.
        """
        start_time = time.time()
        analysis_id = analysis_id or str(uuid.uuid4())

        cap = None
        writer = None

        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")

            # Video properties
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Reset tracker and safety evaluator
            self.tracker.reset()
            self.tracker.total_frames = total_frames
            self.safety_evaluator.reset()

            # Set up restricted zones
            if not self.zone_detector.has_zones:
                self.zone_detector.add_default_zones(width, height)

            # Output video writer
            result_filename = f"{analysis_id}_processed.mp4"
            result_path = os.path.join(result_dir, result_filename)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(result_path, fourcc, fps, (width, height))

            # Aggregate counts across frames
            aggregate_counts = {"workers": 0, "forklifts": 0, "boxes": 0, "pallets": 0}
            max_counts = {"workers": 0, "forklifts": 0, "boxes": 0, "pallets": 0}
            all_detections = []
            total_zone_violations = 0
            frame_idx = 0
            last_annotated = None
            latest_safety_alerts = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip == 0:
                    # Track objects
                    annotated, frame_stats = self.tracker.track_frame(frame, frame_idx)

                    # Get detections from tracker or fall back to detector
                    frame_detections = frame_stats.get("detections", [])
                    if not frame_detections and self.detector.is_ready():
                        det_result = self.detector.detect(frame)
                        frame_detections = det_result["detections"]

                    # Check zone violations
                    frame_counts = frame_stats.get("counts", {})
                    zone_result = self.zone_detector.check_violations(frame_detections)
                    total_zone_violations += zone_result.get("total_violations", 0)

                    # Draw zones
                    annotated = self.zone_detector.draw_zones(annotated)
                    if zone_result["violations"]:
                        annotated = self.zone_detector.annotate_violations(
                            annotated, zone_result["violations"]
                        )

                    # Check Machine Proximity & Restricted Zone Safety
                    safety_eval = self.safety_evaluator.evaluate_detections(
                        detections=frame_detections,
                        frame_idx=frame_idx,
                        is_video=True,
                        image_shape=(height, width),
                        zones=self.zone_detector.zones,
                    )
                    frame_alerts = safety_eval.get("alerts", [])
                    if frame_alerts:
                        latest_safety_alerts = frame_alerts

                    # Annotate Machine Safety Violations (danger zone, worker red highlight, line)
                    annotated = self.safety_evaluator.annotate_frame(
                        annotated, frame_alerts, frame_detections
                    )

                    # Update max counts
                    for key in max_counts:
                        count_key = key
                        val = frame_counts.get(count_key, 0)
                        max_counts[key] = max(max_counts[key], val)

                    last_annotated = annotated
                    writer.write(annotated)
                else:
                    # Write original frame for skipped frames
                    if last_annotated is not None:
                        writer.write(last_annotated)
                    else:
                        writer.write(frame)

                frame_idx += 1

            # Finalize writer before processing thumbnail or transcoding
            if writer:
                writer.release()
                writer = None

            # Generate thumbnail poster from last annotated frame
            thumb_filename = f"{analysis_id}_thumb.jpg"
            thumb_path = os.path.join(result_dir, thumb_filename)
            has_thumb = False
            if last_annotated is not None:
                try:
                    cv2.imwrite(thumb_path, last_annotated)
                    has_thumb = True
                except Exception as e:
                    logger.warning(f"Could not save video thumbnail: {e}")

            # Transcode video to H.264 for universal browser playback if ffmpeg is available
            final_result_filename = result_filename
            try:
                ffmpeg_bin = None
                try:
                    import imageio_ffmpeg
                    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
                except Exception:
                    import shutil
                    ffmpeg_bin = shutil.which("ffmpeg")

                if ffmpeg_bin and os.path.exists(result_path):
                    h264_filename = f"{analysis_id}_web.mp4"
                    h264_path = os.path.join(result_dir, h264_filename)
                    cmd = [
                        ffmpeg_bin, "-y",
                        "-i", result_path,
                        "-c:v", "libx264",
                        "-preset", "veryfast",
                        "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                        h264_path,
                    ]
                    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
                    if proc.returncode == 0 and os.path.exists(h264_path) and os.path.getsize(h264_path) > 0:
                        final_result_filename = h264_filename
                        logger.info(f"Successfully transcoded {result_filename} to H.264 {h264_filename}")
            except Exception as ex:
                logger.warning(f"Video transcoding skipped: {ex}")

            # Get tracking summary
            tracking_summary = self.tracker.get_summary()

            # Use max per-frame counts as the reported counts
            final_counts = max_counts

            # Occupancy from last processed frame
            occupancy_result = {"occupancy_percent": 0, "space_utilization_percent": 0}
            if last_annotated is not None and self.detector.is_ready():
                last_det = self.detector.detect(frame if frame is not None else last_annotated)
                occupancy_result = self.occupancy_analyzer.analyze(
                    last_det["detections"], (height, width)
                )

            # Inventory from last frame
            inventory_result = {"box_count": 0, "pallet_count": 0, "estimated_items": 0}
            if last_annotated is not None and self.detector.is_ready():
                inventory_result = self.inventory_analyzer.analyze(
                    last_det["detections"], (height, width)
                )

            # Compute analytics
            zone_result_final = {"total_violations": total_zone_violations, "violations": []}
            analytics = self.analytics_engine.compute(
                object_counts=final_counts,
                occupancy_data=occupancy_result,
                zone_violations=zone_result_final,
                inventory_data=inventory_result,
                tracking_data=tracking_summary,
            )

            # Aggregate all safety incidents tracked throughout the video
            all_incidents = []
            for inc in list(self.safety_evaluator.active_incidents.values()) + self.safety_evaluator.resolved_incidents:
                if inc.consecutive_frames >= self.safety_evaluator.config.persistence_frames:
                    m_track = inc.machine_id.split("#")[-1] if (inc.machine_id and "#" in inc.machine_id) else inc.machine_id
                    all_incidents.append({
                        "incident_id": inc.incident_id,
                        "event_type": inc.event_type,
                        "zone_name": inc.zone_name,
                        "worker_id": inc.worker_id,
                        "worker_track_id": inc.worker_id.replace("Worker #", ""),
                        "machine_id": inc.machine_id,
                        "machine_track_id": m_track,
                        "machine_type": inc.machine_type,
                        "severity": inc.severity,
                        "status": "ACTIVE" if inc.is_active else "RESOLVED",
                        "first_detected_at": inc.first_detected_at.isoformat(),
                        "last_detected_at": inc.last_detected_at.isoformat(),
                        "resolved_at": inc.resolved_at.isoformat() if inc.resolved_at else None,
                        "first_frame": inc.first_frame,
                        "last_frame": inc.last_frame,
                        "consecutive_frames": inc.consecutive_frames,
                        "estimated_distance_m": inc.min_estimated_distance_m,
                        "configured_threshold_m": inc.configured_threshold_m if inc.min_estimated_distance_m is not None else None,
                        "proximity_method": inc.proximity_method,
                        "proximity_label": inc.last_proximity_label,
                        "is_acknowledged": inc.is_acknowledged,
                        "acknowledged_at": inc.acknowledged_at.isoformat() if inc.acknowledged_at else None,
                    })

            processing_time = round(time.time() - start_time, 2)

            return {
                "analysis_id": analysis_id,
                "status": "completed",
                "processing_time": processing_time,
                "object_counts": final_counts,
                "analytics": analytics,
                "tracking": tracking_summary,
                "safety_alerts": latest_safety_alerts,
                "safety_alerts_count": len(all_incidents),
                "safety_incidents": all_incidents,
                "result_path": f"/results/videos/{final_result_filename}",
                "thumbnail_path": f"/results/videos/{thumb_filename}" if has_thumb else None,
                "video_info": {
                    "fps": fps,
                    "total_frames": total_frames,
                    "processed_frames": tracking_summary.get("processed_frames", 0),
                    "width": width,
                    "height": height,
                    "duration_seconds": round(total_frames / fps, 1) if fps > 0 else 0,
                },
                "zone_violations": zone_result_final,
                "occupancy": occupancy_result,
                "inventory": inventory_result,
            }

        except Exception as e:
            processing_time = round(time.time() - start_time, 2)
            logger.error(f"Video processing failed: {e}", exc_info=True)
            return {
                "analysis_id": analysis_id,
                "status": "failed",
                "processing_time": processing_time,
                "error_message": str(e),
                "object_counts": {"workers": 0, "forklifts": 0, "boxes": 0, "pallets": 0},
                "analytics": {
                    "occupancy": 0,
                    "space_utilization": 0,
                    "restricted_zone_violations": 0,
                    "insights": [f"Video analysis failed: {str(e)}"],
                },
            }
        finally:
            if cap:
                cap.release()
            if writer:
                writer.release()
