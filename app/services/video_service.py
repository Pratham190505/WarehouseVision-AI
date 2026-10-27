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
import threading
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
from app.vision.box_counting import BoxCountingEngine

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
        self._process_lock = threading.Lock()

    def process(self, video_path: str, result_dir: str,
                analysis_id: str = None, frame_skip: int = FRAME_SKIP,
                box_tracking_config: Dict[str, Any] = None) -> Dict[str, Any]:
        # This singleton owns mutable Ultralytics tracker and safety state.
        with self._process_lock:
            return self._process(video_path, result_dir, analysis_id, frame_skip, box_tracking_config)

    def _process(self, video_path: str, result_dir: str,
                 analysis_id: str = None, frame_skip: int = FRAME_SKIP,
                 box_tracking_config: Dict[str, Any] = None) -> Dict[str, Any]:
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
        result_path = None
        thumb_path = None
        h264_path = None

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

            if width <= 0 or height <= 0 or fps <= 0:
                raise ValueError("Uploaded video has invalid frame dimensions or frame rate.")
            if box_tracking_config is not None and not self.detector.is_ready():
                raise ValueError("The configured YOLO object detector is unavailable.")
            counting_engine = BoxCountingEngine(**(box_tracking_config or {})) if box_tracking_config is not None else None
            processing_step = 1 if counting_engine else frame_skip

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
            if not writer.isOpened():
                raise ValueError("Could not create the processed video output. Check codec and output permissions.")

            # Aggregate counts across frames
            aggregate_counts = {"workers": 0, "forklifts": 0, "boxes": 0, "pallets": 0}
            max_counts = {"workers": 0, "forklifts": 0, "boxes": 0, "pallets": 0}
            all_detections = []
            total_zone_violations = 0
            frame_idx = 0
            last_annotated = None
            latest_safety_alerts = []
            untracked_box_detections = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % processing_step == 0:
                    # Track objects
                    annotated, frame_stats = self.tracker.track_frame(
                        frame,
                        frame_idx,
                        confidence=float(os.getenv("BOX_TRACK_DETECTION_CONFIDENCE", "0.1")) if counting_engine else None,
                        annotate_box_detections=not bool(counting_engine),
                    )

                    # Get detections from tracker or fall back to detector
                    frame_detections = frame_stats.get("detections", [])
                    if not frame_detections and self.detector.is_ready():
                        det_result = self.detector.detect(frame)
                        frame_detections = det_result["detections"]

                    if counting_engine:
                        for detection in frame_detections:
                            if detection.get("class") != "box":
                                continue
                            if detection.get("track_id") is None:
                                counting_engine.note_untracked_detection()
                                continue
                            counting_engine.observe(
                                detection["track_id"], detection.get("bbox", [0, 0, 0, 0]),
                                detection.get("center", [0, 0]), detection.get("confidence", 0), frame_idx,
                            )
                        counting_engine.advance_frame(frame_idx)
                        summary = counting_engine.summary(frame_idx)
                        for track in counting_engine.visible_tracks(frame_idx):
                            x1, y1, x2, y2 = track["bbox"]
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (40, 220, 90), 3)
                            label = f"{track['box_id']} {track['average_confidence']:.2f}"
                            cv2.rectangle(annotated, (x1, max(0, y1 - 24)), (x1 + 180, y1), (40, 220, 90), -1)
                            cv2.putText(annotated, label, (x1 + 4, max(16, y1 - 7)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (15, 20, 25), 2)
                        cv2.rectangle(annotated, (8, 8), (440, 88), (20, 25, 35), -1)
                        cv2.putText(annotated, f"TOTAL UNIQUE BOXES: {summary['total_unique_boxes']}", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
                        cv2.putText(annotated, f"CURRENTLY VISIBLE BOXES: {summary['currently_visible_boxes']}", (20, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 220, 90), 2)

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
                    has_thumb = bool(cv2.imwrite(thumb_path, last_annotated))
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
            box_tracking_summary = counting_engine.summary(max(0, frame_idx - 1)) if counting_engine else None
            if box_tracking_summary is not None:
                box_tracking_summary["tracker_configuration"].update({
                    "tracker": "bytetrack.yaml",
                    "candidate_confidence": float(os.getenv("BOX_TRACK_DETECTION_CONFIDENCE", "0.1")),
                })
                tracking_summary["unique_boxes"] = box_tracking_summary["total_unique_boxes"]
                max_counts["boxes"] = box_tracking_summary["total_unique_boxes"]
            untracked_box_detections = counting_engine.untracked_detections if counting_engine else 0
            box_model_mapping = {}
            if counting_engine:
                class_names = getattr(self.detector.model, "names", {})
                for class_id, warehouse_label in self.detector.class_mapping.items():
                    if warehouse_label != "box":
                        continue
                    source_name = class_names.get(class_id, str(class_id)) if isinstance(class_names, dict) else (
                        class_names[class_id] if class_id < len(class_names) else str(class_id)
                    )
                    box_model_mapping[str(class_id)] = {"source_name": str(source_name), "warehouse_label": "box"}

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

            result = {
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
            if box_tracking_summary is not None:
                result["box_tracking"] = {
                    **box_tracking_summary,
                    "tracker": "Ultralytics ByteTrack",
                    "tracker_ids_are_session_scoped": True,
                    "detector_box_mapping": box_model_mapping,
                    "detection_model": self.detector.model_path,
                    "box_detection_source": self.detector.box_detection_source,
                    "untracked_box_detections": untracked_box_detections,
                    "tracking_warning": (
                        "Some box detections had no persistent tracker ID and were not counted."
                        if untracked_box_detections else None
                    ),
                    "output_filename": final_result_filename,
                    "thumbnail_filename": thumb_filename if has_thumb else None,
                }
                result["result_path"] = f"/api/box-tracking/{analysis_id}/result"
                result["thumbnail_path"] = (
                    f"/api/box-tracking/{analysis_id}/thumbnail" if has_thumb else None
                )
            return result

        except Exception as e:
            processing_time = round(time.time() - start_time, 2)
            logger.error(f"Video processing failed: {e}", exc_info=True)
            if writer:
                writer.release()
                writer = None
            for partial_path in (result_path, thumb_path, h264_path):
                if partial_path and os.path.isfile(partial_path):
                    try:
                        os.remove(partial_path)
                    except OSError as cleanup_error:
                        logger.warning("Could not remove partial output %s: %s", partial_path, cleanup_error)
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
