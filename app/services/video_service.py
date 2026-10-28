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
from typing import Dict, Any, Optional

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

# Process every frame by default for complete frame-by-frame annotation
FRAME_SKIP = 1


def get_ffmpeg_binary() -> Optional[str]:
    """Get path to a working FFmpeg binary if available."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:
        pass
    import shutil
    return shutil.which("ffmpeg")


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
        raw_path = None

        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")

            # Video properties
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            if fps <= 0 or np.isnan(fps):
                fps = 30.0
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

            # Output video paths and writer setup
            final_result_filename = f"{analysis_id}_processed.mp4"
            result_path = os.path.join(result_dir, final_result_filename)
            ffmpeg_bin = get_ffmpeg_binary()

            # If FFmpeg is available, write raw frames to an intermediate file and transcode to H.264
            if ffmpeg_bin:
                raw_filename = f"{analysis_id}_raw.mp4"
                raw_path = os.path.join(result_dir, raw_filename)
                target_writer_path = raw_path
            else:
                target_writer_path = result_path

            # Attempt to open VideoWriter using supported codecs
            # Avoid "avc1" / "H264" in OpenCV VideoWriter to prevent OpenH264 DLL loading errors.
            # Frames are written cleanly with mp4v and then transcoded to browser-compliant H.264 via FFmpeg.
            codecs_to_try = ["mp4v", "XVID", "MJPG"]
            for codec in codecs_to_try:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    w = cv2.VideoWriter(target_writer_path, fourcc, fps, (width, height))
                    if w and w.isOpened():
                        writer = w
                        break
                    if w:
                        w.release()
                except Exception:
                    pass

            if not writer or not writer.isOpened():
                raise ValueError("Could not create the processed video output. Check codec and output permissions.")

            # Aggregate counts across frames
            aggregate_counts = {"workers": 0, "forklifts": 0, "boxes": 0, "pallets": 0}
            max_counts = {"workers": 0, "forklifts": 0, "boxes": 0, "pallets": 0}
            all_detections = []
            total_zone_violations = 0
            frame_idx = 0
            written_frames = 0
            first_annotated = None
            last_annotated = None
            latest_safety_alerts = []
            untracked_box_detections = 0

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if frame_idx % processing_step == 0:
                    # Track objects on current frame
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
                        val = frame_counts.get(key, 0)
                        max_counts[key] = max(max_counts[key], val)

                    if first_annotated is None:
                        first_annotated = annotated.copy()
                    last_annotated = annotated
                    writer.write(annotated)
                    written_frames += 1
                else:
                    # Draw zones on skipped frame so it doesn't freeze or drop
                    curr_annotated = self.zone_detector.draw_zones(frame.copy())
                    if first_annotated is None:
                        first_annotated = curr_annotated.copy()
                    last_annotated = curr_annotated
                    writer.write(curr_annotated)
                    written_frames += 1

                frame_idx += 1

            # Finalize writer before processing thumbnail or transcoding
            if writer:
                writer.release()
                writer = None

            if written_frames == 0:
                raise ValueError("No video frames could be processed.")

            # Transcode video to H.264 for universal browser playback if ffmpeg is available
            if ffmpeg_bin and raw_path and os.path.exists(raw_path):
                cmd = [
                    ffmpeg_bin, "-y",
                    "-i", raw_path,
                    "-r", str(fps),
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    result_path,
                ]
                proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=120)
                if proc.returncode == 0 and os.path.isfile(result_path) and os.path.getsize(result_path) > 0:
                    try:
                        os.remove(raw_path)
                    except OSError:
                        pass
                    logger.info(f"Successfully encoded {final_result_filename} to H.264")
                else:
                    err_msg = proc.stderr.decode("utf-8", errors="ignore") if proc.stderr else "Unknown error"
                    logger.warning(f"FFmpeg transcoding failed ({err_msg}); falling back to raw output")
                    if os.path.isfile(raw_path):
                        if os.path.exists(result_path):
                            try:
                                os.remove(result_path)
                            except OSError:
                                pass
                        os.rename(raw_path, result_path)

            # Verification of output video file if written to disk
            if os.path.isfile(result_path):
                if os.path.getsize(result_path) == 0:
                    raise ValueError(f"Output video file {final_result_filename} is empty.")
                verify_cap = cv2.VideoCapture(result_path)
                if not verify_cap.isOpened():
                    raise ValueError(f"Final processed video {final_result_filename} cannot be opened by decoder.")
                out_frame_count = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                ret_read, frame_sample = verify_cap.read()
                verify_cap.release()
                if not ret_read or frame_sample is None:
                    raise ValueError(f"Final processed video {final_result_filename} cannot be decoded.")
                if out_frame_count <= 1 and total_frames > 1 and written_frames > 1:
                    raise ValueError(f"Processed video has only {out_frame_count} frame(s), expected full video.")

            # Generate thumbnail poster from first (or last) annotated frame
            thumb_filename = f"{analysis_id}_thumb.jpg"
            thumb_path = os.path.join(result_dir, thumb_filename)
            has_thumb = False
            thumb_source = first_annotated if first_annotated is not None else last_annotated
            if thumb_source is not None:
                try:
                    has_thumb = bool(cv2.imwrite(thumb_path, thumb_source))
                except Exception as e:
                    logger.warning(f"Could not save video thumbnail: {e}")

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

            processed_video_url = (
                f"/api/box-tracking/{analysis_id}/result"
                if counting_engine
                else f"/api/analysis/{analysis_id}/video"
            )
            thumbnail_url = (
                f"/api/box-tracking/{analysis_id}/thumbnail"
                if counting_engine
                else (f"/api/analysis/{analysis_id}/thumbnail" if has_thumb else None)
            )

            result = {
                "analysis_id": analysis_id,
                "media_type": "video",
                "status": "completed",
                "processing_time": processing_time,
                "object_counts": final_counts,
                "analytics": analytics,
                "tracking": tracking_summary,
                "safety_alerts": latest_safety_alerts,
                "safety_alerts_count": len(all_incidents),
                "safety_incidents": all_incidents,
                "processed_video_url": processed_video_url,
                "result_path": processed_video_url,
                "video_filename": final_result_filename,
                "thumbnail_path": thumbnail_url,
                "video_info": {
                    "fps": fps,
                    "total_frames": total_frames,
                    "processed_frames": tracking_summary.get("processed_frames", written_frames),
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
            return result

        except Exception as e:
            processing_time = round(time.time() - start_time, 2)
            logger.error(f"Video processing failed: {e}", exc_info=True)
            if writer:
                writer.release()
                writer = None
            for partial_path in (result_path, thumb_path, raw_path):
                if partial_path and os.path.isfile(partial_path):
                    try:
                        os.remove(partial_path)
                    except OSError as cleanup_error:
                        logger.warning("Could not remove partial output %s: %s", partial_path, cleanup_error)
            return {
                "analysis_id": analysis_id,
                "media_type": "video",
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
