"""
Object tracker for warehouse video analysis.

Uses YOLO's built-in tracking (ByteTrack) for multi-object tracking
across video frames. Tracks unique object IDs to count distinct
workers, forklifts, boxes, and pallets.
"""

import logging
from typing import Dict, Any, List, Optional, Set, Tuple

import cv2
import numpy as np

from app.vision.detector import DEFAULT_CLASS_MAPPING, WAREHOUSE_LABELS

logger = logging.getLogger(__name__)


class WarehouseTracker:
    """Multi-object tracker for warehouse video analysis."""

    def __init__(self, detector):
        """
        Args:
            detector: WarehouseDetector instance with loaded YOLO model.
        """
        self.detector = detector
        self.class_mapping = DEFAULT_CLASS_MAPPING.copy()

        # Track unique IDs per class
        self.unique_ids: Dict[str, Set[int]] = {
            "worker": set(),
            "forklift": set(),
            "box": set(),
            "pallet": set(),
        }

        # Frame-level data for analytics
        self.frame_data: List[Dict[str, Any]] = []
        self.total_frames = 0
        self.processed_frames = 0

    def reset(self):
        """Reset tracker state for a new video."""
        for key in self.unique_ids:
            self.unique_ids[key] = set()
        self.frame_data = []
        self.total_frames = 0
        self.processed_frames = 0

    def track_frame(self, frame: np.ndarray, frame_idx: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Track objects in a single video frame using YOLO's built-in tracker.

        Args:
            frame: BGR video frame.
            frame_idx: Frame index number.

        Returns:
            Tuple of (annotated_frame, frame_stats).
        """
        if not self.detector.is_ready():
            return frame, {"frame": frame_idx, "counts": {}, "tracking_ids": {}}

        try:
            results = self.detector.model.track(
                frame,
                conf=self.detector.confidence,
                persist=True,
                verbose=False,
                tracker="bytetrack.yaml",
            )
        except Exception as e:
            logger.warning(f"Tracking failed on frame {frame_idx}, falling back to detection: {e}")
            annotated, detections, counts = self.detector.detect_frame(frame)
            frame_stats = {
                "frame": frame_idx,
                "counts": counts,
                "tracking_ids": {},
            }
            self.frame_data.append(frame_stats)
            self.processed_frames += 1
            return annotated, frame_stats

        annotated = frame.copy()
        counts = {"workers": 0, "forklifts": 0, "boxes": 0, "pallets": 0}
        tracking_ids: Dict[str, List[int]] = {
            "worker": [], "forklift": [], "box": [], "pallet": []
        }

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                warehouse_class = self.class_mapping.get(cls_id)
                if warehouse_class is None:
                    continue

                # Get track ID if available
                track_id = None
                if box.id is not None:
                    track_id = int(box.id[0])
                    self.unique_ids[warehouse_class].add(track_id)
                    tracking_ids[warehouse_class].append(track_id)

                # Count
                count_key = warehouse_class + "s" if warehouse_class != "box" else "boxes"
                counts[count_key] = counts.get(count_key, 0) + 1

                # Draw annotation
                info = WAREHOUSE_LABELS.get(warehouse_class, {"color": (200, 200, 200)})
                color = info["color"]

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                label = f"{warehouse_class}"
                if track_id is not None:
                    label += f" #{track_id}"
                label += f" {conf:.2f}"

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        frame_stats = {
            "frame": frame_idx,
            "counts": counts,
            "tracking_ids": {k: v for k, v in tracking_ids.items()},
        }
        self.frame_data.append(frame_stats)
        self.processed_frames += 1

        return annotated, frame_stats

    def get_summary(self) -> Dict[str, Any]:
        """Get tracking summary after video processing."""
        return {
            "unique_workers": len(self.unique_ids["worker"]),
            "unique_forklifts": len(self.unique_ids["forklift"]),
            "unique_boxes": len(self.unique_ids["box"]),
            "unique_pallets": len(self.unique_ids["pallet"]),
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "frame_data": self.frame_data[-100:] if len(self.frame_data) > 100 else self.frame_data,
        }
