"""Per-video confirmation and unique counting for tracked box detections."""

import os
from typing import Any, Dict, List, Optional


class BoxCountingEngine:
    """Count confirmed tracker IDs once and report the same current-visible metric."""

    TRACK_BUFFER_FRAMES = 30

    def __init__(
        self,
        min_observations: Optional[int] = None,
        min_average_confidence: Optional[float] = None,
        visibility_window_frames: Optional[int] = None,
        track_buffer_frames: Optional[int] = None,
    ):
        min_observations = min_observations if min_observations is not None else int(os.getenv("BOX_TRACK_MIN_OBSERVATIONS", "3"))
        min_average_confidence = min_average_confidence if min_average_confidence is not None else float(os.getenv("BOX_TRACK_MIN_AVERAGE_CONFIDENCE", "0.35"))
        visibility_window_frames = visibility_window_frames if visibility_window_frames is not None else int(os.getenv("BOX_VISIBLE_WINDOW_FRAMES", "2"))
        track_buffer_frames = track_buffer_frames if track_buffer_frames is not None else int(os.getenv("BOX_TRACK_BUFFER_FRAMES", str(self.TRACK_BUFFER_FRAMES)))
        if min_observations < 1:
            raise ValueError("Minimum track observations must be at least one.")
        if not 0 <= min_average_confidence <= 1:
            raise ValueError("Minimum average confidence must be between 0 and 1.")
        if visibility_window_frames < 0 or track_buffer_frames < 1:
            raise ValueError("Track frame windows must be non-negative and buffer must be positive.")
        self.min_observations = min_observations
        self.min_average_confidence = min_average_confidence
        self.visibility_window_frames = visibility_window_frames
        self.track_buffer_frames = track_buffer_frames
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.confirmed_ids: Dict[int, str] = {}
        self.current_frame = 0
        self.untracked_detections = 0

    @property
    def total_unique_boxes(self) -> int:
        return len(self.confirmed_ids)

    def observe(self, tracker_id: int, bbox, center, confidence: float, frame_index: int) -> Dict[str, Any]:
        """Record one tracked detection; confirm only after minimum evidence."""
        tracker_id = int(tracker_id)
        confidence = max(0.0, min(1.0, float(confidence)))
        frame_index = int(frame_index)
        track = self.tracks.get(tracker_id)
        if track is None:
            track = {
                "tracker_id": tracker_id,
                "box_id": None,
                "first_seen_frame": frame_index,
                "last_seen_frame": frame_index,
                "observations": 1,
                "confidence_sum": confidence,
                "average_confidence": round(confidence, 3),
                "last_confidence": confidence,
                "bbox": list(map(int, bbox)),
                "center": [float(center[0]), float(center[1])],
                "confirmed": False,
            }
            self.tracks[tracker_id] = track

        if track["last_seen_frame"] != frame_index:
            track["observations"] += 1
            track["confidence_sum"] += confidence
        track["last_seen_frame"] = frame_index
        track["last_confidence"] = confidence
        track["average_confidence"] = round(
            track["confidence_sum"] / max(1, track["observations"]), 3
        )
        track["bbox"] = list(map(int, bbox))
        track["center"] = [float(center[0]), float(center[1])]

        if (
            not track["confirmed"]
            and track["observations"] >= self.min_observations
            and track["average_confidence"] >= self.min_average_confidence
        ):
            box_id = f"BOX-{self.total_unique_boxes + 1:03d}"
            self.confirmed_ids[tracker_id] = box_id
            track["box_id"] = box_id
            track["confirmed"] = True

        self.current_frame = max(self.current_frame, frame_index)
        return track

    def note_untracked_detection(self) -> None:
        """Track detector fallback events that had no stable ByteTrack ID."""
        self.untracked_detections += 1

    def advance_frame(self, frame_index: int) -> None:
        """Advance visibility and discard only expired, never-confirmed false tracks."""
        self.current_frame = max(self.current_frame, int(frame_index))
        expired = [
            tracker_id
            for tracker_id, track in self.tracks.items()
            if not track["confirmed"]
            and self.current_frame - track["last_seen_frame"] > self.track_buffer_frames
        ]
        for tracker_id in expired:
            del self.tracks[tracker_id]

    def visible_tracks(self, frame_index: Optional[int] = None) -> List[Dict[str, Any]]:
        frame_index = self.current_frame if frame_index is None else int(frame_index)
        return [
            track for track in self.tracks.values()
            if track["confirmed"]
            and frame_index - track["last_seen_frame"] <= self.visibility_window_frames
        ]

    def summary(self, frame_index: Optional[int] = None) -> Dict[str, Any]:
        frame_index = self.current_frame if frame_index is None else int(frame_index)
        confirmed = sorted(
            (track for track in self.tracks.values() if track["confirmed"]),
            key=lambda track: int(track["box_id"].split("-")[1]),
        )
        tracks = [
            {
                "tracker_id": track["tracker_id"],
                "box_id": track["box_id"],
                "first_seen_frame": track["first_seen_frame"],
                "last_seen_frame": track["last_seen_frame"],
                "observations": track["observations"],
                "average_confidence": track["average_confidence"],
                "last_confidence": track["last_confidence"],
                "last_bbox": track["bbox"],
            }
            for track in confirmed
        ]
        return {
            "total_unique_boxes": len(self.confirmed_ids),
            "currently_visible_boxes": len(self.visible_tracks(frame_index)),
            "confirmed_tracks": tracks,
            "tracker_configuration": {
                "min_observations": self.min_observations,
                "min_average_confidence": self.min_average_confidence,
                "visibility_window_frames": self.visibility_window_frames,
                "track_buffer_frames": self.track_buffer_frames,
            },
            "untracked_box_detections": self.untracked_detections,
        }
