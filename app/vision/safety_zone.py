"""
Machine Proximity Safety Zone & Alert Engine.

Responsibilities:
- Configure safety thresholds per machine type and machine ID.
- Determine whether a worker enters a machine's danger zone.
- Support calibrated ground-plane metric distance estimation OR image-space/polygon proximity.
- Guarantee that metric distances in metres are ONLY reported when valid calibration is provided.
- Maintain incident state across video frames (active, persistence, resolve, cooldown).
- Avoid generating duplicate events for every frame of the same continuous incident.
- Provide visual annotation for media: danger zones, worker red highlights, hazard connection lines.

DISCLAIMER:
This feature is a visual safety assistance and awareness prototype.
It is NOT a certified industrial safety system (such as ISO 13849/IEC 62061 safety PLCs)
and must NOT replace mandatory warehouse safety procedures, designated walkways, or physical guarding.
"""

import math
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Set

import cv2
import numpy as np

from app.vision.safety_config import SafetyConfig, get_safety_config

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


class SafetyIncident:
    """
    Tracks a single machine-worker proximity incident across time/frames.
    Prevents alert flapping and multiple DB events for continuous events.
    """

    def __init__(
        self,
        incident_id: str,
        worker_id: str,
        machine_id: str,
        machine_type: str,
        configured_threshold_m: float,
        proximity_method: str,
        first_frame: int = 0,
    ):
        self.incident_id = incident_id
        self.worker_id = worker_id
        self.machine_id = machine_id
        self.machine_type = machine_type
        self.configured_threshold_m = configured_threshold_m
        self.proximity_method = proximity_method

        self.first_detected_at: datetime = _utc_now()
        self.last_detected_at: datetime = _utc_now()
        self.first_frame: int = first_frame
        self.last_frame: int = first_frame

        self.consecutive_frames: int = 0
        self.is_active: bool = False  # Becomes True once persistence threshold is met
        self.is_resolved: bool = False
        self.resolved_at: Optional[datetime] = None

        self.is_acknowledged: bool = False
        self.acknowledged_at: Optional[datetime] = None
        self.acknowledged_by: Optional[str] = None

        self.severity: str = "HIGH"
        self.last_estimated_distance_m: Optional[float] = None
        self.min_estimated_distance_m: Optional[float] = None
        self.last_proximity_label: str = ""
        self.db_persisted: bool = False


class SafetyZoneEvaluator:
    """
    Evaluates worker-machine proximity and manages safety incidents.
    Supports both single-image evaluation and stateful video tracking.
    """

    def __init__(self, config: Optional[SafetyConfig] = None):
        self.config = config or get_safety_config()

        # Stateful incident tracking for videos: key is (worker_id, machine_id)
        self.active_incidents: Dict[Tuple[str, str], SafetyIncident] = {}
        self.resolved_incidents: List[SafetyIncident] = []
        self.cooldown_tracker: Dict[Tuple[str, str], float] = {}

    def reset(self):
        """Reset stateful tracker for a new video analysis."""
        self.active_incidents.clear()
        self.resolved_incidents.clear()
        self.cooldown_tracker.clear()

    @staticmethod
    def get_ground_contact_point(bbox: List[int]) -> Tuple[int, int]:
        """
        Calculate the bottom-center point of a bounding box.
        This approximates the physical object's contact with the warehouse ground plane.
        bbox format: [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int(y2)

    def calculate_distance_meters(
        self,
        worker_pos: Tuple[int, int],
        machine_pos: Tuple[int, int],
    ) -> Optional[float]:
        """
        Estimate real-world Euclidean distance in metres ONLY if camera calibration is valid.
        If camera calibration is not enabled or incomplete, returns None to avoid fabricating metrics.
        """
        calib = self.config.camera_calibration
        if not calib or not calib.get("enabled", False):
            return None

        method = calib.get("method")
        try:
            if method == "homography" and calib.get("homography_matrix") is not None:
                h_mat = np.array(calib["homography_matrix"], dtype=np.float64)
                w_pt = np.array([[[worker_pos[0], worker_pos[1]]]], dtype=np.float64)
                m_pt = np.array([[[machine_pos[0], machine_pos[1]]]], dtype=np.float64)

                w_world = cv2.perspectiveTransform(w_pt, h_mat)[0][0]
                m_world = cv2.perspectiveTransform(m_pt, h_mat)[0][0]

                dist = float(np.hypot(w_world[0] - m_world[0], w_world[1] - m_world[1]))
                return round(dist, 2)

            elif method == "pixels_per_meter" and calib.get("pixels_per_meter"):
                ppm = float(calib["pixels_per_meter"])
                if ppm > 0:
                    pixel_dist = math.hypot(worker_pos[0] - machine_pos[0], worker_pos[1] - machine_pos[1])
                    return round(pixel_dist / ppm, 2)

        except Exception as e:
            logger.warning(f"Error computing calibrated distance: {e}")

        return None

    def get_machine_danger_zone(
        self,
        machine_bbox: List[int],
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Define the safety zone footprint around a machine.
        When uncalibrated, uses an image-space elliptical danger zone projected around the machine base.
        """
        x1, y1, x2, y2 = machine_bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        cx = int((x1 + x2) / 2)
        cy = int(y2)  # Base point

        expansion = self.config.image_space_expansion_ratio
        # Major and minor axis for ground contact ellipse
        axis_x = int((width / 2) * expansion)
        axis_y = int((height * 0.35) * expansion)

        return {
            "type": "ellipse",
            "center": (cx, cy),
            "axes": (axis_x, axis_y),
            "bbox_base": (cx, cy),
        }

    def is_point_in_machine_zone(
        self,
        point: Tuple[int, int],
        machine_bbox: List[int],
        calibrated_distance_m: Optional[float] = None,
        threshold_m: float = 2.0,
    ) -> Tuple[bool, str]:
        """
        Determine whether a worker's ground contact point penetrates a machine's danger zone.
        Returns: (in_danger_zone: bool, proximity_description: str)
        """
        # 1. If camera calibration distance is available, check against metric threshold
        if calibrated_distance_m is not None:
            in_zone = calibrated_distance_m <= threshold_m
            desc = f"{calibrated_distance_m:.1f}m (Threshold: {threshold_m:.1f}m)"
            return in_zone, desc

        # 2. Configurable polygon check if user defined specific danger zone polygons
        if self.config.danger_zone_polygons:
            for pzone in self.config.danger_zone_polygons:
                poly = np.array(pzone.get("polygon", []), dtype=np.int32)
                if len(poly) >= 3:
                    if cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False) >= 0:
                        zone_name = pzone.get("name", "Restricted Area")
                        return True, f"Inside configured polygon zone: {zone_name}"

        # 3. Uncalibrated image-space elliptical proximity zone around machine base
        zone = self.get_machine_danger_zone(machine_bbox)
        cx, cy = zone["center"]
        rx, ry = zone["axes"]
        rx = max(1, rx)
        ry = max(1, ry)

        # Normalized distance to ellipse: ((x - cx)^2 / rx^2) + ((y - cy)^2 / ry^2)
        norm_dist = ((point[0] - cx) ** 2) / (rx ** 2) + ((point[1] - cy) ** 2) / (ry ** 2)
        in_zone = norm_dist <= 1.0

        if in_zone:
            desc = "Worker inside machine perimeter (Image-space proximity zone)"
        else:
            desc = "Worker outside machine perimeter"

        return in_zone, desc

    def evaluate_detections(
        self,
        detections: List[Dict[str, Any]],
        frame_idx: int = 0,
        is_video: bool = False,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate all workers against all machines in a single frame/image.

        Args:
            detections: List of detection dicts with 'bbox', 'class', and optional 'track_id'.
            frame_idx: Current frame index (for videos).
            is_video: Whether this is part of continuous video analysis.
            image_shape: (height, width) of the image.

        Returns:
            Dict containing:
            - alerts: List of active danger alerts for this frame
            - active_incidents: List of currently active SafetyIncident objects
            - resolved_incidents: Incidents resolved in this frame
            - pairs_evaluated: Total worker-machine pairs inspected
        """
        # Separate workers and hazardous machines
        workers = []
        machines = []

        for i, det in enumerate(detections):
            if not isinstance(det, dict):
                continue
            bbox = det.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue

            cls_name = det.get("class", "").lower()
            track_id = det.get("track_id")

            # Determine human-readable identifier
            if self.config.is_worker(cls_name):
                wid = f"Worker #{track_id}" if track_id is not None else f"Worker #{i + 1}"
                workers.append({"det": det, "id": wid, "raw_id": track_id or str(i + 1)})
            elif self.config.is_machine(cls_name):
                mid = f"{cls_name.capitalize()} #{track_id}" if track_id is not None else f"{cls_name.capitalize()} #{i + 1}"
                machines.append({"det": det, "id": mid, "raw_id": track_id or str(i + 1), "type": cls_name})

        current_breach_pairs: Set[Tuple[str, str]] = set()
        alerts = []
        pairs_evaluated = 0
        now = _utc_now()

        for w in workers:
            w_bbox = w["det"]["bbox"]
            w_pos = self.get_ground_contact_point(w_bbox)
            w_id = w["id"]

            for m in machines:
                pairs_evaluated += 1
                m_bbox = m["det"]["bbox"]
                m_pos = self.get_ground_contact_point(m_bbox)
                m_id = m["id"]
                m_type = m["type"]
                pair_key = (w_id, m_id)

                # Get configured threshold for this machine
                threshold_m = self.config.get_threshold_m(m_type, str(m["raw_id"]))

                # Estimate distance if calibrated
                dist_m = self.calculate_distance_meters(w_pos, m_pos)
                proximity_method = (
                    "camera_calibration_ground_plane"
                    if dist_m is not None
                    else "image_space_proximity"
                )

                in_danger_zone, prox_label = self.is_point_in_machine_zone(
                    point=w_pos,
                    machine_bbox=m_bbox,
                    calibrated_distance_m=dist_m,
                    threshold_m=threshold_m,
                )

                if in_danger_zone:
                    current_breach_pairs.add(pair_key)

                    # Determine severity based on distance or deep penetration
                    if dist_m is not None:
                        if dist_m <= (threshold_m * 0.5):
                            severity = "CRITICAL"
                        else:
                            severity = "HIGH"
                    else:
                        severity = "HIGH"

                    alert_data = {
                        "worker_id": w_id,
                        "worker_track_id": w["raw_id"],
                        "machine_id": m_id,
                        "machine_track_id": m["raw_id"],
                        "machine_type": m_type,
                        "severity": severity,
                        "in_danger_zone": True,
                        "estimated_distance_m": dist_m,
                        "configured_threshold_m": threshold_m if dist_m is not None else None,
                        "proximity_label": prox_label,
                        "proximity_method": proximity_method,
                        "worker_bbox": w_bbox,
                        "machine_bbox": m_bbox,
                        "worker_pos": w_pos,
                        "machine_pos": m_pos,
                        "timestamp": now.isoformat(),
                        "status": "ACTIVE",
                    }
                    alerts.append(alert_data)

                    # Stateful video tracking & anti-flapping logic
                    if is_video:
                        incident = self.active_incidents.get(pair_key)
                        if incident is None:
                            # Check cooldown before opening a new incident
                            last_resolved_time = self.cooldown_tracker.get(pair_key, 0)
                            if (time.time() - last_resolved_time) >= self.config.cooldown_seconds:
                                incident = SafetyIncident(
                                    incident_id=f"{w['raw_id']}_{m['raw_id']}_{frame_idx}",
                                    worker_id=w_id,
                                    machine_id=m_id,
                                    machine_type=m_type,
                                    configured_threshold_m=threshold_m,
                                    proximity_method=proximity_method,
                                    first_frame=frame_idx,
                                )
                                self.active_incidents[pair_key] = incident
                        if incident:
                            incident.last_frame = frame_idx
                            incident.last_detected_at = now
                            incident.consecutive_frames += 1
                            incident.last_estimated_distance_m = dist_m
                            incident.last_proximity_label = prox_label
                            incident.severity = severity
                            if dist_m is not None:
                                if incident.min_estimated_distance_m is None or dist_m < incident.min_estimated_distance_m:
                                    incident.min_estimated_distance_m = dist_m
                            # Activate once persistence frames threshold is satisfied
                            if incident.consecutive_frames >= self.config.persistence_frames:
                                incident.is_active = True

        # Video: Check for resolved incidents (pairs no longer in breach)
        resolved_this_frame = []
        if is_video:
            for pair_key, incident in list(self.active_incidents.items()):
                if pair_key not in current_breach_pairs:
                    # Worker left the danger zone -> Resolve incident
                    incident.is_active = False
                    incident.is_resolved = True
                    incident.resolved_at = now
                    self.resolved_incidents.append(incident)
                    self.cooldown_tracker[pair_key] = time.time()
                    resolved_this_frame.append(incident)
                    del self.active_incidents[pair_key]

        return {
            "alerts": alerts,
            "active_incidents": list(self.active_incidents.values()) if is_video else [],
            "resolved_incidents": resolved_this_frame,
            "pairs_evaluated": pairs_evaluated,
            "has_hazard": len(alerts) > 0,
        }

    def annotate_frame(
        self,
        frame: np.ndarray,
        alerts: List[Dict[str, Any]],
        all_detections: Optional[List[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        """
        Draw visual safety indicators on the media:
        - Highlights machine safety zone with warning outline and danger shading.
        - Highlights workers inside the danger zone in bold RED with warning banner.
        - Draws a danger connection line between machine and worker.
        - Shows machine ID and safety alert status.
        - Avoids displaying unsupported numeric distances.
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Draw machine safety zones first (as semi-transparent overlay)
        overlay = annotated.copy()

        # Track which machines and workers are in breach to style appropriately
        breached_machines = {a["machine_id"] for a in alerts}
        breached_workers = {a["worker_id"] for a in alerts}

        for alert in alerts:
            m_bbox = alert.get("machine_bbox")
            w_pos = alert.get("worker_pos")
            m_pos = alert.get("machine_pos")

            if m_bbox:
                zone = self.get_machine_danger_zone(m_bbox)
                cx, cy = zone["center"]
                rx, ry = zone["axes"]

                # Fill danger ellipse on overlay (Red/Amber)
                cv2.ellipse(overlay, (cx, cy), (rx, ry), 0, 0, 360, (0, 0, 220), -1)

            # Draw connection line between worker base and machine base
            if w_pos and m_pos:
                cv2.line(annotated, w_pos, m_pos, (0, 0, 255), 2, cv2.LINE_AA)

        # Blend danger zone overlay
        alpha = 0.22
        cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)

        # Redraw ellipse boundaries cleanly
        for alert in alerts:
            m_bbox = alert.get("machine_bbox")
            if m_bbox:
                zone = self.get_machine_danger_zone(m_bbox)
                cx, cy = zone["center"]
                rx, ry = zone["axes"]
                cv2.ellipse(annotated, (cx, cy), (rx, ry), 0, 0, 360, (0, 0, 255), 2, cv2.LINE_AA)

        # Highlight workers inside danger zone in bold RED
        for alert in alerts:
            w_bbox = alert.get("worker_bbox")
            m_bbox = alert.get("machine_bbox")
            w_id = alert.get("worker_id", "Worker")
            m_id = alert.get("machine_id", "Machine")
            dist_m = alert.get("estimated_distance_m")

            if w_bbox:
                wx1, wy1, wx2, wy2 = w_bbox
                # Heavy RED bounding box around breached worker
                cv2.rectangle(annotated, (wx1, wy1), (wx2, wy2), (0, 0, 255), 3)

                # Worker danger label
                w_label = f"DANGER! {w_id}"
                if dist_m is not None:
                    w_label += f" ({dist_m:.1f}m)"

                (tw, th), _ = cv2.getTextSize(w_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(annotated, (wx1, wy1 - th - 10), (wx1 + tw + 6, wy1), (0, 0, 220), -1)
                cv2.putText(annotated, w_label, (wx1 + 3, wy1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            if m_bbox:
                mx1, my1, mx2, my2 = m_bbox
                # Machine hazard label
                m_label = f"⚠️ {m_id} [PROXIMITY ALERT]"
                (mtw, mth), _ = cv2.getTextSize(m_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(annotated, (mx1, my1 - mth - 10), (mx1 + mtw + 6, my1), (0, 0, 200), -1)
                cv2.putText(annotated, m_label, (mx1 + 3, my1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # If there are active alerts, add a prominent red safety banner at top of frame
        if alerts:
            banner_h = 36
            banner_overlay = annotated.copy()
            cv2.rectangle(banner_overlay, (0, 0), (w, banner_h), (0, 0, 180), -1)
            cv2.addWeighted(banner_overlay, 0.85, annotated, 0.15, 0, annotated)

            banner_text = f"CRITICAL: {len(alerts)} MACHINE PROXIMITY HAZARD{'S' if len(alerts) > 1 else ''} DETECTED"
            cv2.putText(annotated, banner_text, (20, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        return annotated
