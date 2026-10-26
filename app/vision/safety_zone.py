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

from app.vision.safety_config import SafetyConfig, get_safety_config, normalize_class

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


class SafetyIncident:
    """
    Tracks a single machine-worker proximity or restricted zone incident across time/frames.
    Prevents alert flapping and multiple DB events for continuous events.
    """

    def __init__(
        self,
        incident_id: str,
        worker_id: str,
        machine_id: Optional[str] = None,
        machine_type: Optional[str] = None,
        configured_threshold_m: Optional[float] = None,
        proximity_method: str = "image_space",
        first_frame: int = 0,
        event_type: str = "machine_proximity",
        zone_name: Optional[str] = None,
    ):
        self.incident_id = incident_id
        self.worker_id = worker_id
        self.machine_id = machine_id
        self.machine_type = machine_type
        self.configured_threshold_m = configured_threshold_m
        self.proximity_method = proximity_method
        self.event_type = event_type
        self.zone_name = zone_name

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

        # 2. Uncalibrated image-space elliptical proximity zone around machine base
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
        zones: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate all workers against restricted warehouse polygon zones
        AND hazardous machine proximity perimeters.

        Args:
            detections: List of detection dicts with 'bbox', 'class', and optional 'track_id'.
            frame_idx: Current frame index (for videos).
            is_video: Whether this is part of continuous video analysis.
            image_shape: (height, width) of the image.
            zones: Optional list of restricted zones (e.g. from ZoneDetector).

        Returns:
            Dict containing:
            - alerts: List of active danger alerts for this frame
            - active_incidents: List of currently active SafetyIncident objects
            - resolved_incidents: Incidents resolved in this frame
            - pairs_evaluated: Total pairs inspected
            - has_hazard: Boolean indicating whether any hazard is active
        """
        workers = []
        machines = []

        for i, det in enumerate(detections):
            if not isinstance(det, dict):
                continue
            bbox = det.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue

            raw_cls = str(det.get("class", "")).lower()
            norm_cls = normalize_class(raw_cls)
            track_id = det.get("track_id")
            pos = self.get_ground_contact_point(bbox)
            center = det.get("center") or [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]

            # Determine human-readable identifier & classify
            if self.config.is_worker(norm_cls) or self.config.is_worker(raw_cls):
                wid = f"Worker #{track_id}" if track_id is not None else f"Worker #{i + 1}"
                workers.append({
                    "det": det,
                    "id": wid,
                    "raw_id": str(track_id) if track_id is not None else str(i + 1),
                    "pos": pos,
                    "center": center,
                    "bbox": bbox,
                    "class": norm_cls,
                })
            elif self.config.is_machine(norm_cls) or self.config.is_machine(raw_cls):
                m_label = norm_cls.capitalize() if norm_cls != "unknown" else "Machine"
                mid = f"{m_label} #{track_id}" if track_id is not None else f"{m_label} #{i + 1}"
                machines.append({
                    "det": det,
                    "id": mid,
                    "raw_id": str(track_id) if track_id is not None else str(i + 1),
                    "type": norm_cls,
                    "pos": pos,
                    "center": center,
                    "bbox": bbox,
                    "class": norm_cls,
                })

        current_breach_pairs: Set[Tuple[str, str]] = set()
        alerts: List[Dict[str, Any]] = []
        pairs_evaluated = 0
        now = _utc_now()

        # =====================================================================
        # 1. EVALUATE CONFIGURED RESTRICTED POLYGON ZONES (e.g. Loading Bay)
        # =====================================================================
        candidate_zones: List[Any] = []
        if zones:
            candidate_zones.extend(zones)
        if self.config.danger_zone_polygons:
            candidate_zones.extend(self.config.danger_zone_polygons)

        for zone_item in candidate_zones:
            if hasattr(zone_item, "polygon"):
                poly = zone_item.polygon
                zone_name = getattr(zone_item, "name", "Restricted Area")
            elif isinstance(zone_item, dict):
                poly = np.array(zone_item.get("polygon", []), dtype=np.int32)
                zone_name = zone_item.get("name", "Restricted Area")
            else:
                continue

            if len(poly) < 3:
                continue

            # Identify any machine operating inside or intersecting this zone
            associated_machine = None
            for m in machines:
                m_pt = (float(m["pos"][0]), float(m["pos"][1]))
                m_cnt = (float(m["center"][0]), float(m["center"][1]))
                if cv2.pointPolygonTest(poly, m_pt, False) >= 0 or cv2.pointPolygonTest(poly, m_cnt, False) >= 0:
                    associated_machine = m
                    break

            for w in workers:
                w_pt = (float(w["pos"][0]), float(w["pos"][1]))
                w_cnt = (float(w["center"][0]), float(w["center"][1]))

                # Check if worker's ground contact or center point is inside restricted polygon
                in_zone = (cv2.pointPolygonTest(poly, w_pt, False) >= 0 or
                           cv2.pointPolygonTest(poly, w_cnt, False) >= 0)

                if in_zone:
                    pair_key = (w["id"], f"zone_{zone_name}")
                    current_breach_pairs.add(pair_key)

                    m_id_val = associated_machine["id"] if associated_machine else (machines[0]["id"] if machines else None)
                    m_track_val = associated_machine["raw_id"] if associated_machine else (machines[0]["raw_id"] if machines else None)
                    m_type_val = associated_machine["type"] if associated_machine else (machines[0]["type"] if machines else "forklift")

                    alert_data = {
                        "violation_detected": True,
                        "event_type": "restricted_zone_violation",
                        "worker_id": w["id"],
                        "worker_track_id": w["raw_id"],
                        "machine_id": m_id_val,
                        "machine_track_id": m_track_val,
                        "machine_type": m_type_val,
                        "zone_name": zone_name,
                        "severity": "HIGH",
                        "in_danger_zone": True,
                        "estimated_distance_m": None,  # NEVER fabricate metric distance without calibration
                        "configured_threshold_m": None,
                        "proximity_label": f"Worker inside restricted area: {zone_name}",
                        "proximity_method": "image_space",
                        "worker_bbox": w["bbox"],
                        "machine_bbox": associated_machine["bbox"] if associated_machine else (machines[0]["bbox"] if machines else None),
                        "worker_pos": w["pos"],
                        "machine_pos": associated_machine["pos"] if associated_machine else (machines[0]["pos"] if machines else None),
                        "timestamp": now.isoformat(),
                        "status": "ACTIVE",
                    }
                    alerts.append(alert_data)

                    # Stateful video tracking
                    if is_video:
                        incident = self.active_incidents.get(pair_key)
                        if incident is None:
                            last_resolved_time = self.cooldown_tracker.get(pair_key, 0)
                            if (time.time() - last_resolved_time) >= self.config.cooldown_seconds:
                                incident = SafetyIncident(
                                    incident_id=f"{w['raw_id']}_{zone_name.lower().replace(' ', '_')}_{frame_idx}",
                                    worker_id=w["id"],
                                    machine_id=m_id_val,
                                    machine_type=m_type_val,
                                    configured_threshold_m=None,
                                    proximity_method="image_space",
                                    first_frame=frame_idx,
                                    event_type="restricted_zone_violation",
                                    zone_name=zone_name,
                                )
                                self.active_incidents[pair_key] = incident
                        if incident:
                            incident.last_frame = frame_idx
                            incident.last_detected_at = now
                            incident.consecutive_frames += 1
                            incident.last_proximity_label = f"Worker inside restricted area: {zone_name}"
                            incident.severity = "HIGH"
                            if incident.consecutive_frames >= self.config.persistence_frames:
                                incident.is_active = True

        # =====================================================================
        # 2. EVALUATE MACHINE-WORKER PROXIMITY PAIRS
        # =====================================================================
        for w in workers:
            w_pos = w["pos"]
            w_id = w["id"]

            for m in machines:
                pairs_evaluated += 1
                m_pos = m["pos"]
                m_id = m["id"]
                m_type = m["type"]
                pair_key = (w_id, m_id)

                threshold_m = self.config.get_threshold_m(m_type, str(m["raw_id"]))
                dist_m = self.calculate_distance_meters(w_pos, m_pos)
                proximity_method = (
                    "camera_calibration_ground_plane"
                    if dist_m is not None
                    else "image_space_proximity"
                )

                in_danger_zone, prox_label = self.is_point_in_machine_zone(
                    point=w_pos,
                    machine_bbox=m["bbox"],
                    calibrated_distance_m=dist_m,
                    threshold_m=threshold_m,
                )

                if in_danger_zone:
                    current_breach_pairs.add(pair_key)

                    if dist_m is not None:
                        severity = "CRITICAL" if dist_m <= (threshold_m * 0.5) else "HIGH"
                    else:
                        severity = "HIGH"

                    alert_data = {
                        "violation_detected": True,
                        "event_type": "machine_proximity",
                        "worker_id": w_id,
                        "worker_track_id": w["raw_id"],
                        "machine_id": m_id,
                        "machine_track_id": m["raw_id"],
                        "machine_type": m_type,
                        "zone_name": None,
                        "severity": severity,
                        "in_danger_zone": True,
                        "estimated_distance_m": dist_m,
                        "configured_threshold_m": threshold_m if dist_m is not None else None,
                        "proximity_label": prox_label,
                        "proximity_method": proximity_method,
                        "worker_bbox": w["bbox"],
                        "machine_bbox": m["bbox"],
                        "worker_pos": w_pos,
                        "machine_pos": m_pos,
                        "timestamp": now.isoformat(),
                        "status": "ACTIVE",
                    }
                    alerts.append(alert_data)

                    # Stateful video tracking
                    if is_video:
                        incident = self.active_incidents.get(pair_key)
                        if incident is None:
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
                                    event_type="machine_proximity",
                                    zone_name=None,
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
                            if incident.consecutive_frames >= self.config.persistence_frames:
                                incident.is_active = True

        # Video: Check for resolved incidents (pairs no longer in breach)
        resolved_this_frame = []
        if is_video:
            for pair_key, incident in list(self.active_incidents.items()):
                if pair_key not in current_breach_pairs:
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
        - Highlights workers inside restricted zones or machine danger zones in bold RED with warning banner.
        - Draws a danger connection line between machine and worker.
        - Shows machine/zone ID and safety alert status.
        - Avoids displaying unsupported numeric distances.
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        if not alerts:
            return annotated

        # Draw machine safety zones first (as semi-transparent overlay)
        overlay = annotated.copy()

        for alert in alerts:
            m_bbox = alert.get("machine_bbox")
            w_pos = alert.get("worker_pos")
            m_pos = alert.get("machine_pos")

            if m_bbox:
                zone = self.get_machine_danger_zone(m_bbox)
                cx, cy = zone["center"]
                rx, ry = zone["axes"]
                cv2.ellipse(overlay, (cx, cy), (rx, ry), 0, 0, 360, (0, 0, 220), -1)

            # Draw connection line between worker base and machine base
            if w_pos and m_pos:
                cv2.line(annotated, tuple(w_pos), tuple(m_pos), (0, 0, 255), 2, cv2.LINE_AA)

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

        # Highlight workers inside danger / restricted zone in bold RED
        for alert in alerts:
            w_bbox = alert.get("worker_bbox")
            m_bbox = alert.get("machine_bbox")
            w_id = alert.get("worker_id", "Worker")
            m_id = alert.get("machine_id", "Machine")
            dist_m = alert.get("estimated_distance_m")
            event_type = alert.get("event_type", "machine_proximity")
            zone_name = alert.get("zone_name")

            if w_bbox:
                wx1, wy1, wx2, wy2 = w_bbox
                # Heavy RED bounding box around breached worker
                cv2.rectangle(annotated, (wx1, wy1), (wx2, wy2), (0, 0, 255), 3)

                # Worker danger label
                if event_type == "restricted_zone_violation":
                    w_label = f"RESTRICTED ZONE: {zone_name or 'VIOLATION'}"
                else:
                    w_label = f"DANGER! {w_id}"
                    if dist_m is not None:
                        w_label += f" ({dist_m:.1f}m)"

                (tw, th), _ = cv2.getTextSize(w_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(annotated, (wx1, wy1 - th - 10), (wx1 + tw + 6, wy1), (0, 0, 220), -1)
                cv2.putText(annotated, w_label, (wx1 + 3, wy1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            if m_bbox and event_type == "machine_proximity":
                mx1, my1, mx2, my2 = m_bbox
                m_label = f"⚠️ {m_id} [PROXIMITY ALERT]"
                (mtw, mth), _ = cv2.getTextSize(m_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(annotated, (mx1, my1 - mth - 10), (mx1 + mtw + 6, my1), (0, 0, 200), -1)
                cv2.putText(annotated, m_label, (mx1 + 3, my1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Prominent red safety banner at top of frame
        banner_h = 36
        banner_overlay = annotated.copy()
        cv2.rectangle(banner_overlay, (0, 0), (w, banner_h), (0, 0, 180), -1)
        cv2.addWeighted(banner_overlay, 0.85, annotated, 0.15, 0, annotated)

        banner_text = f"CRITICAL: {len(alerts)} SAFETY HAZARD{'S' if len(alerts) > 1 else ''} DETECTED"
        cv2.putText(annotated, banner_text, (20, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        return annotated
