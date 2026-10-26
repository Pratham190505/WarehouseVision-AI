"""
Restricted zone detection for warehouse environments.

Defines configurable polygon zones within the warehouse.
Detects when objects (especially workers or forklifts)
enter restricted areas and counts violations.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class RestrictedZone:
    """Represents a single restricted zone as a polygon."""

    def __init__(self, name: str, polygon: List[Tuple[int, int]], color: Tuple[int, int, int] = (0, 0, 255)):
        """
        Args:
            name: Human-readable zone name.
            polygon: List of (x, y) points defining the polygon.
            color: BGR color for visualization.
        """
        self.name = name
        self.polygon = np.array(polygon, dtype=np.int32)
        self.color = color

    def contains_point(self, point: Tuple[int, int]) -> bool:
        """Check if a point is inside this zone."""
        result = cv2.pointPolygonTest(self.polygon, (float(point[0]), float(point[1])), False)
        return result >= 0

    def draw(self, image: np.ndarray, alpha: float = 0.25) -> np.ndarray:
        """Draw the zone overlay on the image."""
        overlay = image.copy()
        cv2.fillPoly(overlay, [self.polygon], self.color)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        cv2.polylines(image, [self.polygon], True, self.color, 2)

        # Label
        centroid = self.polygon.mean(axis=0).astype(int)
        label = f"RESTRICTED: {self.name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(image, label, (centroid[0] - tw // 2, centroid[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return image


class ZoneDetector:
    """Manages restricted zones and detects violations."""

    def __init__(self, zones: Optional[List[Dict[str, Any]]] = None):
        """
        Args:
            zones: List of zone configs, each with 'name', 'polygon', and optional 'color'.
        """
        self.zones: List[RestrictedZone] = []

        if zones:
            for z in zones:
                self.add_zone(
                    name=z.get("name", "Restricted"),
                    polygon=z.get("polygon", []),
                    color=tuple(z.get("color", (0, 0, 255))),
                )

    def add_zone(self, name: str, polygon: List[Tuple[int, int]],
                 color: Tuple[int, int, int] = (0, 0, 255)):
        """Add a restricted zone."""
        if len(polygon) >= 3:
            self.zones.append(RestrictedZone(name, polygon, color))
            logger.info(f"Added restricted zone: {name}")

    def add_default_zones(self, image_width: int, image_height: int):
        """
        Add default demo restricted zones based on image dimensions.
        Creates a zone in the bottom-right quadrant.
        """
        margin_x = int(image_width * 0.6)
        margin_y = int(image_height * 0.6)

        self.add_zone(
            name="Loading Bay",
            polygon=[
                (margin_x, margin_y),
                (image_width - 20, margin_y),
                (image_width - 20, image_height - 20),
                (margin_x, image_height - 20),
            ],
            color=(0, 0, 200),
        )

    def check_violations(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check detections against restricted zones.

        Args:
            detections: List of detection dicts with 'center', 'bbox', 'class', etc.

        Returns:
            Dict with violation count and details.
        """
        violations = []
        total_violations = 0

        for detection in detections:
            bbox = detection.get("bbox")
            if bbox and len(bbox) == 4:
                ground_pt = ((bbox[0] + bbox[2]) // 2, bbox[3])
                center_pt = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            else:
                raw_center = detection.get("center", [0, 0])
                ground_pt = tuple(raw_center)
                center_pt = tuple(raw_center)

            from app.vision.safety_config import normalize_class
            raw_cls = detection.get("class", "unknown")
            obj_class = normalize_class(raw_cls)

            for zone in self.zones:
                if zone.contains_point(ground_pt) or zone.contains_point(center_pt):
                    violations.append({
                        "zone": zone.name,
                        "object_class": obj_class,
                        "raw_class": raw_cls,
                        "position": ground_pt,
                        "center": center_pt,
                        "confidence": detection.get("confidence", 0),
                    })
                    total_violations += 1

        return {
            "total_violations": total_violations,
            "violations": violations,
        }

    def draw_zones(self, image: np.ndarray) -> np.ndarray:
        """Draw all restricted zones on the image."""
        for zone in self.zones:
            image = zone.draw(image)
        return image

    def annotate_violations(self, image: np.ndarray, violations: List[Dict]) -> np.ndarray:
        """Draw violation markers on the image."""
        for v in violations:
            pos = v.get("position", (0, 0))
            # Draw warning circle
            cv2.circle(image, pos, 20, (0, 0, 255), 3)
            cv2.putText(image, "VIOLATION", (pos[0] - 40, pos[1] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return image

    @property
    def has_zones(self) -> bool:
        """Whether any zones are configured."""
        return len(self.zones) > 0
