"""
Occupancy and space utilization estimation for warehouse environments.

IMPORTANT LIMITATION:
A single 2D image cannot provide accurate physical square-meter
measurements without camera calibration, known reference dimensions,
or depth information. All metrics here are APPROXIMATE and based on
the proportion of the frame covered by detected objects.

These metrics should be treated as relative indicators, not absolute
physical measurements.
"""

import logging
from typing import Dict, Any, List

import numpy as np

logger = logging.getLogger(__name__)


class OccupancyAnalyzer:
    """
    Estimates warehouse occupancy and space utilization from detections.

    Occupancy is computed as the approximate percentage of the frame area
    covered by detected objects. Space utilization is derived from the
    ratio of occupied area to a configurable warehouse region.
    """

    def __init__(self, warehouse_region: Dict[str, int] = None):
        """
        Args:
            warehouse_region: Optional dict with 'x', 'y', 'width', 'height'
                             defining the warehouse area within the frame.
                             If None, the entire frame is considered.
        """
        self.warehouse_region = warehouse_region

    def analyze(self, detections: List[Dict[str, Any]],
                image_dims: tuple = None) -> Dict[str, Any]:
        """
        Estimate occupancy and space utilization.

        Args:
            detections: List of detection dicts with 'bbox' [x1, y1, x2, y2].
            image_dims: (height, width) of the image.

        Returns:
            Dict with occupancy and utilization metrics.
        """
        if not image_dims or len(detections) == 0:
            return {
                "occupancy_percent": 0.0,
                "space_utilization_percent": 0.0,
                "occupied_area_pixels": 0,
                "total_area_pixels": 0,
                "note": "Approximate estimate based on 2D bounding box coverage.",
            }

        h, w = image_dims[:2]

        # Define warehouse region
        if self.warehouse_region:
            r = self.warehouse_region
            total_area = r.get("width", w) * r.get("height", h)
        else:
            total_area = w * h

        # Calculate occupied area using bounding boxes
        # Create a mask to avoid double-counting overlapping boxes
        mask = np.zeros((h, w), dtype=np.uint8)

        for det in detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            mask[y1:y2, x1:x2] = 1

        occupied_pixels = int(np.sum(mask))
        occupancy = (occupied_pixels / total_area) * 100 if total_area > 0 else 0.0
        occupancy = min(occupancy, 100.0)

        # Space utilization: consider only storage-related objects
        storage_mask = np.zeros((h, w), dtype=np.uint8)
        for det in detections:
            if det.get("class") in ("box", "pallet"):
                bbox = det.get("bbox", [0, 0, 0, 0])
                x1, y1, x2, y2 = bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                storage_mask[y1:y2, x1:x2] = 1

        storage_pixels = int(np.sum(storage_mask))
        space_util = (storage_pixels / total_area) * 100 if total_area > 0 else 0.0
        space_util = min(space_util, 100.0)

        return {
            "occupancy_percent": round(occupancy, 1),
            "space_utilization_percent": round(space_util, 1),
            "occupied_area_pixels": occupied_pixels,
            "total_area_pixels": total_area,
            "note": "Approximate estimate based on 2D bounding box coverage.",
        }
