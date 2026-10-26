"""
Image processing service for WarehouseVision AI.

Handles the complete pipeline for image analysis:
Upload → Detection → Zone Check → Occupancy → Inventory → Analytics → Save
"""

import os
import time
import uuid
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import cv2
import numpy as np

from app.vision.detector import WarehouseDetector
from app.vision.zone_detection import ZoneDetector
from app.vision.occupancy import OccupancyAnalyzer
from app.vision.inventory import InventoryAnalyzer
from app.vision.analytics import WarehouseAnalytics
from app.vision.safety_zone import SafetyZoneEvaluator
from app.database.models import AnalysisRecord, ObjectCounts, AnalyticsData

logger = logging.getLogger(__name__)


class ImageService:
    """Service for processing warehouse images."""

    def __init__(self):
        self.detector = WarehouseDetector()
        self.zone_detector = ZoneDetector()
        self.occupancy_analyzer = OccupancyAnalyzer()
        self.inventory_analyzer = InventoryAnalyzer()
        self.analytics_engine = WarehouseAnalytics()
        self.safety_evaluator = SafetyZoneEvaluator()

    def process(self, image_path: str, result_dir: str,
                analysis_id: str = None) -> Dict[str, Any]:
        """
        Process a warehouse image through the full analysis pipeline.

        Args:
            image_path: Path to the uploaded image.
            result_dir: Directory to save annotated results.
            analysis_id: Optional pre-assigned analysis ID.

        Returns:
            Complete analysis result dict.
        """
        start_time = time.time()
        analysis_id = analysis_id or str(uuid.uuid4())

        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

            h, w = image.shape[:2]

            # 1. Object Detection
            detection_result = self.detector.detect(image)
            detections = detection_result["detections"]
            counts = detection_result["counts"]
            annotated = detection_result["annotated_image"]

            # 2. Restricted Zone Detection
            if not self.zone_detector.has_zones:
                self.zone_detector.add_default_zones(w, h)

            zone_result = self.zone_detector.check_violations(detections)
            annotated = self.zone_detector.draw_zones(annotated)
            if zone_result["violations"]:
                annotated = self.zone_detector.annotate_violations(
                    annotated, zone_result["violations"]
                )

            # 3. Occupancy Analysis
            occupancy_result = self.occupancy_analyzer.analyze(detections, (h, w))

            # 4. Inventory Analysis
            inventory_result = self.inventory_analyzer.analyze(detections, (h, w))

            # 5. Machine Proximity Safety Zone & Restricted Zone Evaluation
            safety_result = self.safety_evaluator.evaluate_detections(
                detections=detections,
                frame_idx=0,
                is_video=False,
                image_shape=(h, w),
                zones=self.zone_detector.zones,
            )
            safety_alerts = safety_result.get("alerts", [])
            annotated = self.safety_evaluator.annotate_frame(annotated, safety_alerts, detections)

            # 6. Compute Analytics
            analytics = self.analytics_engine.compute(
                object_counts=counts,
                occupancy_data=occupancy_result,
                zone_violations=zone_result,
                inventory_data=inventory_result,
            )

            # 7. Save annotated image
            result_filename = f"{analysis_id}_annotated.jpg"
            result_path = os.path.join(result_dir, result_filename)
            cv2.imwrite(result_path, annotated)

            processing_time = round(time.time() - start_time, 2)

            return {
                "analysis_id": analysis_id,
                "status": "completed",
                "processing_time": processing_time,
                "object_counts": counts,
                "analytics": analytics,
                "safety_alerts": safety_alerts,
                "safety_alerts_count": len(safety_alerts),
                "safety_incidents": safety_alerts,
                "result_path": f"/results/images/{result_filename}",
                "detections": detections,
                "zone_violations": zone_result,
                "occupancy": occupancy_result,
                "inventory": inventory_result,
            }

        except Exception as e:
            processing_time = round(time.time() - start_time, 2)
            logger.error(f"Image processing failed: {e}", exc_info=True)
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
                    "insights": [f"Analysis failed: {str(e)}"],
                },
            }
