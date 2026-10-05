"""
YOLO-based object detector for warehouse environments.

Uses Ultralytics YOLO for detecting warehouse objects:
- Workers (person)
- Forklifts (truck/vehicle)
- Boxes (suitcase/handbag as proxy)
- Pallets (bench/dining table as proxy)

NOTE: The default YOLO model uses COCO classes. For production use,
a custom-trained warehouse model should be specified via MODEL_PATH
in the environment configuration. The class mappings below provide
reasonable defaults using COCO classes as proxies.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# COCO class ID → warehouse label mapping
# These map standard COCO classes to warehouse-relevant categories.
# Replace with custom model class IDs for production use.
DEFAULT_CLASS_MAPPING = {
    0: "worker",       # COCO: person
    # Forklifts: use truck/car as proxy
    7: "forklift",     # COCO: truck
    # Boxes: use suitcase/backpack as proxy
    24: "box",         # COCO: backpack
    28: "box",         # COCO: suitcase
    # Pallets: use bench/dining table as proxy
    13: "pallet",      # COCO: bench
    60: "pallet",      # COCO: dining table
}

# Warehouse label → display info
WAREHOUSE_LABELS = {
    "worker": {"color": (46, 204, 113), "icon": "👷"},
    "forklift": {"color": (52, 152, 219), "icon": "🏗️"},
    "box": {"color": (230, 126, 34), "icon": "📦"},
    "pallet": {"color": (155, 89, 182), "icon": "🎨"},
}

# Confidence threshold
DEFAULT_CONFIDENCE = 0.35


class WarehouseDetector:
    """YOLO-based warehouse object detector."""

    def __init__(self, model_path: Optional[str] = None, confidence: float = DEFAULT_CONFIDENCE):
        self.model = None
        self.model_loaded = False
        self.confidence = confidence
        self.class_mapping = DEFAULT_CLASS_MAPPING.copy()

        model_path = model_path or os.getenv("MODEL_PATH", "models/warehouse_model.pt")

        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load the YOLO model."""
        try:
            from ultralytics import YOLO

            path = Path(model_path)
            if path.exists():
                logger.info(f"Loading custom model from: {model_path}")
                self.model = YOLO(str(path))
            else:
                # Fall back to pretrained YOLOv8n
                logger.info("Custom model not found. Using pretrained YOLOv8n.")
                self.model = YOLO("yolov8n.pt")

            self.model_loaded = True
            logger.info("YOLO model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
            self.model_loaded = False

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run detection on a single image.

        Args:
            image: BGR image as numpy array.

        Returns:
            Dict with detections, counts, and annotated image.
        """
        if not self.model_loaded or self.model is None:
            return {
                "detections": [],
                "counts": {"workers": 0, "forklifts": 0, "boxes": 0, "pallets": 0},
                "annotated_image": image,
                "raw_detection_count": 0,
            }

        results = self.model(image, conf=self.confidence, verbose=False)

        detections = []
        counts = {"workers": 0, "forklifts": 0, "boxes": 0, "pallets": 0}
        annotated = image.copy()

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Map COCO class to warehouse class
                warehouse_class = self.class_mapping.get(cls_id)
                if warehouse_class is None:
                    continue

                # Count
                count_key = warehouse_class + "s" if warehouse_class != "box" else "boxes"
                if count_key == "boxs":
                    count_key = "boxes"
                counts[count_key] = counts.get(count_key, 0) + 1

                detection = {
                    "class": warehouse_class,
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2],
                    "center": [(x1 + x2) // 2, (y1 + y2) // 2],
                }
                detections.append(detection)

                # Draw annotation
                info = WAREHOUSE_LABELS.get(warehouse_class, {"color": (200, 200, 200)})
                color = info["color"]

                # Bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Label background
                label = f"{warehouse_class} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return {
            "detections": detections,
            "counts": counts,
            "annotated_image": annotated,
            "raw_detection_count": len(detections),
        }

    def detect_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict], Dict]:
        """
        Detect objects in a video frame. Returns annotated frame, detections, and counts.
        """
        result = self.detect(frame)
        return result["annotated_image"], result["detections"], result["counts"]

    def is_ready(self) -> bool:
        """Check if the detector is ready."""
        return self.model_loaded and self.model is not None
