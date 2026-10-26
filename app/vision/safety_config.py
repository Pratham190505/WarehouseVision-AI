"""
Configuration settings for Machine Proximity Safety Alerts.

Defines safety thresholds, supported machine classes, zone geometries,
persistence and cooldown settings, and camera calibration parameters.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default machine proximity threshold in metres (prototype setting, not certified)
DEFAULT_PROXIMITY_THRESHOLD_M = 2.0

# Supported machine categories that can be treated as hazardous equipment
DEFAULT_MACHINE_CLASSES = ["forklift", "truck", "heavy_equipment", "machinery"]

# Supported worker categories
DEFAULT_WORKER_CLASSES = ["worker", "person"]

# Per-machine type threshold overrides (in metres)
DEFAULT_MACHINE_TYPE_THRESHOLDS: Dict[str, float] = {
    "forklift": 2.0,
    "truck": 2.5,
    "heavy_equipment": 3.0,
    "machinery": 2.0,
}

# Image-space proximity ratio when uncalibrated (relative to machine bounding box base)
# Defines a safety margin ellipse around the machine's base contact point
DEFAULT_IMAGE_SPACE_EXPANSION_RATIO = 1.35

# Persistence settings to reduce false alarms in video:
# Worker must be in zone for at least this many consecutive frames to trigger an alert
DEFAULT_PERSISTENCE_FRAMES = 2

# Cooldown settings:
# Seconds after an incident is resolved before a new alert can be created for the same pair
DEFAULT_COOLDOWN_SECONDS = 3.0


class SafetyConfig:
    """
    Configuration container for machine proximity safety monitoring.
    Can be instantiated from a dictionary, JSON file, or environment variables.
    """

    def __init__(
        self,
        default_threshold_m: float = DEFAULT_PROXIMITY_THRESHOLD_M,
        machine_thresholds: Optional[Dict[str, float]] = None,
        machine_id_overrides: Optional[Dict[str, float]] = None,
        machine_classes: Optional[List[str]] = None,
        worker_classes: Optional[List[str]] = None,
        image_space_expansion_ratio: float = DEFAULT_IMAGE_SPACE_EXPANSION_RATIO,
        persistence_frames: int = DEFAULT_PERSISTENCE_FRAMES,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        danger_zone_polygons: Optional[List[Dict[str, Any]]] = None,
        camera_calibration: Optional[Dict[str, Any]] = None,
    ):
        self.default_threshold_m = float(default_threshold_m)
        self.machine_thresholds = machine_thresholds or DEFAULT_MACHINE_TYPE_THRESHOLDS.copy()
        self.machine_id_overrides = machine_id_overrides or {}
        self.machine_classes = machine_classes or DEFAULT_MACHINE_CLASSES.copy()
        self.worker_classes = worker_classes or DEFAULT_WORKER_CLASSES.copy()
        self.image_space_expansion_ratio = float(image_space_expansion_ratio)
        self.persistence_frames = int(persistence_frames)
        self.cooldown_seconds = float(cooldown_seconds)
        self.danger_zone_polygons = danger_zone_polygons or []
        self.camera_calibration = camera_calibration or {
            "enabled": False,
            "method": "none",  # "homography" | "pixels_per_meter" | "none"
            "homography_matrix": None,
            "pixels_per_meter": None,
            "description": "Camera calibration is not active. Using image-space safety zone.",
        }

    def get_threshold_m(self, machine_type: str, machine_id: Optional[str] = None) -> float:
        """
        Get the configured threshold in metres for a specific machine.
        Evaluates machine_id overrides first, then machine_type, then default.
        """
        if machine_id and str(machine_id) in self.machine_id_overrides:
            return self.machine_id_overrides[str(machine_id)]
        return self.machine_thresholds.get(machine_type.lower(), self.default_threshold_m)

    def is_machine(self, class_name: str) -> bool:
        """Check if detected class is a hazardous machine."""
        return class_name.lower() in [c.lower() for c in self.machine_classes]

    def is_worker(self, class_name: str) -> bool:
        """Check if detected class is a worker."""
        return class_name.lower() in [c.lower() for c in self.worker_classes]

    def to_dict(self) -> Dict[str, Any]:
        """Export config to serializable dictionary."""
        return {
            "default_threshold_m": self.default_threshold_m,
            "machine_thresholds": self.machine_thresholds,
            "machine_id_overrides": self.machine_id_overrides,
            "machine_classes": self.machine_classes,
            "worker_classes": self.worker_classes,
            "image_space_expansion_ratio": self.image_space_expansion_ratio,
            "persistence_frames": self.persistence_frames,
            "cooldown_seconds": self.cooldown_seconds,
            "danger_zone_polygons": self.danger_zone_polygons,
            "camera_calibration": self.camera_calibration,
        }

    @classmethod
    def load_from_file(cls, path: str) -> "SafetyConfig":
        """Load configuration from a JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(**data)
        except Exception as e:
            logger.warning(f"Could not load safety config from {path}: {e}. Using defaults.")
            return cls()

    @classmethod
    def default(cls) -> "SafetyConfig":
        """Get default safety configuration."""
        return cls()


# Default global instance
_global_config: Optional[SafetyConfig] = None


def get_safety_config() -> SafetyConfig:
    """Get the active safety configuration."""
    global _global_config
    if _global_config is None:
        config_path = os.getenv("SAFETY_CONFIG_PATH")
        if config_path and Path(config_path).exists():
            _global_config = SafetyConfig.load_from_file(config_path)
        else:
            _global_config = SafetyConfig.default()
    return _global_config


def set_safety_config(config: SafetyConfig):
    """Set the active safety configuration (useful for testing and customization)."""
    global _global_config
    _global_config = config
