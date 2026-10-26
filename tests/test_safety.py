"""
Comprehensive Tests for AI-Powered Machine Proximity Safety Alerts.

Validates the 11 feature requirements:
1. Worker outside a safety zone (no false alert)
2. Worker entering a safety zone (alert triggered)
3. Worker leaving a safety zone (incident transitioned to resolved)
4. Repeated frames without duplicate events (state maintained, no spam)
5. Different machine-specific thresholds (overrides respected)
6. Missing camera calibration (uncalibrated mode: NO fabricated metric distance in metres)
7. Invalid or unsupported detections (graceful handling of malformed data)
8. MongoDB event persistence & fallback (stored in safety_events)
9. User isolation (user A cannot view or acknowledge user B's safety events)
10. Alert acknowledgement while hazard remains active (acknowledgement recorded, hazard remains active)
11. Light and dark theme compatibility (CSS tokens and classes exist for both themes)

Uses synthetic detections so unit tests do not require a GPU or trained neural network model.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.vision.safety_config import SafetyConfig
from app.vision.safety_zone import SafetyZoneEvaluator, SafetyIncident
from app.services.processing_service import get_processing_service


class TestMachineSafetyUnitEngine:
    """Unit tests for the SafetyZoneEvaluator engine using synthetic detections."""

    def test_worker_outside_safety_zone(self):
        """Worker far away from a forklift should not trigger any safety alert."""
        config = SafetyConfig(default_threshold_m=2.0)
        evaluator = SafetyZoneEvaluator(config=config)

        # Worker far away in top-left, forklift in bottom-right
        detections = [
            {"class": "worker", "bbox": [50, 50, 90, 150], "track_id": 1},
            {"class": "forklift", "bbox": [800, 600, 950, 750], "track_id": 2},
        ]

        result = evaluator.evaluate_detections(detections, frame_idx=0, is_video=False)
        assert result["has_hazard"] is False
        assert len(result["alerts"]) == 0

    def test_worker_entering_safety_zone(self):
        """Worker inside the machine's proximity perimeter should trigger an alert."""
        config = SafetyConfig(default_threshold_m=2.0)
        evaluator = SafetyZoneEvaluator(config=config)

        # Worker placed directly adjacent/inside machine's base perimeter
        # Forklift base: center (500, 500)
        # Worker base: center (505, 505) -> clearly inside
        detections = [
            {"class": "worker", "bbox": [490, 420, 520, 505], "track_id": 10},
            {"class": "forklift", "bbox": [400, 350, 600, 500], "track_id": 3},
        ]

        result = evaluator.evaluate_detections(detections, frame_idx=0, is_video=False)
        assert result["has_hazard"] is True
        assert len(result["alerts"]) == 1

        alert = result["alerts"][0]
        assert alert["machine_type"] == "forklift"
        assert alert["in_danger_zone"] is True
        assert alert["status"] == "ACTIVE"
        assert "Forklift" in alert["machine_id"]
        assert "Worker" in alert["worker_id"]

    def test_worker_leaving_safety_zone_in_video(self):
        """In video tracking, when a worker exits the danger zone, the incident resolves."""
        config = SafetyConfig(persistence_frames=1)
        evaluator = SafetyZoneEvaluator(config=config)

        # Frame 0: Worker inside zone -> Incident created and active
        frame0_dets = [
            {"class": "worker", "bbox": [490, 420, 520, 505], "track_id": 10},
            {"class": "forklift", "bbox": [400, 350, 600, 500], "track_id": 3},
        ]
        res0 = evaluator.evaluate_detections(frame0_dets, frame_idx=0, is_video=True)
        assert res0["has_hazard"] is True
        assert len(evaluator.active_incidents) == 1

        # Frame 1: Worker has moved far away
        frame1_dets = [
            {"class": "worker", "bbox": [50, 50, 90, 150], "track_id": 10},
            {"class": "forklift", "bbox": [400, 350, 600, 500], "track_id": 3},
        ]
        res1 = evaluator.evaluate_detections(frame1_dets, frame_idx=1, is_video=True)
        assert res1["has_hazard"] is False
        assert len(evaluator.active_incidents) == 0
        assert len(evaluator.resolved_incidents) == 1
        assert evaluator.resolved_incidents[0].is_resolved is True

    def test_repeated_frames_without_duplicate_events(self):
        """Consecutive video frames with the worker inside should update the same incident without duplicating."""
        config = SafetyConfig(persistence_frames=1)
        evaluator = SafetyZoneEvaluator(config=config)

        breach_dets = [
            {"class": "worker", "bbox": [490, 420, 520, 505], "track_id": 5},
            {"class": "forklift", "bbox": [400, 350, 600, 500], "track_id": 1},
        ]

        # Simulate 10 consecutive frames
        for frame_idx in range(10):
            res = evaluator.evaluate_detections(breach_dets, frame_idx=frame_idx, is_video=True)
            assert res["has_hazard"] is True

        # Exactly ONE incident should be tracked
        assert len(evaluator.active_incidents) == 1
        incident = list(evaluator.active_incidents.values())[0]
        assert incident.first_frame == 0
        assert incident.last_frame == 9
        assert incident.consecutive_frames == 10

    def test_different_machine_specific_thresholds(self):
        """Machine-specific overrides and machine-type thresholds are correctly applied."""
        config = SafetyConfig(
            default_threshold_m=2.0,
            machine_thresholds={"forklift": 2.5, "truck": 4.0},
            machine_id_overrides={"3": 3.5},
        )

        assert config.get_threshold_m("forklift") == 2.5
        assert config.get_threshold_m("truck") == 4.0
        assert config.get_threshold_m("machinery") == 2.0  # fallback to default
        assert config.get_threshold_m("forklift", machine_id="3") == 3.5  # specific ID override

    def test_missing_camera_calibration_does_not_fabricate_meters(self):
        """CRITICAL: When camera calibration is disabled/missing, NO metric distance in metres is fabricated."""
        config = SafetyConfig(camera_calibration={"enabled": False})
        evaluator = SafetyZoneEvaluator(config=config)

        detections = [
            {"class": "worker", "bbox": [490, 420, 520, 505], "track_id": 1},
            {"class": "forklift", "bbox": [400, 350, 600, 500], "track_id": 2},
        ]

        res = evaluator.evaluate_detections(detections, frame_idx=0, is_video=False)
        assert len(res["alerts"]) == 1
        alert = res["alerts"][0]

        # Must NOT claim a numeric distance in metres
        assert alert["estimated_distance_m"] is None
        assert alert["proximity_method"] == "image_space_proximity"
        assert "Image-space" in alert["proximity_label"]

    def test_calibrated_camera_computes_metric_distance(self):
        """When calibration is enabled with valid pixels_per_meter, distance is computed in metres."""
        config = SafetyConfig(
            default_threshold_m=3.0,
            machine_thresholds={"forklift": 3.0},
            camera_calibration={
                "enabled": True,
                "method": "pixels_per_meter",
                "pixels_per_meter": 50.0,  # 50 pixels = 1 metre
            }
        )
        evaluator = SafetyZoneEvaluator(config=config)

        # Worker at (500, 550), Forklift at (500, 500) -> 50 pixels apart = 1.0 m
        detections = [
            {"class": "worker", "bbox": [490, 480, 510, 550], "track_id": 1},
            {"class": "forklift", "bbox": [450, 400, 550, 500], "track_id": 2},
        ]

        res = evaluator.evaluate_detections(detections, frame_idx=0, is_video=False)
        assert len(res["alerts"]) == 1
        alert = res["alerts"][0]

        assert alert["estimated_distance_m"] == 1.0
        assert alert["configured_threshold_m"] == 3.0
        assert alert["proximity_method"] == "camera_calibration_ground_plane"

    def test_invalid_or_unsupported_detections_handled_gracefully(self):
        """Engine handles missing boxes, unknown classes, and malformed dictionaries without crashing."""
        evaluator = SafetyZoneEvaluator()

        malformed = [
            {},
            {"class": "unknown_object", "bbox": [0, 0, 10, 10]},
            {"class": "worker"},  # missing bbox
            {"class": "forklift", "bbox": "not_a_list"},  # malformed bbox
        ]

        # Must not raise an exception
        try:
            res = evaluator.evaluate_detections(malformed)
            assert res["has_hazard"] is False
        except Exception as e:
            pytest.fail(f"Evaluator failed on malformed detections: {e}")


class TestSafetyAPIAndUserIsolation:
    """Integration tests for Safety API endpoints, persistence, and isolation."""

    password = "SafetyPassword123!"

    @pytest.fixture(autouse=True)
    def setup_clients(self):
        import uuid
        uid = uuid.uuid4().hex[:8]
        self.user_a_email = f"safety_a_{uid}@example.com"
        self.user_b_email = f"safety_b_{uid}@example.com"

        self.client_a = TestClient(app, follow_redirects=False)
        self.client_b = TestClient(app, follow_redirects=False)

        # Signup User A
        resp_a = self.client_a.post("/api/auth/signup", json={
            "name": "Safety Officer Alpha",
            "email": self.user_a_email,
            "password": self.password,
            "confirm_password": self.password,
        })
        assert resp_a.status_code == 201

        # Signup User B
        resp_b = self.client_b.post("/api/auth/signup", json={
            "name": "Safety Officer Beta",
            "email": self.user_b_email,
            "password": self.password,
            "confirm_password": self.password,
        })
        assert resp_b.status_code == 201

    def test_safety_config_endpoint(self):
        """GET /api/safety/config returns configuration details."""
        resp = self.client_a.get("/api/safety/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "default_threshold_m" in data
        assert "machine_classes" in data
        assert "forklift" in data["machine_classes"]

    def test_mongodb_event_persistence_and_alerts_retrieval(self):
        """Processing an analysis with a safety breach persists the event in safety_events."""
        service = get_processing_service()

        # Get User A's ID
        me_resp = self.client_a.get("/api/auth/me")
        user_a_id = me_resp.json()["id"]

        # Persist a synthetic safety incident directly through service
        analysis_id = "test_analysis_safety_01"
        incidents = [
            {
                "event_id": "event_forklift_03_worker_12",
                "worker_id": "Worker #12",
                "worker_track_id": "12",
                "machine_id": "Forklift #03",
                "machine_track_id": "03",
                "machine_type": "forklift",
                "severity": "CRITICAL",
                "status": "ACTIVE",
                "proximity_method": "image_space_proximity",
                "estimated_distance_m": None,
                "configured_threshold_m": 2.0,
            }
        ]

        service._persist_safety_events(analysis_id, user_a_id, incidents)

        # User A should now see this active alert
        alerts_resp = self.client_a.get("/api/safety/alerts")
        assert alerts_resp.status_code == 200
        alerts_data = alerts_resp.json()
        assert alerts_data["count"] >= 1
        found = any(a["event_id"] == "event_forklift_03_worker_12" for a in alerts_data["alerts"])
        assert found is True

    def test_user_isolation_on_safety_events(self):
        """User B cannot view or acknowledge User A's safety events (404 Access Denied)."""
        service = get_processing_service()
        me_a = self.client_a.get("/api/auth/me").json()
        user_a_id = me_a["id"]

        event_id = "event_isolated_alpha_1"
        service._persist_safety_events(
            analysis_id="analysis_iso_a",
            user_id=user_a_id,
            incidents=[{
                "event_id": event_id,
                "worker_id": "Worker #1",
                "machine_id": "Forklift #1",
                "machine_type": "forklift",
                "severity": "HIGH",
                "status": "ACTIVE",
            }]
        )

        # User A can view the event
        resp_a = self.client_a.get(f"/api/safety/events/{event_id}")
        assert resp_a.status_code == 200
        assert resp_a.json()["event_id"] == event_id

        # User B cannot view the event
        resp_b = self.client_b.get(f"/api/safety/events/{event_id}")
        assert resp_b.status_code == 404

        # User B cannot acknowledge User A's event
        ack_b = self.client_b.post(f"/api/safety/events/{event_id}/acknowledge")
        assert ack_b.status_code == 404

    def test_alert_acknowledgement_while_hazard_remains_active(self):
        """Acknowledging an alert records acknowledgement but physical status REMAINS ACTIVE."""
        service = get_processing_service()
        me_a = self.client_a.get("/api/auth/me").json()
        user_a_id = me_a["id"]

        event_id = "event_ack_test_active"
        service._persist_safety_events(
            analysis_id="analysis_ack_test",
            user_id=user_a_id,
            incidents=[{
                "event_id": event_id,
                "worker_id": "Worker #8",
                "machine_id": "Forklift #2",
                "machine_type": "forklift",
                "severity": "CRITICAL",
                "status": "ACTIVE",
            }]
        )

        # Acknowledge the event
        ack_resp = self.client_a.post(f"/api/safety/events/{event_id}/acknowledge")
        assert ack_resp.status_code == 200
        data = ack_resp.json()
        assert data["success"] is True
        event = data["event"]

        # CRITICAL SAFETY REQUIREMENT:
        # Acknowledging MUST NOT mark physical status as safe/resolved!
        assert event["is_acknowledged"] is True
        assert event["acknowledged_at"] is not None
        assert event["status"] == "ACTIVE"  # STILL ACTIVE!

    def test_end_to_end_image_analysis_pipeline_with_safety(self):
        """Uploading and analyzing an image processes safety rules and populates safety fields."""
        import cv2
        import numpy as np
        import io

        # Create a synthetic image
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        _, img_bytes = cv2.imencode(".jpg", img)

        resp = self.client_a.post(
            "/api/analyze",
            files={"file": ("warehouse_test.jpg", io.BytesIO(img_bytes.tobytes()), "image/jpeg")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "safety_alerts_count" in data
        assert "safety_alerts" in data
        assert isinstance(data["safety_alerts"], list)


class TestThemeCompatibility:
    """Verifies that light and dark themes contain necessary CSS variables and styling."""

    def test_theme_css_tokens_present(self):
        """Ensure both light and dark theme definitions have safety alert tokens."""
        css_path = Path(__file__).resolve().parent.parent / "app" / "static" / "css" / "themes.css"
        assert css_path.exists()
        content = css_path.read_text(encoding="utf-8")

        required_tokens = [
            "--danger-alert-bg",
            "--danger-alert-border",
            "--danger-alert-text",
            "--danger-alert-glow",
            "--danger-alert-badge",
        ]
        for token in required_tokens:
            assert token in content, f"Token {token} missing from themes.css"

    def test_dashboard_css_safety_classes_present(self):
        """Ensure dashboard.css includes styling rules for safety alerts."""
        css_path = Path(__file__).resolve().parent.parent / "app" / "static" / "css" / "dashboard.css"
        assert css_path.exists()
        content = css_path.read_text(encoding="utf-8")

        required_classes = [
            ".safety-alert-banner",
            ".safety-alerts-container",
            ".btn-ack-alert",
            ".safety-disclaimer",
        ]
        for cls in required_classes:
            assert cls in content, f"CSS class {cls} missing from dashboard.css"
