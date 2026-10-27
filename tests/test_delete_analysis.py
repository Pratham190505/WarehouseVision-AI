import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.processing_service import get_processing_service, RESULT_DIR
from app.services.auth_service import get_current_user


class TestDeleteAnalysisServiceAndAPI:
    """Test suite for analysis history deletion and user isolation."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.service = get_processing_service()
        self.client = TestClient(app)
        yield
        app.dependency_overrides.clear()

    def test_unauthenticated_delete_returns_401(self):
        resp = self.client.delete("/api/analysis/some-random-id")
        assert resp.status_code == 401

    def test_delete_analysis_success_and_cleanup(self):
        user_id = "test_user_delete_123"
        app.dependency_overrides[get_current_user] = lambda: {
            "id": user_id, "name": "Test User", "email": "del@test.com"
        }

        # Create dummy file
        dummy_dir = RESULT_DIR / "images"
        dummy_dir.mkdir(parents=True, exist_ok=True)
        dummy_res = dummy_dir / "dummy_test_delete.jpg"
        dummy_res.write_text("dummy")

        analysis_id = self.service.create_analysis(
            filename="delete_me.jpg",
            original_filename="delete_me.jpg",
            media_type="image",
            file_size=1024,
            media_path=str(dummy_res),
            user_id=user_id,
        )

        self.service._save_result(
            analysis_id=analysis_id,
            result={
                "object_counts": {"workers": 1, "forklifts": 0, "boxes": 2, "pallets": 0},
                "result_path": "/results/images/dummy_test_delete.jpg",
                "analytics": {"occupancy": 10, "space_utilization": 20, "restricted_zone_violations": 0},
            },
            media_type="image",
        )

        # Verify it exists in history
        history_before = self.service.get_history(user_id=user_id)
        assert any(h["analysis_id"] == analysis_id for h in history_before)

        # Call delete endpoint
        resp = self.client.delete(f"/api/analysis/{analysis_id}")
        assert resp.status_code == 200
        assert resp.json()["success"] is True

        # Verify it no longer exists
        history_after = self.service.get_history(user_id=user_id)
        assert not any(h["analysis_id"] == analysis_id for h in history_after)

        # Verify get_analysis returns None
        assert self.service.get_analysis(analysis_id, user_id=user_id) is None

        # Verify file cleanup
        assert not dummy_res.exists()

    def test_delete_analysis_user_isolation(self):
        user_a = "user_a_owner"
        user_b = "user_b_intruder"

        analysis_id = self.service.create_analysis(
            filename="user_a_file.jpg",
            original_filename="user_a_file.jpg",
            media_type="image",
            file_size=2048,
            media_path="/tmp/fake.jpg",
            user_id=user_a,
        )

        # User B attempts to delete User A's analysis
        app.dependency_overrides[get_current_user] = lambda: {
            "id": user_b, "name": "User B", "email": "userb@test.com"
        }

        resp = self.client.delete(f"/api/analysis/{analysis_id}")
        assert resp.status_code == 404

        # Analysis still exists for user A
        assert self.service.get_analysis(analysis_id, user_id=user_a) is not None

        # Clean up by user A
        assert self.service.delete_analysis(analysis_id, user_id=user_a) is True

    def test_delete_nonexistent_analysis_returns_404(self):
        app.dependency_overrides[get_current_user] = lambda: {
            "id": "user_123", "name": "User", "email": "u@test.com"
        }
        resp = self.client.delete("/api/analysis/nonexistent-uuid-1234")
        assert resp.status_code == 404
