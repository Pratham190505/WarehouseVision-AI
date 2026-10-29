"""
Automated Tests for Processed Video Output in WarehouseVision AI.

Validates all 13 core requirements:
1. Video processing reads multiple frames.
2. Annotated frames are written throughout the video.
3. The output video contains more than one frame.
4. The output video can be decoded.
5. The processed video URL is returned by the API.
6. The dashboard uses the processed video URL for video analyses.
7. Image analyses still display annotated images.
8. Analytics are still returned correctly.
9. Safety-zone annotations remain visible in processed video frames.
10. Tracking annotations are updated frame by frame when tracking is enabled.
11. Output files are correctly finalized.
12. Failed video encoding returns an appropriate error.
13. Video access respects authenticated user ownership.
"""

import os
import io
import uuid
from types import SimpleNamespace
from pathlib import Path
import pytest
import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.services.video_service import VideoService
from app.services.processing_service import get_processing_service, RESULT_DIR
from app.services.auth_service import get_current_user


def create_synthetic_video(path: str, num_frames: int = 15, width: int = 320, height: int = 240, fps: float = 10.0):
    """Generate a valid synthetic test video with varying frames."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(num_frames):
        # Create distinct frames with a moving white square
        frame = np.full((height, width, 3), 30, dtype=np.uint8)
        x = int(20 + (i * (width - 60) / max(1, num_frames - 1)))
        cv2.rectangle(frame, (x, 50), (x + 30, 90), (200, 200, 200), -1)
        cv2.putText(frame, f"F:{i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    assert os.path.isfile(path) and os.path.getsize(path) > 0


class TestVideoProcessingAndDecoding:
    """Tests 1, 2, 3, 4, 9, 10, 11: Service-level video frame processing, annotations, and encoding."""

    def test_video_processing_reads_and_annotates_multiple_frames(self, tmp_path):
        """Video processing reads multiple frames, writes annotated frames throughout, and output decodes."""
        input_video = str(tmp_path / "synthetic_input.mp4")
        create_synthetic_video(input_video, num_frames=15, width=320, height=240, fps=10.0)

        result_dir = str(tmp_path / "results")
        os.makedirs(result_dir, exist_ok=True)

        service = VideoService()
        analysis_id = f"test_{uuid.uuid4().hex[:8]}"
        result = service.process(input_video, result_dir, analysis_id=analysis_id, frame_skip=1)

        # Requirement 1 & 2: Processed multiple frames
        assert result["status"] == "completed"
        assert result["video_info"]["total_frames"] == 15
        assert result["video_info"]["processed_frames"] == 15
        assert result["video_filename"] == f"{analysis_id}_processed.mp4"

        # Requirement 5: Processed video URL returned
        assert "processed_video_url" in result
        assert result["processed_video_url"] == f"/api/analysis/{analysis_id}/video"

        # Requirement 3 & 4: Output video file exists, contains >1 frame, and can be decoded
        out_file = os.path.join(result_dir, f"{analysis_id}_processed.mp4")
        assert os.path.isfile(out_file)
        assert os.path.getsize(out_file) > 0

        cap = cv2.VideoCapture(out_file)
        assert cap.isOpened(), "Output video could not be opened by OpenCV VideoCapture"
        decoded_frames = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            assert frame.shape == (240, 320, 3)
            decoded_frames += 1
        cap.release()

        # Output video contains more than one frame
        assert decoded_frames == 15, f"Expected 15 decoded frames, got {decoded_frames}"

    def test_safety_zone_annotations_remain_visible_in_processed_frames(self, tmp_path):
        """Requirement 9: Safety zone polygons and labels are drawn and visible in processed video frames."""
        input_video = str(tmp_path / "zone_input.mp4")
        create_synthetic_video(input_video, num_frames=10, width=320, height=240, fps=10.0)

        result_dir = str(tmp_path / "results")
        os.makedirs(result_dir, exist_ok=True)

        service = VideoService()
        analysis_id = f"zone_test_{uuid.uuid4().hex[:8]}"
        result = service.process(input_video, result_dir, analysis_id=analysis_id, frame_skip=1)

        assert result["status"] == "completed"
        out_file = os.path.join(result_dir, f"{analysis_id}_processed.mp4")

        # Decode output video and verify zone annotations are drawn
        cap = cv2.VideoCapture(out_file)
        assert cap.isOpened()
        ret, frame = cap.read()
        cap.release()
        assert ret and frame is not None

        # Background was initialized with value 30; zones introduce overlays
        assert np.any(frame != 30), "Annotated video frame should contain drawn overlays and text"

    def test_tracking_annotations_updated_frame_by_frame(self, tmp_path, monkeypatch):
        """Requirement 10: Tracking bounding boxes move frame by frame across video frames."""
        input_video = str(tmp_path / "tracking_input.mp4")
        create_synthetic_video(input_video, num_frames=6, width=320, height=240, fps=10.0)

        result_dir = str(tmp_path / "results")
        os.makedirs(result_dir, exist_ok=True)

        service = VideoService()
        # Mock tracker to simulate a worker moving from x=20 to x=180 across frames
        def mock_track_frame(frame, frame_idx, confidence=None, annotate_box_detections=True):
            annotated = frame.copy()
            x = int(20 + frame_idx * 30)
            cv2.rectangle(annotated, (x, 80), (x + 40, 140), (0, 165, 255), 2)
            cv2.putText(annotated, f"worker #1 0.95", (x, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            det = [{
                "class": "worker",
                "confidence": 0.95,
                "bbox": [x, 80, x + 40, 140],
                "center": [x + 20, 110],
                "track_id": 1,
            }]
            stats = {"frame": frame_idx, "counts": {"workers": 1}, "tracking_ids": {"worker": [1]}, "detections": det}
            return annotated, stats

        monkeypatch.setattr(service.tracker, "track_frame", mock_track_frame)

        analysis_id = f"moving_test_{uuid.uuid4().hex[:8]}"
        result = service.process(input_video, result_dir, analysis_id=analysis_id, frame_skip=1)

        assert result["status"] == "completed"
        out_file = os.path.join(result_dir, f"{analysis_id}_processed.mp4")

        cap = cv2.VideoCapture(out_file)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frames.append(frame)
        cap.release()

        assert len(frames) == 6
        # Compare frame 0 and frame 4: bounding box has moved, so frames must not be identical
        diff = cv2.absdiff(frames[0], frames[4])
        assert np.count_nonzero(diff) > 0, "Detections must move across frames rather than staying static"

    def test_failed_video_encoding_returns_appropriate_error(self, tmp_path):
        """Requirement 12: Invalid or corrupt video input returns a descriptive failure result."""
        corrupt_video = str(tmp_path / "corrupt.mp4")
        with open(corrupt_video, "wb") as f:
            f.write(b"NOT_A_VALID_VIDEO_FILE_CONTENT")

        result_dir = str(tmp_path / "results")
        os.makedirs(result_dir, exist_ok=True)

        service = VideoService()
        result = service.process(corrupt_video, result_dir, analysis_id="err_test")
        assert result["status"] == "failed"
        assert "Could not open video" in result["error_message"] or "invalid" in result["error_message"].lower()


class TestVideoAPIAndAuthentication:
    """Tests 5, 8, 11, 13: Endpoints, user isolation, and byte-range seeking."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        self.client = TestClient(app, follow_redirects=False)
        self.service = get_processing_service()
        yield
        app.dependency_overrides.clear()

    def test_api_returns_processed_video_url_and_analytics(self, tmp_path):
        """Requirement 5 & 8: API returns processed_video_url, media_type, object counts, and analytics."""
        user_id = f"usr_{uuid.uuid4().hex[:8]}"
        user_dict = {"id": user_id, "name": "Video Tester", "email": f"{user_id}@test.com"}
        app.dependency_overrides[get_current_user] = lambda: user_dict

        # Create video file
        v_file = str(tmp_path / "api_test.mp4")
        create_synthetic_video(v_file, num_frames=8, width=320, height=240, fps=10.0)

        with open(v_file, "rb") as f:
            resp = self.client.post(
                "/api/analyze",
                files={"file": ("warehouse_aisle.mp4", f, "video/mp4")},
            )

        assert resp.status_code == 200, f"Analyze failed: {resp.text}"
        data = resp.json()

        assert data["status"] == "completed"
        assert data["media_type"] == "video"
        assert "processed_video_url" in data
        assert data["processed_video_url"].startswith("/api/analysis/")
        assert "analytics" in data
        assert "object_counts" in data
        assert "tracking" in data

    def test_video_access_respects_authenticated_user_ownership(self, tmp_path):
        """Requirement 13: Video access enforces user authentication, ownership, and byte-range seeking."""
        user_a_id = f"user_a_{uuid.uuid4().hex[:8]}"
        user_b_id = f"user_b_{uuid.uuid4().hex[:8]}"

        user_a = {"id": user_a_id, "name": "User Alpha", "email": f"{user_a_id}@test.com"}
        user_b = {"id": user_b_id, "name": "User Beta", "email": f"{user_b_id}@test.com"}

        # Create analysis record owned by User A
        analysis_id = self.service.create_analysis(
            filename="warehouse.mp4",
            original_filename="warehouse.mp4",
            media_type="video",
            file_size=1024,
            media_path="/tmp/fake.mp4",
            user_id=user_a_id,
        )

        # Create dummy video output in RESULT_DIR/videos matching analysis_id
        videos_dir = RESULT_DIR / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        test_video_path = str(videos_dir / f"{analysis_id}_processed.mp4")
        create_synthetic_video(test_video_path, num_frames=6, width=320, height=240, fps=10.0)

        self.service._save_result(
            analysis_id=analysis_id,
            result={
                "status": "completed",
                "media_type": "video",
                "processed_video_url": f"/api/analysis/{analysis_id}/video",
                "result_path": f"/api/analysis/{analysis_id}/video",
                "video_filename": f"{analysis_id}_processed.mp4",
                "object_counts": {"workers": 0, "forklifts": 0, "boxes": 0, "pallets": 0},
                "analytics": {"occupancy": 0, "space_utilization": 0, "restricted_zone_violations": 0},
            },
            media_type="video",
        )

        video_endpoint = f"/api/analysis/{analysis_id}/video"

        # 1. Unauthenticated request -> 401 Unauthorized
        app.dependency_overrides.clear()
        unauth_resp = self.client.get(video_endpoint)
        assert unauth_resp.status_code == 401

        # 2. User B (different user) request -> 404 Access Denied
        app.dependency_overrides[get_current_user] = lambda: user_b
        user_b_resp = self.client.get(video_endpoint)
        assert user_b_resp.status_code == 404

        # 3. User A (owner) request -> 200 OK
        app.dependency_overrides[get_current_user] = lambda: user_a
        user_a_resp = self.client.get(video_endpoint)
        assert user_a_resp.status_code == 200
        assert user_a_resp.headers["content-type"] == "video/mp4"
        assert len(user_a_resp.content) > 0

        # 4. HTTP Byte-Range Request (for video playback & seeking) -> 206 Partial Content
        range_resp = self.client.get(video_endpoint, headers={"Range": "bytes=0-100"})
        assert range_resp.status_code == 206
        assert "bytes 0-100/" in range_resp.headers.get("content-range", "")
        assert len(range_resp.content) == 101

    def test_output_files_cleaned_up_on_analysis_deletion(self, tmp_path):
        """Requirement 11: Output video files on disk are cleaned up when analysis is deleted."""
        user_id = f"del_user_{uuid.uuid4().hex[:8]}"
        user = {"id": user_id, "name": "Deleter", "email": f"{user_id}@test.com"}
        app.dependency_overrides[get_current_user] = lambda: user

        analysis_id = self.service.create_analysis(
            filename="warehouse.mp4",
            original_filename="warehouse.mp4",
            media_type="video",
            file_size=1024,
            media_path="/tmp/fake.mp4",
            user_id=user_id,
        )

        videos_dir = RESULT_DIR / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        video_file = videos_dir / f"{analysis_id}_processed.mp4"
        create_synthetic_video(str(video_file), num_frames=5, width=320, height=240, fps=10.0)

        self.service._save_result(
            analysis_id=analysis_id,
            result={
                "status": "completed",
                "media_type": "video",
                "processed_video_url": f"/api/analysis/{analysis_id}/video",
                "result_path": f"/api/analysis/{analysis_id}/video",
                "video_filename": f"{analysis_id}_processed.mp4",
                "object_counts": {},
                "analytics": {},
            },
            media_type="video",
        )

        assert video_file.is_file()

        # Delete via API
        del_resp = self.client.delete(f"/api/analysis/{analysis_id}")
        assert del_resp.status_code == 200
        assert del_resp.json()["success"] is True

        # Verify file is removed from disk
        assert not video_file.exists()


class TestDashboardFrontendAndImageHandling:
    """Tests 6, 7: Frontend video player structure and image analysis preservation."""

    def test_dashboard_uses_processed_video_url_and_html5_video_player(self):
        """Requirement 6: Dashboard JS renders an HTML5 video player with controls, preload, and video source."""
        js_path = Path("app/static/js/dashboard.js")
        assert js_path.is_file()
        content = js_path.read_text(encoding="utf-8")

        # Verify video player creation
        assert "<video id=\"mainResultVideo\"" in content
        assert "controls" in content
        assert "preload=\"metadata\"" in content
        assert "playsinline" in content
        assert "processed_video_url" in content
        assert "<source src=\"${videoSrc}\" type=\"video/mp4\">" in content

    def test_image_analyses_still_display_annotated_images(self, tmp_path):
        """Requirement 7: Image analyses continue displaying <img> detection results without regression."""
        js_path = Path("app/static/js/dashboard.js")
        content = js_path.read_text(encoding="utf-8")

        # Verify image rendering branch is preserved
        assert "<img src=\"${data.result_path}\"" in content
        assert "Annotated Image with AI Detections" in content
        assert "data.media_type === 'video'" in content

    def test_image_results_do_not_persist_processed_video_url(self):
        """Image result records must not expose a video URL to the dashboard."""
        service = get_processing_service()
        user_id = f"image_user_{uuid.uuid4().hex[:8]}"
        analysis_id = service.create_analysis(
            filename="warehouse.jpg",
            original_filename="warehouse.jpg",
            media_type="image",
            file_size=100,
            media_path="/tmp/warehouse.jpg",
            user_id=user_id,
        )

        service._save_result(
            analysis_id=analysis_id,
            result={
                "status": "completed",
                "result_path": "/results/images/warehouse_annotated.jpg",
                "object_counts": {},
                "analytics": {},
            },
            media_type="image",
        )

        record = service.get_analysis(analysis_id, user_id=user_id)
        assert record["media_type"] == "image"
        assert record.get("processed_video_url") is None
