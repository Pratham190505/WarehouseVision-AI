from types import SimpleNamespace

import numpy as np
import pytest

from app.vision.box_counting import BoxCountingEngine


def see(engine, tracker_id=1, frame=0, confidence=0.9, bbox=None):
    bbox = bbox or [10, 10, 30, 30]
    return engine.observe(tracker_id, bbox, [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], confidence, frame)


def test_one_box_seen_for_100_frames_counts_once():
    counter = BoxCountingEngine()
    for frame in range(100):
        track = see(counter, frame=frame)
        counter.advance_frame(frame)
    summary = counter.summary(99)
    assert summary["total_unique_boxes"] == 1
    assert summary["currently_visible_boxes"] == 1
    assert summary["confirmed_tracks"][0]["observations"] == 100
    assert track["box_id"] == "BOX-001"


def test_five_distinct_confirmed_tracks_count_as_five():
    counter = BoxCountingEngine()
    for frame in range(3):
        for tracker_id in range(5):
            see(counter, tracker_id=tracker_id, frame=frame)
        counter.advance_frame(frame)
    assert counter.summary(2)["total_unique_boxes"] == 5
    assert {track["box_id"] for track in counter.summary(2)["confirmed_tracks"]} == {
        "BOX-001", "BOX-002", "BOX-003", "BOX-004", "BOX-005"
    }


def test_long_lived_track_never_increments_more_than_once():
    counter = BoxCountingEngine()
    for frame in range(500):
        see(counter, tracker_id=42, frame=frame)
        counter.advance_frame(frame)
    assert counter.total_unique_boxes == 1


def test_short_detection_gap_preserves_tracker_identity():
    counter = BoxCountingEngine()
    for frame in range(3):
        see(counter, tracker_id=7, frame=frame)
    counter.advance_frame(10)
    recovered = see(counter, tracker_id=7, frame=11)
    assert recovered["box_id"] == "BOX-001"
    assert counter.total_unique_boxes == 1


def test_short_lived_unconfirmed_false_track_is_never_counted():
    counter = BoxCountingEngine()
    see(counter, tracker_id=8, frame=0)
    see(counter, tracker_id=8, frame=1)
    counter.advance_frame(32)
    assert counter.summary(32)["total_unique_boxes"] == 0
    assert counter.tracks == {}


def test_low_confidence_track_does_not_confirm():
    counter = BoxCountingEngine(min_average_confidence=0.35)
    for frame in range(5):
        see(counter, tracker_id=9, frame=frame, confidence=0.2)
    assert counter.total_unique_boxes == 0


def test_each_engine_session_resets_count_and_readable_id_sequence():
    first = BoxCountingEngine()
    for frame in range(3):
        see(first, tracker_id=77, frame=frame)
    second = BoxCountingEngine()
    for frame in range(3):
        recovered = see(second, tracker_id=2, frame=frame)
    assert first.total_unique_boxes == second.total_unique_boxes == 1
    assert first.summary()["confirmed_tracks"][0]["box_id"] == "BOX-001"
    assert recovered["box_id"] == "BOX-001"


def test_visible_count_can_drop_without_changing_total():
    counter = BoxCountingEngine(visibility_window_frames=1)
    for frame in range(3):
        see(counter, frame=frame)
    counter.advance_frame(3)
    assert counter.summary(3)["currently_visible_boxes"] == 1
    counter.advance_frame(5)
    summary = counter.summary(5)
    assert summary["currently_visible_boxes"] == 0
    assert summary["total_unique_boxes"] == 1


def test_authenticated_analyze_creates_a_separate_session(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.services.auth_service import get_current_user
    import app.routes.box_tracking as route

    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    calls = {}

    class ProcessingStub:
        def get_analysis(self, analysis_id, user_id=None):
            if analysis_id != "uploaded-session" or user_id != "owner-1":
                return None
            return {"media_type": "video", "media_path": str(source), "filename": "stored.mp4",
                    "original_filename": "line.mp4", "file_size": 5}
        def create_analysis(self, **kwargs):
            calls["create"] = kwargs
            return "new-session"
        def process_media(self, analysis_id, media_path, media_type, box_tracking_config=None):
            calls["process"] = (analysis_id, media_path, media_type, box_tracking_config)
            return {"analysis_id": analysis_id, "status": "completed", "box_tracking": {"total_unique_boxes": 0}}

    monkeypatch.setattr(route, "get_processing_service", ProcessingStub)
    app.dependency_overrides[get_current_user] = lambda: {"id": "owner-1"}
    try:
        response = TestClient(app).post("/api/box-tracking/uploaded-session/analyze")
    finally:
        app.dependency_overrides.pop(get_current_user, None)
    assert response.status_code == 200
    assert response.json()["analysis_id"] == "new-session"
    assert calls["process"][0] == "new-session"
    assert calls["process"][3] == {}


def test_processed_video_route_is_owner_checked_and_private(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.services.auth_service import get_current_user
    import app.routes.box_tracking as route

    (tmp_path / "session_processed.mp4").write_bytes(b"private video")
    monkeypatch.setattr(route, "BOX_TRACKING_RESULT_DIR", tmp_path)

    class ProcessingStub:
        def get_analysis(self, analysis_id, user_id=None):
            if analysis_id != "session" or user_id != "owner":
                return None
            return {"box_tracking": {"output_filename": "session_processed.mp4"}}

    monkeypatch.setattr(route, "get_processing_service", ProcessingStub)
    app.dependency_overrides[get_current_user] = lambda: {"id": "owner"}
    try:
        response = TestClient(app).get("/api/box-tracking/session/result")
    finally:
        app.dependency_overrides.pop(get_current_user, None)
    assert response.status_code == 200
    assert response.content == b"private video"
    assert response.headers["content-disposition"].startswith("inline")


def test_overlay_and_result_summary_share_the_same_counter(monkeypatch, tmp_path):
    import app.services.video_service as video_module

    frames = [np.zeros((100, 200, 3), dtype=np.uint8) for _ in range(4)]
    overlay_text = []

    class Capture:
        def __init__(self, _path): self.index = 0
        def isOpened(self): return True
        def get(self, prop):
            return {video_module.cv2.CAP_PROP_FPS: 30, video_module.cv2.CAP_PROP_FRAME_COUNT: 4,
                    video_module.cv2.CAP_PROP_FRAME_WIDTH: 200, video_module.cv2.CAP_PROP_FRAME_HEIGHT: 100}[prop]
        def read(self):
            if self.index >= len(frames): return False, None
            frame = frames[self.index].copy()
            self.index += 1
            return True, frame
        def release(self): pass

    class Writer:
        def isOpened(self): return True
        def write(self, _frame): pass
        def release(self): pass

    class Detector:
        model = SimpleNamespace(names={5: "carton"})
        model_path = "carton.pt"
        box_detection_source = "dedicated model class"
        class_mapping = {5: "box"}
        def is_ready(self): return True
        def detect(self, _frame): return {"detections": []}

    class Tracker:
        def __init__(self, _detector): pass
        def reset(self): pass
        def get_summary(self): return {"unique_workers": 0, "unique_forklifts": 0, "unique_boxes": 1,
                                       "unique_pallets": 0, "total_frames": 4, "processed_frames": 4, "frame_data": []}
        def track_frame(self, frame, frame_idx, confidence=None, annotate_box_detections=True):
            detection = {"class": "box", "track_id": 17, "bbox": [90, 20, 110, 40],
                         "center": [100, 30], "confidence": 0.9}
            return frame.copy(), {"counts": {"boxes": 1}, "detections": [detection]}

    class Zones:
        has_zones = True
        zones = []
        def check_violations(self, _d): return {"total_violations": 0, "violations": []}
        def draw_zones(self, frame): return frame
        def annotate_violations(self, frame, _v): return frame

    class Safety:
        active_incidents = {}
        resolved_incidents = []
        config = SimpleNamespace(persistence_frames=3)
        def reset(self): pass
        def evaluate_detections(self, **_kwargs): return {"alerts": []}
        def annotate_frame(self, frame, *_args): return frame

    monkeypatch.setattr(video_module, "WarehouseDetector", Detector)
    monkeypatch.setattr(video_module, "WarehouseTracker", Tracker)
    monkeypatch.setattr(video_module, "ZoneDetector", Zones)
    monkeypatch.setattr(video_module, "SafetyZoneEvaluator", Safety)
    monkeypatch.setattr(video_module, "OccupancyAnalyzer", lambda: SimpleNamespace(analyze=lambda *_: {}))
    monkeypatch.setattr(video_module, "InventoryAnalyzer", lambda: SimpleNamespace(analyze=lambda *_: {}))
    monkeypatch.setattr(video_module, "WarehouseAnalytics", lambda: SimpleNamespace(compute=lambda **_: {}))
    monkeypatch.setattr(video_module.cv2, "VideoCapture", Capture)
    monkeypatch.setattr(video_module.cv2, "VideoWriter", lambda *_args: Writer())
    monkeypatch.setattr(video_module.cv2, "imwrite", lambda *_args: True)
    original_put_text = video_module.cv2.putText
    def capture_text(image, text, *args, **kwargs):
        overlay_text.append(text)
        return original_put_text(image, text, *args, **kwargs)
    monkeypatch.setattr(video_module.cv2, "putText", capture_text)

    result = video_module.VideoService().process("synthetic.mp4", str(tmp_path), "session-1", box_tracking_config={})
    summary = result["box_tracking"]
    assert result["status"] == "completed"
    assert summary["total_unique_boxes"] == result["tracking"]["unique_boxes"] == result["object_counts"]["boxes"] == 1
    assert summary["currently_visible_boxes"] == 1
    assert "TOTAL UNIQUE BOXES: 1" in overlay_text
    assert "CURRENTLY VISIBLE BOXES: 1" in overlay_text


def test_invalid_video_returns_useful_error_without_output(monkeypatch, tmp_path):
    import app.services.video_service as video_module

    class Capture:
        def __init__(self, _path): pass
        def isOpened(self): return False
        def release(self): pass

    class Detector:
        model = None
        class_mapping = {}
        def is_ready(self): return False

    monkeypatch.setattr(video_module, "WarehouseDetector", Detector)
    monkeypatch.setattr(video_module.cv2, "VideoCapture", Capture)
    result = video_module.VideoService().process("corrupt.mp4", str(tmp_path), "bad-session", box_tracking_config={})
    assert result["status"] == "failed"
    assert "Could not open video" in result["error_message"]
    assert list(tmp_path.iterdir()) == []
