"""Tests for web API endpoints and camera worker integration."""

import time
import cv2
import numpy as np
import pytest

from src.detection.detector import BoxDetection
from src.logic.sop_loader import Step
from src.logic.state_machine import StepEngine
from src.logic.verification_log import VerificationLog
from src.pipeline.adapter import Slot
from src.pipeline.live import LiveSession
from src.web.app import create_app
from src.web.auth import create_user
from src.web.camera import CameraWorker


class ScriptedDetector:
    """Fake detector returning scripted BoxDetections across frame invocations."""

    def __init__(self, script: list[list[BoxDetection]]):
        self.script = script
        self.call_count = 0

    def detect(self, frame_bgr: np.ndarray) -> list[BoxDetection]:
        if self.call_count < len(self.script):
            res = self.script[self.call_count]
        else:
            res = []
        self.call_count += 1
        return res


class FakeFrameSource:
    """Fake video frame source that yields frames and can be cleanly closed."""

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.is_open = True

    def isOpened(self) -> bool:
        return self.is_open

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.is_open:
            return False, None
        time.sleep(0.01)  # small throttle for thread
        return True, np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def release(self) -> None:
        self.is_open = False


class UnopenableSource:
    """Camera source that fails to open."""

    def isOpened(self) -> bool:
        return False

    def read(self) -> tuple[bool, None]:
        return False, None

    def release(self) -> None:
        pass


@pytest.fixture
def test_setup(tmp_path):
    users_db = str(tmp_path / "test_users.db")
    create_user(users_db, "operator@example.com", "Password123!")

    slots = [
        Slot(id=1, x_min=0.0, x_max=0.333, y_min=0.0, y_max=1.0),
        Slot(id=2, x_min=0.333, x_max=0.666, y_min=0.0, y_max=1.0),
        Slot(id=3, x_min=0.666, x_max=1.0, y_min=0.0, y_max=1.0),
    ]
    steps = [
        Step(id=1, name="Inspect cell 1", slot=1, required_class="cell_up"),
        Step(id=2, name="Inspect cell 2", slot=2, required_class="cell_down"),
        Step(id=3, name="Inspect cell 3", slot=3, required_class="cell_up"),
    ]

    # Build scripted 15 frames: 5 cell_up in slot 1, 5 cell_down in slot 2, 5 cell_up in slot 3
    box_s1 = BoxDetection(label="cell_up", confidence=0.95, xyxy=(100, 260, 300, 460))
    box_s2 = BoxDetection(label="cell_down", confidence=0.92, xyxy=(540, 260, 740, 460))
    box_s3 = BoxDetection(label="cell_up", confidence=0.98, xyxy=(900, 260, 1100, 460))

    script = [[box_s1]] * 5 + [[box_s2]] * 5 + [[box_s3]] * 5
    detector = ScriptedDetector(script)

    def session_factory(db_path: str):
        return LiveSession(
            detector=detector,
            slots=slots,
            engine=StepEngine(steps, confirm_frames=5),
            log=VerificationLog(db_path),
            sop_version=1,
        )

    frame_source = FakeFrameSource()
    worker = CameraWorker(
        session_factory=session_factory,
        frame_source_factory=lambda idx: frame_source,
    )

    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test_secret_key",
            "USERS_DB": users_db,
            "camera_worker": worker,
        }
    )

    return app, worker, frame_source


def test_api_routes_return_401_when_not_signed_in(test_setup):
    app, _, _ = test_setup
    client = app.test_client()

    for path, method in [
        ("/api/status", "get"),
        ("/api/start", "post"),
        ("/api/stop", "post"),
        ("/api/reset", "post"),
    ]:
        func = getattr(client, method)
        resp = func(path, headers={"X-BV-Request": "1"})
        assert resp.status_code == 401
        assert resp.is_json
        assert resp.json == {"error": "Unauthorized"}


def test_post_without_header_returns_400(test_setup):
    app, _, _ = test_setup
    client = app.test_client()
    client.post(
        "/login",
        data={"email": "operator@example.com", "password": "Password123!"},
    )

    for path in ["/api/start", "/api/stop", "/api/reset"]:
        # Without header
        resp = client.post(path)
        assert resp.status_code == 400
        assert resp.is_json

        # With wrong header value
        resp_wrong = client.post(path, headers={"X-BV-Request": "0"})
        assert resp_wrong.status_code == 400


def test_start_run_completion_and_chain_ok(test_setup):
    app, worker, _ = test_setup
    client = app.test_client()
    client.post(
        "/login",
        data={"email": "operator@example.com", "password": "Password123!"},
    )

    # Initial chain_ok is None before start
    init_status = client.get("/api/status").json
    assert init_status["chain_ok"] is None
    assert init_status["running"] is False

    # Start
    start_resp = client.post("/api/start", headers={"X-BV-Request": "1"})
    assert start_resp.status_code == 200
    assert start_resp.json["running"] is True

    # Poll status until 3 steps are complete (at most 3 seconds)
    completed = False
    status_data = None
    for _ in range(30):
        time.sleep(0.1)
        resp = client.get("/api/status")
        status_data = resp.json
        if status_data.get("complete") is True:
            completed = True
            break

    assert completed, f"Sequence did not complete in time, status: {status_data}"
    assert status_data["counts"]["STEP_VERIFIED"] == 3
    assert status_data["chain_ok"] is True
    assert len(status_data["events"]) == 3
    # Newest event first
    assert status_data["events"][0]["step"] == 3
    assert status_data["events"][0]["type"] == "STEP_VERIFIED"
    assert status_data["events"][2]["step"] == 1

    # Stop sets running false
    stop_resp = client.post("/api/stop", headers={"X-BV-Request": "1"})
    assert stop_resp.status_code == 200
    assert stop_resp.json["running"] is False

    final_status = client.get("/api/status").json
    assert final_status["running"] is False
    assert final_status["chain_ok"] is True

    # Reset restarts the sequence
    reset_resp = client.post("/api/reset", headers={"X-BV-Request": "1"})
    assert reset_resp.status_code == 200
    assert reset_resp.json["complete"] is False
    assert reset_resp.json["counts"]["STEP_VERIFIED"] == 0
    assert reset_resp.json["steps"][0]["state"] == "current"


def test_unopenable_camera_gives_error_without_exception():
    worker = CameraWorker(
        session_factory=lambda db: None,
        frame_source_factory=lambda idx: UnopenableSource(),
    )
    worker.start()

    # Allow thread to execute
    for _ in range(20):
        if worker.status()["error"] is not None:
            break
        time.sleep(0.05)

    status = worker.status()
    assert status["running"] is False
    assert status["error"] == "Camera could not be opened"


def test_video_feed_returns_multipart_when_signed_in(test_setup):
    app, worker, _ = test_setup
    client = app.test_client()

    # Unsigned in redirects
    resp_anon = client.get("/video_feed")
    assert resp_anon.status_code == 302
    assert resp_anon.headers["Location"].endswith("/login")

    # Sign in
    client.post(
        "/login",
        data={"email": "operator@example.com", "password": "Password123!"},
    )

    # When signed in and camera stopped, returns placeholder multipart stream
    resp_signed = client.get("/video_feed")
    assert resp_signed.status_code == 200
    assert resp_signed.content_type.startswith("multipart/x-mixed-replace")
    stream_iter = iter(resp_signed.response)
    chunk = next(stream_iter)
    assert b"--frame" in chunk
    resp_signed.close()


def test_signed_in_pages_return_200_and_anon_redirects_to_login(test_setup):
    app, _, _ = test_setup
    client = app.test_client()

    for path in ["/dashboard", "/results", "/insights"]:
        resp_anon = client.get(path)
        assert resp_anon.status_code == 302
        assert resp_anon.headers["Location"].endswith("/login")

    # Sign in
    client.post(
        "/login",
        data={"email": "operator@example.com", "password": "Password123!"},
    )

    for path in ["/app", "/dashboard", "/results", "/insights"]:
        resp = client.get(path)
        assert resp.status_code == 200


def test_idle_video_feed_yields_at_least_two_frames(test_setup):
    app, worker, _ = test_setup
    client = app.test_client()
    client.post(
        "/login",
        data={"email": "operator@example.com", "password": "Password123!"},
    )

    assert worker.running is False
    resp = client.get("/video_feed")
    assert resp.status_code == 200
    assert resp.content_type.startswith("multipart/x-mixed-replace")

    stream_iter = iter(resp.response)
    chunk1 = next(stream_iter)
    chunk2 = next(stream_iter)

    assert b"--frame" in chunk1
    assert b"Content-Type: image/jpeg" in chunk1
    assert b"--frame" in chunk2
    assert b"Content-Type: image/jpeg" in chunk2
    resp.close()


def test_every_signed_in_page_contains_four_sidebar_links(test_setup):
    app, _, _ = test_setup
    client = app.test_client()
    client.post(
        "/login",
        data={"email": "operator@example.com", "password": "Password123!"},
    )

    for path in ["/app", "/dashboard", "/results", "/insights"]:
        resp = client.get(path)
        assert resp.status_code == 200
        html = resp.data.decode("utf-8")
        assert "Overview" in html
        assert "Dashboard" in html
        assert "Results" in html
        assert "Insights" in html


def test_logout_stops_camera_worker(test_setup):
    app, worker, _ = test_setup
    client = app.test_client()
    client.post(
        "/login",
        data={"email": "operator@example.com", "password": "Password123!"},
    )

    worker.running = True
    assert worker.running is True

    resp = client.post("/logout")
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/signed-out")
    assert worker.running is False

