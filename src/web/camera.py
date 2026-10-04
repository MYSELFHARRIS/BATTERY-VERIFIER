"""Camera worker managing background capture, LiveSession processing, and status."""

import collections
from datetime import datetime
from pathlib import Path
import threading
import time
from typing import Any, Callable

import cv2
import numpy as np

from src.logic.state_machine import EventType
from src.pipeline.overlay import draw_overlay


class CameraWorker:
    """Manages video capture loop, LiveSession integration, and real-time statistics."""

    def __init__(
        self,
        session_factory: Callable[..., Any] | None = None,
        frame_source_factory: Callable[..., Any] | None = None,
        camera_index: int = 0,
    ) -> None:
        self.session_factory = session_factory
        self.frame_source_factory = frame_source_factory
        self.camera_index = camera_index

        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

        self.running: bool = False
        self.error: str | None = None
        self._has_started: bool = False
        self.chain_ok: bool | None = None

        self.session: Any | None = None
        self._latest_jpeg: bytes | None = None
        self._events: list[dict[str, Any]] = []
        self._counts: dict[str, int] = {
            "STEP_VERIFIED": 0,
            "WRONG_ORIENTATION": 0,
            "SKIPPED_STEP": 0,
        }
        self.fps: float = 0.0
        self.ms_per_frame: float = 0.0

        try:
            from src.logic.sop_loader import load_sop
            _, self._default_steps = load_sop("configs/sop.yaml")
        except Exception:
            self._default_steps = []

    def start(self) -> None:
        """Start the background camera capture loop if not already running."""
        with self._lock:
            if self.running or (self._thread and self._thread.is_alive()):
                return

            db_path = Path("outputs/web_session.db")
            if self.session is not None and getattr(self.session, "log", None) is not None:
                try:
                    self.session.log.close()
                except Exception:
                    pass
                self.session = None

            if db_path.exists():
                try:
                    db_path.unlink()
                except Exception:
                    pass
            for ext in ("-journal", "-wal", "-shm"):
                aux = Path(f"outputs/web_session.db{ext}")
                if aux.exists():
                    try:
                        aux.unlink()
                    except Exception:
                        pass
            db_path.parent.mkdir(parents=True, exist_ok=True)

            self._has_started = True
            self.chain_ok = True
            self.error = None
            self.running = True

            self._thread = threading.Thread(
                target=self._run_loop, args=(str(db_path),), daemon=True
            )
            self._thread.start()

    def _run_loop(self, db_path: str) -> None:
        """Capture loop executed in a background thread."""
        try:
            self.session = self._make_session(db_path)
        except Exception as e:
            with self._lock:
                self.error = str(e)
                self.running = False
            return

        try:
            cap = self._make_frame_source()
        except Exception:
            with self._lock:
                self.error = "Camera could not be opened"
                self.running = False
            return

        if cap is None:
            with self._lock:
                self.error = "Camera could not be opened"
                self.running = False
            return

        if hasattr(cap, "isOpened") and not cap.isOpened():
            with self._lock:
                self.error = "Camera could not be opened"
                self.running = False
            if hasattr(cap, "release"):
                try:
                    cap.release()
                except Exception:
                    pass
            return

        frame_index = 0
        recent_durations: collections.deque = collections.deque(maxlen=30)
        prev_logged_key = None
        consecutive_failures = 0

        try:
            while self.running:
                t0 = time.perf_counter()
                ret, frame = cap.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures > 50:
                        break
                    time.sleep(0.01)
                    continue

                consecutive_failures = 0

                boxes, event = self.session.process(frame)

                # Check if LiveSession logged an event
                if self.session._last_logged_key != prev_logged_key:
                    prev_logged_key = self.session._last_logged_key
                    now_str = datetime.now().strftime("%H:%M:%S")
                    conf = float(event.confidence) if event.confidence is not None else 0.0
                    ev_type_str = (
                        event.type.name
                        if hasattr(event.type, "name")
                        else str(event.type)
                    )

                    with self._lock:
                        if ev_type_str in self._counts:
                            self._counts[ev_type_str] += 1
                        self._events.append(
                            {
                                "time": now_str,
                                "frame": frame_index,
                                "type": ev_type_str,
                                "step": event.step_id,
                                "message": event.message,
                                "confidence": conf,
                            }
                        )
                        if len(self._events) > 200:
                            self._events.pop(0)

                # Determine overlay step_text
                if self.session.engine.is_complete:
                    step_text = "COMPLETE"
                elif self.session.engine.current_step is not None:
                    step_text = (
                        f"Step {self.session.engine.current_step.id}/{len(self.session.engine.steps)}: "
                        f"{self.session.engine.current_step.name}"
                    )
                else:
                    step_text = "COMPLETE"

                annotated = draw_overlay(
                    frame_bgr=frame,
                    boxes=boxes,
                    slots=self.session.slots,
                    step_text=step_text,
                    event_text=self.session.last_event_text,
                    event_type=self.session.last_event_type,
                    frame_index=frame_index,
                )

                ret_enc, buf = cv2.imencode(
                    ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                )
                if ret_enc:
                    with self._lock:
                        self._latest_jpeg = buf.tobytes()

                dt = time.perf_counter() - t0
                recent_durations.append(dt)
                avg_dt = sum(recent_durations) / len(recent_durations)
                avg_ms = avg_dt * 1000.0
                cur_fps = (1.0 / avg_dt) if avg_dt > 0 else 0.0

                chain_ok = True
                if self.session and self.session.log:
                    try:
                        chain_ok, _ = self.session.log.verify_chain()
                    except Exception:
                        chain_ok = False

                with self._lock:
                    self.ms_per_frame = round(avg_ms, 1)
                    self.fps = round(cur_fps, 1)
                    self.chain_ok = chain_ok

                frame_index += 1
        except Exception as e:
            with self._lock:
                self.error = str(e)
        finally:
            if hasattr(cap, "release"):
                try:
                    cap.release()
                except Exception:
                    pass
            if self.session and self.session.log:
                try:
                    chain_ok, _ = self.session.log.verify_chain()
                    with self._lock:
                        self.chain_ok = chain_ok
                except Exception:
                    pass
            with self._lock:
                self.running = False

    def stop(self) -> None:
        """Stop the background capture loop."""
        with self._lock:
            self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def reset(self) -> None:
        """Reset the LiveSession sequence."""
        if self.session is not None:
            self.session.reset()
        with self._lock:
            self._counts = {
                "STEP_VERIFIED": 0,
                "WRONG_ORIENTATION": 0,
                "SKIPPED_STEP": 0,
            }
            self._events.clear()
            self.error = None

    def get_latest_jpeg(self) -> bytes | None:
        """Retrieve the latest annotated JPEG bytes."""
        with self._lock:
            return self._latest_jpeg

    def status(self) -> dict[str, Any]:
        """Return the current worker status dictionary."""
        with self._lock:
            running = self.running
            error = self.error
            events = list(reversed(self._events))[:50]
            counts = dict(self._counts)
            fps = float(self.fps)
            ms_per_frame = float(self.ms_per_frame)
            chain_ok = self.chain_ok

        if self.session is not None:
            engine = self.session.engine
            complete = engine.is_complete
            if complete:
                step_text = "COMPLETE"
            elif engine.current_step is not None:
                step_text = (
                    f"Step {engine.current_step.id}/{len(engine.steps)}: "
                    f"{engine.current_step.name}"
                )
            else:
                step_text = "COMPLETE"

            steps = []
            for idx, s in enumerate(engine.steps):
                if complete or idx < engine._current_step_index:
                    st = "done"
                elif idx == engine._current_step_index:
                    st = "current"
                else:
                    st = "pending"
                steps.append({"id": s.id, "name": s.name, "state": st})
        else:
            complete = False
            step_text = "Ready"
            steps = []
            for s in self._default_steps:
                steps.append({"id": s.id, "name": s.name, "state": "pending"})

        return {
            "running": running,
            "error": error,
            "complete": complete,
            "step_text": step_text,
            "steps": steps,
            "events": events,
            "counts": counts,
            "fps": fps,
            "ms_per_frame": ms_per_frame,
            "chain_ok": chain_ok,
        }

    def _make_session(self, db_path: str) -> Any:
        if self.session_factory is not None:
            try:
                return self.session_factory(db_path)
            except TypeError:
                return self.session_factory()

        from src.detection.detector import YoloDetector
        from src.logic.sop_loader import load_sop
        from src.logic.state_machine import StepEngine
        from src.logic.verification_log import VerificationLog
        from src.pipeline.adapter import load_slots
        from src.pipeline.live import LiveSession

        slots = load_slots("configs/slots.yaml")
        sop_version, steps = load_sop("configs/sop.yaml")
        engine = StepEngine(steps)
        detector = YoloDetector("models/cells_v3.pt")
        log = VerificationLog(db_path)
        return LiveSession(
            detector=detector,
            slots=slots,
            engine=engine,
            log=log,
            sop_version=sop_version,
        )

    def _make_frame_source(self) -> Any:
        if self.frame_source_factory is not None:
            try:
                return self.frame_source_factory(self.camera_index)
            except TypeError:
                return self.frame_source_factory()

        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap
