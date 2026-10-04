"""Flask web application factory and route definitions."""

import os
from pathlib import Path
import secrets
import time
from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for
import cv2
import numpy as np

from src.web.auth import init_db, verify_user


def make_placeholder_frame() -> bytes:
    """Generate a clean dark placeholder JPEG frame."""
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:] = (32, 27, 17)  # #0B1220 in BGR
    text = "Camera inactive"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
    x = (640 - tw) // 2
    y = (360 + th) // 2
    cv2.putText(img, text, (x, y), font, scale, (138, 151, 173), 2, cv2.LINE_AA)
    ret, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes() if ret else b""


PLACEHOLDER_JPEG = make_placeholder_frame()


def create_app(config: dict | None = None) -> Flask:
    """Create and configure the Flask web application."""
    base_dir = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        template_folder=str(base_dir / "templates"),
        static_folder=str(base_dir / "static"),
    )

    # 1. Secret Key configuration
    secret_key = None
    if config and "SECRET_KEY" in config:
        secret_key = config["SECRET_KEY"]

    if not secret_key:
        secret_key = os.environ.get("BV_SECRET_KEY")

    if not secret_key:
        key_file = Path("outputs/secret_key.txt")
        if key_file.is_file():
            secret_key = key_file.read_text(encoding="utf-8").strip()
        if not secret_key:
            secret_key = secrets.token_hex(32)
            key_file.parent.mkdir(parents=True, exist_ok=True)
            key_file.write_text(secret_key, encoding="utf-8")

    app.secret_key = secret_key

    # 2. Database configuration
    users_db = "outputs/users.db"
    if config and "USERS_DB" in config:
        users_db = config["USERS_DB"]
    app.config["USERS_DB"] = str(users_db)
    init_db(app.config["USERS_DB"])

    # 3. Testing configuration
    if config and config.get("TESTING"):
        app.config["TESTING"] = True

    # 4. Cookie security
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    # 5. Camera worker configuration
    _worker = None
    if config and "camera_worker" in config:
        _worker = config["camera_worker"]

    def get_worker():
        nonlocal _worker
        if _worker is None:
            from src.web.camera import CameraWorker
            _worker = CameraWorker()
        return _worker

    @app.route("/")
    def index():
        if "user_email" in session:
            return redirect(url_for("app_page"))
        return redirect(url_for("login"))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "GET":
            if "user_email" in session:
                return redirect(url_for("app_page"))
            return render_template("login.html")

        email = request.form.get("email", "")
        password = request.form.get("password", "")

        if verify_user(app.config["USERS_DB"], email, password):
            session.clear()
            session["user_email"] = email.strip().lower()
            return redirect(url_for("app_page"))

        # Constant delay on failure to resist timing attacks
        time.sleep(0.5)
        return render_template("login.html", error="Invalid email or password")

    @app.route("/app")
    def app_page():
        if "user_email" not in session:
            return redirect(url_for("login"))
        return render_template("app.html", email=session["user_email"], active_page="overview")

    @app.route("/dashboard")
    def dashboard():
        if "user_email" not in session:
            return redirect(url_for("login"))
        return render_template("dashboard.html", email=session["user_email"], active_page="dashboard")

    @app.route("/results")
    def results():
        if "user_email" not in session:
            return redirect(url_for("login"))
        return render_template("results.html", email=session["user_email"], active_page="results")

    @app.route("/insights")
    def insights():
        if "user_email" not in session:
            return redirect(url_for("login"))
        return render_template("insights.html", email=session["user_email"], active_page="insights")

    @app.route("/logout", methods=["POST"])
    def logout():
        worker = get_worker()
        if worker is not None and hasattr(worker, "stop"):
            worker.stop()
        session.clear()
        return redirect(url_for("signed_out"))

    @app.route("/signed-out")
    def signed_out():
        return render_template("signed_out.html")

    @app.route("/video_feed")
    def video_feed():
        if "user_email" not in session:
            return redirect(url_for("login"))

        worker = get_worker()

        def generate():
            try:
                while True:
                    if not worker.running:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + PLACEHOLDER_JPEG + b"\r\n"
                        )
                        time.sleep(0.2)  # ~5 frames per second
                    else:
                        jpeg = worker.get_latest_jpeg() or PLACEHOLDER_JPEG
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                        )
                        time.sleep(0.1)  # ~10 frames per second
            except (GeneratorExit, BrokenPipeError, ConnectionResetError):
                pass

        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/start", methods=["POST"])
    def api_start():
        if "user_email" not in session:
            return jsonify({"error": "Unauthorized"}), 401
        if request.headers.get("X-BV-Request") != "1":
            return jsonify({"error": "Missing or invalid X-BV-Request header"}), 400
        worker = get_worker()
        worker.start()
        return jsonify(worker.status())

    @app.route("/api/stop", methods=["POST"])
    def api_stop():
        if "user_email" not in session:
            return jsonify({"error": "Unauthorized"}), 401
        if request.headers.get("X-BV-Request") != "1":
            return jsonify({"error": "Missing or invalid X-BV-Request header"}), 400
        worker = get_worker()
        worker.stop()
        return jsonify(worker.status())

    @app.route("/api/reset", methods=["POST"])
    def api_reset():
        if "user_email" not in session:
            return jsonify({"error": "Unauthorized"}), 401
        if request.headers.get("X-BV-Request") != "1":
            return jsonify({"error": "Missing or invalid X-BV-Request header"}), 400
        worker = get_worker()
        worker.reset()
        return jsonify(worker.status())

    @app.route("/api/status", methods=["GET"])
    def api_status():
        if "user_email" not in session:
            return jsonify({"error": "Unauthorized"}), 401
        worker = get_worker()
        return jsonify(worker.status())

    return app
