"""Reachy Mini Gestures - entry point.

Threads:
- vision: newest camera frame -> recognizer -> gesture logic -> motion intents
- robot link: connection + reconnects, wake/sleep, 50Hz motion engine
- daemon monitor: robot model, daemon health, robot-camera pause policy
- uvicorn: HTTPS UI, MJPEG video, WebSocket telemetry and phone camera
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import socket
import sys
import threading
import time
import traceback

import cv2
import segno
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .camera_sources import SOURCES, CameraHub, encode_display
from .gesture_logic import GESTURES, GestureLogic
from .motion_engine import SOUND_FILES, Ego, RobotLink, clip_sound, sound_file_path
from .recognizer import SHAPES, EngageConfig, HandRecognizer, TeachStore
from .robot_profiles import Advisor, DaemonMonitor

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(PKG_DIR, "static")
MODEL_PATH = os.path.join(PKG_DIR, "gesture_recognizer.task")


def _data_dir() -> str:
    """Where settings, taught examples and the HTTPS certificate live: the
    source checkout when run from one (keeps existing local data), otherwise a
    per-user folder, so an installed app never writes into site-packages."""
    path = os.environ.get("REACHY_GESTURES_DATA")
    if not path:
        repo = os.path.dirname(PKG_DIR)
        in_checkout = os.path.exists(os.path.join(repo, "pyproject.toml"))
        path = repo if in_checkout else os.path.join(os.path.expanduser("~"), ".reachy_mini_gestures")
    os.makedirs(path, exist_ok=True)
    return path


DATA_DIR = _data_dir()
TEACH_PATH = os.path.join(DATA_DIR, "teach_samples.json")
SETTINGS_PATH = os.path.join(DATA_DIR, "app_settings.json")
CAMERA_SOURCE_PATH = os.path.join(DATA_DIR, "camera_source.json")
CERT_PATH = os.path.join(DATA_DIR, "cert.pem")
KEY_PATH = os.path.join(DATA_DIR, "key.pem")
PORT = 8765
TEACH_HAND_MAX_AGE_S = 0.6
ROBOT_CAMERA_SILENT_S = 8.0
RECONNECT_COOLDOWN_S = 30.0
SOUND_OUTPUTS = ("auto", "robot", "computer")


def get_lan_ip() -> str:
    """LAN IP a phone on the same WiFi can reach (UDP connect sends nothing)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


LAN_IP = get_lan_ip()
PHONE_URL = f"https://{LAN_IP}:{PORT}/phone"
ASSET_VERSION = str(int(time.time()))

stop_event = threading.Event()


class Settings:
    DEFAULTS = {
        "sensitivity": 0.5,
        "zone_bottom": 0.12,
        "mirror": True,
        "sound": True,
        "sound_output": "auto",
        "free_usb": False,
    }

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.values = dict(self.DEFAULTS)
        try:
            with open(path) as f:
                self._apply(json.load(f))
        except (OSError, json.JSONDecodeError):
            pass

    def _apply(self, changes: dict) -> None:
        for key, value in changes.items():
            default = self.DEFAULTS.get(key)
            if default is None:
                continue
            if isinstance(default, bool):
                self.values[key] = bool(value)
            elif isinstance(default, str):
                if value in SOUND_OUTPUTS:
                    self.values[key] = value
            else:
                self.values[key] = float(value)
        self.values["sensitivity"] = min(1.0, max(0.0, self.values["sensitivity"]))
        self.values["zone_bottom"] = min(0.4, max(0.0, self.values["zone_bottom"]))

    def get(self, key: str):
        with self._lock:
            return self.values[key]

    def as_dict(self) -> dict:
        with self._lock:
            return dict(self.values)

    def update(self, changes: dict) -> dict:
        with self._lock:
            self._apply(changes)
            try:
                with open(self.path, "w") as f:
                    json.dump(self.values, f)
            except OSError:
                pass
            return dict(self.values)


class Telemetry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._version = 0
        self._packet: dict = {}
        self._video_seq = 0
        self._jpeg: bytes | None = None
        self._teach_hand = None
        self._fire_id = 0
        self._last_fire: dict | None = None
        self.pending: dict | None = None

    def publish(self, packet: dict, jpeg: bytes | None = None, teach_hand=None) -> None:
        with self._lock:
            self._version += 1
            self._packet = packet
            if jpeg is not None:
                self._jpeg = jpeg
                self._video_seq += 1
            if teach_hand is not None:
                self._teach_hand = teach_hand

    def record_fire(self, name: str, ok: bool, sound: str | None) -> None:
        with self._lock:
            self._fire_id += 1
            self._last_fire = {"id": self._fire_id, "name": name, "ok": ok, "sound": sound}

    def last_fire(self) -> dict | None:
        with self._lock:
            return dict(self._last_fire) if self._last_fire else None

    def packet(self) -> tuple[int, dict]:
        with self._lock:
            return self._version, self._packet

    def video(self) -> tuple[int, bytes | None]:
        with self._lock:
            return self._video_seq, self._jpeg

    def teach_hand(self):
        with self._lock:
            return self._teach_hand


settings = Settings(SETTINGS_PATH)
engage_cfg = EngageConfig()
logic = GestureLogic()
hub = CameraHub(CAMERA_SOURCE_PATH)
teach = TeachStore(TEACH_PATH)
telemetry = Telemetry()
advisor = Advisor()

_watchdog = {"camera_seen": time.time(), "reconnect_at": 0.0}


def media_policy() -> bool:
    """Runs every couple of seconds on the monitor thread. Returns whether the
    daemon's robot camera/audio pipeline should be paused, and restarts the
    robot session if its camera went silent while everything else is fine."""
    now = time.time()
    cam = hub.status()
    if cam["source"] != "robot" or cam["state"] == "live" or link.state != "connected" or monitor.media_released:
        _watchdog["camera_seen"] = now
    elif now - _watchdog["camera_seen"] > ROBOT_CAMERA_SILENT_S and now - _watchdog["reconnect_at"] > RECONNECT_COOLDOWN_S:
        print("[watchdog] robot camera silent while connected; refreshing the robot session")
        link.command("reconnect")
        _watchdog["reconnect_at"] = now
    return bool(settings.get("free_usb")) and hub.selected != "robot" and monitor.profile.can_release_media


def sound_target() -> str | None:
    """Where reaction sounds play: Reachy's speaker, this computer (browser
    audio, so any paired Bluetooth speaker works), or nowhere."""
    if not settings.get("sound"):
        return None
    robot_speaker_ok = link.state == "connected" and not monitor.media_released
    output = settings.get("sound_output")
    if output == "computer":
        return "computer"
    if output == "robot":
        return "robot" if robot_speaker_ok else None
    return "robot" if robot_speaker_ok else "computer"


monitor = DaemonMonitor(policy=media_policy)
link = RobotLink(hub.robot, sound_enabled=lambda: sound_target() == "robot", stop_event=stop_event)


def apply_settings() -> None:
    logic.cfg.set_sensitivity(settings.get("sensitivity"))
    engage_cfg.zone_bottom = settings.get("zone_bottom")


apply_settings()


def build_packet(
    frame, decision, fps: float, proc_ms: float, vision_error: str | None = None, brightness: float | None = None
) -> dict:
    robot = link.status()
    camera = hub.status()
    daemon = monitor.snapshot()
    target = sound_target()
    hands = []
    mode, active, width, height, lat = "idle", None, 0, 0, None
    if frame is not None and decision is not None:
        height, width = frame.image.shape[:2]
        for tr in decision.tracks:
            if tr.last_seen != frame.t:
                continue
            hd = tr.hand
            hands.append(
                {
                    "id": tr.id,
                    "pts": [[round(float(p[0]), 4), round(float(p[1]), 4)] for p in hd.pts],
                    "engaged": hd.engaged,
                    "reject": hd.reject,
                    "shape": tr.stable,
                    "live": hd.shape,
                    "liveConf": round(hd.conf, 2),
                    "conf": round(tr.display_conf(), 2),
                }
            )
        mode, active = decision.mode, decision.active
        lat = {"vision": round((time.time() - frame.t) * 1000), "motor": robot["motor_ms"]}
    if robot["action"] and mode in ("idle", "watching"):
        mode = "action"
    if robot["sleeping"]:
        mode = "sleeping"
    tips = advisor.tips(
        daemon=daemon,
        camera=camera,
        robot=robot,
        proc_ms=proc_ms,
        hands=hands,
        settings=settings.as_dict(),
        sound_target=target,
        brightness=brightness,
    )
    return {
        "w": width,
        "h": height,
        "hands": hands,
        "mode": mode,
        "active": active,
        "fire": telemetry.last_fire(),
        "fps": round(fps, 1),
        "procMs": round(proc_ms, 1),
        "lat": lat,
        "robot": robot,
        "camera": camera,
        "daemon": daemon,
        "profile": daemon["profile"],
        "soundTarget": target,
        "tips": tips,
        "phoneUrl": PHONE_URL,
        "visionError": vision_error,
    }


def pick_teach_hands(hands):
    """Best hands first, so Teach takes the one (or two) the user is showing."""
    return sorted(hands, key=lambda h: (h.engaged, h.scale), reverse=True)


def vision_loop() -> None:
    try:
        recognizer = HandRecognizer(MODEL_PATH, teach, engage_cfg)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        telemetry.publish(build_packet(None, None, 0.0, 0.0, f"Could not load gesture model: {e}"))
        return

    last_key = None
    fps = 0.0
    last_done: float | None = None
    last_idle = 0.0
    brightness: float | None = None
    vision_error: str | None = None
    while not stop_event.is_set():
        frame = hub.latest()
        key = (hub.selected, frame.seq) if frame is not None else None
        if frame is None or key == last_key:
            now = time.time()
            if frame is None and now - last_idle > 0.25:
                telemetry.publish(build_packet(None, None, 0.0, 0.0, vision_error))
                last_idle = now
                fps = 0.0
            time.sleep(0.004)
            continue
        last_key = key
        try:
            t0 = time.time()
            hands = recognizer.process(frame.image, frame.t)
            engine = link.engine
            # Only the robot camera moves with Reachy; other cameras see a still scene.
            ego = engine.ego_at(frame.t) if engine is not None and hub.selected == "robot" else None
            if hub.selected == "phone" and hub.phone.shaking(frame.t):
                ego = Ego(busy=True)  # hand-held phone moving: the whole picture moves
            decision = logic.update(hands, frame.t, ego)
            if engine is not None:
                if decision.head_uv is not None:
                    engine.follow_head(decision.head_uv[0], decision.head_uv[1], hub.selected, frame.t, decision.head_gain)
                if decision.antennas_deg is not None:
                    engine.follow_antenna_angles(decision.antennas_deg[0], decision.antennas_deg[1], frame.t, decision.body_x)
            if decision.fire:
                ok = engine.trigger(decision.fire, frame.t) if engine is not None else False
                play_here = sound_target() == "computer"
                telemetry.record_fire(decision.fire, ok, clip_sound(decision.fire) if play_here else None)
                ego_txt = f"ego {ego.speed:.0f}deg/s" if ego is not None else "ego n/a"
                print(f"[gesture] {decision.fire} {decision.detail} {ego_txt} src {hub.selected} sent {ok}")
            jpeg = encode_display(frame)
            if frame.seq % 10 == 0 or brightness is None:
                small = cv2.resize(frame.image, (32, 18), interpolation=cv2.INTER_AREA)
                brightness = float(small.mean())
            done = time.time()
            if last_done is not None and done > last_done:
                inst = 1.0 / (done - last_done)
                fps = 0.85 * fps + 0.15 * inst if fps else inst
            last_done = done
            vision_error = None
            teach_hand = pick_teach_hands(hands) or None
            telemetry.publish(
                build_packet(frame, decision, fps, (done - t0) * 1000, brightness=brightness),
                jpeg=jpeg,
                teach_hand=(teach_hand, jpeg, frame.t) if teach_hand is not None else None,
            )
        except Exception as e:  # noqa: BLE001 - keep the loop alive
            vision_error = f"{type(e).__name__}: {e}"
            print(f"[vision] {vision_error}", file=sys.stderr)
            traceback.print_exc()
            time.sleep(0.05)


def ensure_self_signed_cert(lan_ip: str) -> None:
    """Mobile browsers only expose getUserMedia() on HTTPS (or localhost), so
    the app serves HTTPS with a self-signed cert generated on first run."""
    if os.path.exists(CERT_PATH) and os.path.exists(KEY_PATH):
        return

    import datetime
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    print("Generating a self-signed HTTPS certificate (first run only)...")
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Reachy Mini Gesture App")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                    x509.IPAddress(ipaddress.ip_address(lan_ip)),
                ]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    with open(KEY_PATH, "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()))
    with open(CERT_PATH, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))


app = FastAPI(title="Reachy Gestures")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
NO_CACHE = {"Cache-Control": "no-cache"}


def _page(filename: str) -> HTMLResponse:
    with open(os.path.join(STATIC_DIR, filename), encoding="utf-8") as f:
        return HTMLResponse(f.read().replace("__V__", ASSET_VERSION), headers=NO_CACHE)


@app.get("/")
def index() -> HTMLResponse:
    return _page("index.html")


@app.get("/phone")
def phone_page() -> HTMLResponse:
    return _page("phone.html")


@app.get("/api/state")
def api_state() -> dict:
    return {
        "gestures": GESTURES,
        "shapes": SHAPES,
        "source": hub.selected,
        "settings": settings.as_dict(),
        "phoneUrl": PHONE_URL,
        "teachCounts": teach.counts(),
        "profile": monitor.snapshot()["profile"],
        "sounds": SOUND_FILES,
    }


@app.get("/api/sound/{filename}")
def api_sound(filename: str) -> Response:
    path = sound_file_path(filename)
    if path is None:
        return JSONResponse({"error": "unknown sound"}, status_code=404)
    return FileResponse(path, media_type="audio/wav")


@app.get("/api/qr.svg")
def api_qr() -> Response:
    buf = io.BytesIO()
    segno.make(PHONE_URL, error="m").save(buf, kind="svg", scale=8, border=2, dark="#000000", light="#ffffff", xmldecl=False)
    return Response(buf.getvalue(), media_type="image/svg+xml", headers=NO_CACHE)


@app.post("/api/source")
async def api_source(request: Request) -> JSONResponse:
    body = await request.json()
    source = body.get("source")
    if source not in SOURCES:
        return JSONResponse({"error": f"source must be one of {', '.join(SOURCES)}"}, status_code=400)
    if source == "robot" and monitor.media_released:
        await asyncio.to_thread(monitor.set_media_released, False)
    await asyncio.to_thread(hub.select, source)
    monitor.nudge()
    return JSONResponse({"source": source})


@app.post("/api/settings")
async def api_settings(request: Request) -> dict:
    values = settings.update(await request.json())
    apply_settings()
    monitor.nudge()
    return values


@app.post("/api/robot/{action}")
def api_robot(action: str) -> JSONResponse:
    if action in ("wake", "sleep"):
        link.command(action)
    elif action == "restart":
        if not monitor.restart_daemon():
            return JSONResponse({"error": "Could not reach the robot service"}, status_code=502)
        link.command("reconnect")
    elif action == "stop":
        stop_event.set()
    else:
        return JSONResponse({"error": "unknown action"}, status_code=400)
    return JSONResponse({"ok": True})


@app.get("/api/teach/counts")
def api_teach_counts() -> dict:
    return teach.counts()


@app.post("/api/teach/capture/{shape}")
def api_teach_capture(shape: str, hands: int = 1) -> JSONResponse:
    if shape not in SHAPES:
        return JSONResponse({"error": f"'{shape}' can't be taught"}, status_code=400)
    need = max(1, min(2, hands))
    snap = telemetry.teach_hand()
    if snap is None or not snap[0] or time.time() - snap[2] > TEACH_HAND_MAX_AGE_S:
        return JSONResponse({"error": "Show your hand to the camera first"}, status_code=400)
    seen, jpeg, _ = snap
    if len(seen) < need:
        return JSONResponse({"error": "Show both hands to the camera"}, status_code=400)
    chosen = seen[:need]
    telemetry.pending = {"shape": shape, "samples": [(h.world.copy(), h.side) for h in chosen]}
    wrong = next((h for h in chosen if h.shape != shape), None)
    ignored = next((h for h in chosen if not h.engaged), None)
    return JSONResponse(
        {
            "expected": shape,
            "predicted": wrong.shape if wrong else shape,
            "conf": round(min(h.conf for h in chosen), 2),
            "correct": wrong is None,
            "engaged": ignored is None,
            "reject": ignored.reject if ignored else None,
            "hands": need,
            "snapshot": base64.b64encode(jpeg).decode("ascii") if jpeg else None,
        }
    )


@app.post("/api/teach/confirm")
def api_teach_confirm() -> JSONResponse:
    pending = telemetry.pending
    if pending is None:
        return JSONResponse({"error": "Nothing to save"}, status_code=400)
    count = 0
    for world, side in pending["samples"]:
        count = teach.add(pending["shape"], world, side)
    telemetry.pending = None
    return JSONResponse({"shape": pending["shape"], "count": count, "added": len(pending["samples"])})


@app.post("/api/teach/discard")
def api_teach_discard() -> dict:
    telemetry.pending = None
    return {"ok": True}


@app.post("/api/teach/clear/{shape}")
def api_teach_clear(shape: str) -> dict:
    teach.clear(None if shape == "all" else shape)
    return teach.counts()


@app.get("/status")
def legacy_status() -> dict:
    """Kept for manage.ps1."""
    _, p = telemetry.packet()
    robot = p.get("robot") or link.status()
    camera = p.get("camera") or hub.status()
    daemon = p.get("daemon") or {}
    return {
        "gesture": p.get("active") or "none",
        "mode": p.get("mode", "starting"),
        "hands_detected": sum(1 for h in p.get("hands", []) if h["engaged"]),
        "fps": p.get("fps", 0.0),
        "last_error": daemon.get("error") or robot.get("error") or camera.get("error") or p.get("visionError"),
        "phone_connected": hub.phone.connected(),
        "phone_url": PHONE_URL,
        "suppressed": ", ".join(f"{k} {v}" for k, v in sorted(logic.suppressed.items())) or "none",
    }


@app.get("/video")
async def video() -> StreamingResponse:
    async def frames():
        last = -1
        while True:
            seq, jpeg = telemetry.video()
            if jpeg is not None and seq != last:
                last = seq
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
            else:
                await asyncio.sleep(0.008)

    return StreamingResponse(
        frames(), media_type="multipart/x-mixed-replace; boundary=frame", headers={"Cache-Control": "no-store"}
    )


@app.websocket("/ws/ui")
async def ws_ui(ws: WebSocket) -> None:
    await ws.accept()
    last_version, last_sent = -1, 0.0
    try:
        while True:
            version, packet = telemetry.packet()
            now = time.time()
            if packet and (version != last_version or now - last_sent > 0.5):
                await ws.send_text(json.dumps(packet))
                last_version, last_sent = version, now
            await asyncio.sleep(1 / 60)
    except (WebSocketDisconnect, RuntimeError):
        pass


@app.websocket("/ws/phone")
async def ws_phone(ws: WebSocket) -> None:
    await ws.accept()
    hub.phone.clients += 1
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            data = message.get("bytes")
            if data:
                hub.phone.push(data)
                await ws.send_text("ok")
            elif message.get("text"):
                try:
                    hub.phone.set_motion(float(json.loads(message["text"]).get("motion", 0)))
                except (ValueError, TypeError, AttributeError):
                    pass
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        hub.phone.clients -= 1


def _quiet_client_resets(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    # Windows' proactor loop logs a traceback whenever a browser drops a
    # connection (page reload, tab close); that is normal, not an error.
    if isinstance(context.get("exception"), ConnectionResetError):
        return
    loop.default_exception_handler(context)


@app.on_event("startup")
async def _install_loop_handler() -> None:
    asyncio.get_running_loop().set_exception_handler(_quiet_client_resets)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    if not os.path.exists(MODEL_PATH):
        print(f"[fatal] gesture model not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)
    ensure_self_signed_cert(LAN_IP)

    monitor.start()
    link.start()
    threading.Thread(target=vision_loop, name="vision", daemon=True).start()

    config = uvicorn.Config(
        app, host="0.0.0.0", port=PORT, log_level="warning", ssl_certfile=CERT_PATH, ssl_keyfile=KEY_PATH
    )
    server = uvicorn.Server(config)
    print(f"Reachy Gestures running. Desktop: https://localhost:{PORT}  Phone: {PHONE_URL}")

    def watch_for_stop() -> None:
        stop_event.wait()
        link.join(timeout=8.0)
        monitor.stop()
        if monitor.media_released:
            monitor.set_media_released(False)
        server.should_exit = True

    threading.Thread(target=watch_for_stop, daemon=True).start()
    server.run()
    stop_event.set()


if __name__ == "__main__":
    main()
