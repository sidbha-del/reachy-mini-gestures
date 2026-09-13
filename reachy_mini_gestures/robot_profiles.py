"""Robot model profiles, daemon monitoring and live "what went wrong" hints.

Profiles keep model-specific facts (camera limits, what the daemon can pause)
in one place, so the app adapts to Reachy Mini Lite, Wireless and the
simulator instead of hard-coding one model. The advisor turns live telemetry
into short, dismissible hints with one-tap fixes.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.request
from collections import deque
from dataclasses import asdict, dataclass
from typing import Callable

DAEMON_URL = os.environ.get("REACHY_DAEMON_URL", "http://localhost:8000")


@dataclass(frozen=True)
class Profile:
    key: str
    name: str
    camera_fps: float
    camera_note: str
    can_release_media: bool


PROFILES = {
    "lite": Profile(
        "lite",
        "Reachy Mini Lite",
        10.0,
        "On Reachy Mini Lite it runs at about 10 fps because its camera, speaker and motors share one internal USB 2.0 hub.",
        True,
    ),
    "wireless": Profile(
        "wireless",
        "Reachy Mini Wireless",
        25.0,
        "On Reachy Mini Wireless it streams over Wi-Fi, which can add a little delay.",
        False,
    ),
    "sim": Profile("sim", "Reachy Mini simulator", 30.0, "In the simulator the robot camera is virtual.", False),
    "unknown": Profile("unknown", "Reachy Mini", 15.0, "Its speed depends on the robot model and connection.", False),
}


def detect_profile(status: dict | None) -> Profile:
    if not status:
        return PROFILES["unknown"]
    if status.get("simulation_enabled") or status.get("mockup_sim_enabled"):
        return PROFILES["sim"]
    if status.get("wireless_version"):
        return PROFILES["wireless"]
    if status.get("camera_specs_name") == "lite":
        return PROFILES["lite"]
    return PROFILES["unknown"]


class DaemonMonitor:
    """Polls the daemon for model, state and media status, and applies the
    media-release policy (pause the robot camera pipeline when unused)."""

    def __init__(self, policy: Callable[[], bool], base_url: str = DAEMON_URL, interval_s: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.interval_s = interval_s
        self.policy = policy
        self.profile = PROFILES["unknown"]
        self.status: dict | None = None
        self.reachable = False
        self.media_released = False
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._run, name="daemon-monitor", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def nudge(self) -> None:
        """Re-evaluate the media policy now (e.g. right after a camera switch)."""
        self._wake.set()

    def _request(self, path: str, method: str = "GET") -> dict:
        req = urllib.request.Request(self.base_url + path, method=method, data=b"" if method == "POST" else None)
        with urllib.request.urlopen(req, timeout=3.0) as res:
            return json.load(res)

    def refresh(self) -> None:
        try:
            status = self._request("/api/daemon/status")
        except Exception:  # noqa: BLE001 - daemon down or restarting
            status = None
        with self._lock:
            self.status = status
            self.reachable = status is not None
            if status is not None:
                self.profile = detect_profile(status)
                self.media_released = bool(status.get("media_released"))

    def set_media_released(self, released: bool) -> bool:
        if released and not self.profile.can_release_media:
            return False
        if released == self.media_released:
            return True
        try:
            self._request("/api/media/release" if released else "/api/media/acquire", method="POST")
            print(f"[daemon] robot media {'paused' if released else 'resumed'}")
        except Exception as e:  # noqa: BLE001
            print(f"[daemon] media {'release' if released else 'acquire'} failed: {e}", file=sys.stderr)
            return False
        self.refresh()
        return True

    def restart_daemon(self) -> bool:
        """Ask the daemon to restart its robot backend (recovers after the robot
        was power-cycled or its motors stopped responding)."""
        try:
            self._request("/api/daemon/restart", method="POST")
            print("[daemon] restart requested")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[daemon] restart failed: {e}", file=sys.stderr)
            return False

    def _run(self) -> None:
        while not self._stop.is_set():
            self.refresh()
            if self.reachable:
                want = bool(self.policy())
                if want != self.media_released:
                    self.set_media_released(want)
            self._wake.wait(self.interval_s)
            self._wake.clear()

    def snapshot(self) -> dict:
        with self._lock:
            status = self.status or {}
            return {
                "reachable": self.reachable,
                "state": status.get("state"),
                "error": status.get("error") if status.get("state") == "error" else None,
                "mediaReleased": self.media_released,
                "profile": asdict(self.profile),
            }


SOURCE_ACTIONS = {
    "webcam": {"label": "Use laptop camera", "source": "webcam"},
    "phone": {"label": "Use phone", "source": "phone"},
    "robot": {"label": "Use robot camera", "source": "robot"},
}
REJECT_TIPS = {
    "too far": ("hands_far", "Move a little closer so Reachy can see your hands clearly."),
    "cut off": ("hands_cut", "Keep your whole hand inside the picture."),
    "resting": ("hands_low", "Raise your hand above the dotted line to play."),
    "not facing": ("hands_facing", "Turn your palm toward the camera."),
}
DARK_BRIGHTNESS = 45.0


class Advisor:
    """Watches telemetry, notices what is going (or went) wrong, and suggests
    the simplest fix, most important first."""

    def __init__(self) -> None:
        self._since: dict[str, float] = {}
        self._rejects: deque[tuple[float, str | None]] = deque()
        self._drops: deque[tuple[float, int]] = deque()

    def _sustained(self, key: str, cond: bool, now: float, seconds: float) -> bool:
        if not cond:
            self._since.pop(key, None)
            return False
        return now - self._since.setdefault(key, now) >= seconds

    def tips(
        self,
        *,
        daemon: dict,
        camera: dict,
        robot: dict,
        proc_ms: float,
        hands: list[dict],
        settings: dict,
        sound_target: str | None,
        brightness: float | None,
    ) -> list[dict]:
        now = time.time()
        profile = daemon["profile"]
        source = camera["source"]
        live = camera["state"] == "live"
        fps = camera.get("fps") or 0.0
        faster = [SOURCE_ACTIONS[s] for s in ("webcam", "phone") if s != source]
        sounds_move = settings.get("sound") and settings.get("sound_output") == "auto"
        pause_note = "sounds switch to this computer" if sounds_move else "Reachy's speaker pauses too"
        pause_action = {"label": "Pause robot camera", "setting": {"free_usb": True}}
        can_pause = profile["can_release_media"] and source != "robot" and not settings.get("free_usb")
        engaged = any(h["engaged"] for h in hands)
        tips: list[dict] = []

        # --- robot connection ---
        daemon_error = (daemon.get("error") or "").lower()
        if self._sustained("daemon_down", not daemon["reachable"], now, 3.0):
            tips.append({
                "id": "daemon_down",
                "text": "The Reachy service isn't running on this computer. Start it with manage.ps1 start and this page reconnects on its own.",
                "actions": faster if source == "robot" else [],
            })
        elif self._sustained("motor", "motor" in daemon_error, now, 2.0):
            tips.append({
                "id": "motor_error",
                "text": "Reachy's motors aren't responding. Check that its power supply is plugged in and switched on, then tap Reconnect.",
                "actions": [{"label": "Reconnect robot", "robot": "restart"}],
            })
        elif self._sustained("offline", robot["state"] == "error", now, 4.0):
            tips.append({
                "id": "robot_offline",
                "text": "Can't reach the robot right now. It keeps retrying on its own"
                + (", and you can keep playing with another camera meanwhile." if source == "robot" else "."),
                "actions": (faster if source == "robot" else []) + [{"label": "Reconnect robot", "robot": "restart"}],
            })

        if self._sustained("asleep", bool(robot.get("sleeping")) and engaged, now, 1.5):
            tips.append({"id": "asleep", "text": "Reachy is asleep. Wake it up to play.", "actions": [{"label": "Wake Reachy", "robot": "wake"}]})

        # --- hand placement: only when hands are seen but all are ignored ---
        for h in hands:
            self._rejects.append((now, None if h["engaged"] else h["reject"]))
        while self._rejects and now - self._rejects[0][0] > 3.0:
            self._rejects.popleft()
        reasons = [r for _, r in self._rejects]
        if reasons and all(r is not None for r in reasons):
            top = max(set(reasons), key=reasons.count)
            if top in REJECT_TIPS and self._sustained("hands:" + top, True, now, 2.5):
                tip_id, text = REJECT_TIPS[top]
                tips.append({"id": tip_id, "text": text, "actions": []})
        else:
            for key in [k for k in self._since if k.startswith("hands:")]:
                self._since.pop(key, None)

        if self._sustained("dark", live and brightness is not None and brightness < DARK_BRIGHTNESS, now, 3.0):
            tips.append({
                "id": "dark",
                "text": "The picture is quite dark, so hands are harder to spot. Turning on a light or facing a window helps a lot.",
                "actions": [],
            })

        # --- better options available right now ---
        if self._sustained("phone_ready", bool(camera.get("phone_connected")) and source != "phone", now, 3.0):
            tips.append({"id": "phone_ready", "text": "Your phone camera is connected and ready.", "actions": [SOURCE_ACTIONS["phone"]]})

        if self._sustained("sound_none", bool(settings.get("sound")) and sound_target is None, now, 2.0):
            tips.append({
                "id": "sound_unavailable",
                "text": "Reachy's speaker isn't available right now. Play reaction sounds on this computer, or a Bluetooth speaker paired with it, instead?",
                "actions": [{"label": "Play on this computer", "setting": {"sound_output": "computer"}}],
            })

        # --- performance ---
        if source == "robot" and self._sustained("robot_slow", live and fps < 15.0, now, 5.0):
            if profile["key"] == "lite":
                text = (
                    "The robot camera runs at about 10 fps on Reachy Mini Lite, whose camera, speaker and motors share one USB 2.0 hub. "
                    "Your laptop or phone camera runs at up to 30 fps for snappier reactions."
                )
            else:
                text = f"The robot camera is running at {fps:.0f} fps. A laptop or phone camera can make reactions snappier."
            tips.append({"id": "robot_camera_slow", "text": text, "actions": faster})

        if source == "phone" and self._sustained("phone_slow", live and fps < 12.0, now, 4.0):
            tips.append({
                "id": "phone_slow",
                "text": f"Phone video is arriving slowly over Wi-Fi ({fps:.0f} fps). Moving closer to the router, or using the laptop camera, will help.",
                "actions": [SOURCE_ACTIONS["webcam"]],
            })

        if source == "phone" and self._sustained("phone_moving", live and bool(camera.get("phone_moving")), now, 3.0):
            tips.append({
                "id": "phone_moving",
                "text": "Your phone is moving, so waves and pushes are paused. Prop it against something steady for the best tracking.",
                "actions": [],
            })

        self._drops.append((now, int(robot.get("dropped") or 0)))
        while self._drops and now - self._drops[0][0] > 10.0:
            self._drops.popleft()
        if self._drops and self._drops[-1][1] - self._drops[0][1] >= 3:
            tips.append({
                "id": "reactions_late",
                "text": "A few reactions were skipped because video reached the robot too late."
                + (f" Pausing the robot's own camera frees Reachy Mini Lite's shared USB 2.0 connection for the motors ({pause_note})." if can_pause else ""),
                "actions": [pause_action] if can_pause else (faster if source == "robot" else []),
            })

        motor_slow = robot.get("motor_ms") is not None and robot["motor_ms"] > 250
        if self._sustained("busy", live and (proc_ms > 60.0 or motor_slow), now, 5.0):
            if can_pause:
                tips.append({
                    "id": "busy_pause",
                    "text": "Tracking is slowing down. Reachy Mini Lite's camera, speaker and motors share one USB 2.0 connection; "
                    f"pausing the robot camera while you use this one frees it up and saves processing power ({pause_note}).",
                    "actions": [pause_action],
                })
            else:
                tips.append({
                    "id": "busy",
                    "text": "This computer is busy, so hand tracking is running slower. Closing other apps can help.",
                    "actions": [],
                })
        return tips
