"""Camera sources. Each keeps only its newest frame (with capture time), so
recognition never works on stale or buffered video."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

SOURCES = ("robot", "webcam", "phone")
STALE_S = 1.0
PHONE_CONNECTED_S = 2.0
DISPLAY_WIDTH = 960


@dataclass
class Frame:
    image: np.ndarray | None
    t: float
    seq: int
    jpeg: bytes | None = None


class _Slot:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Frame | None = None
        self._seq = 0
        self.fps = 0.0

    def put(self, image: np.ndarray | None, t: float, jpeg: bytes | None = None) -> None:
        with self._lock:
            if self._frame is not None:
                dt = t - self._frame.t
                if dt > 0:
                    self.fps = 0.8 * self.fps + 0.2 * (1.0 / dt) if self.fps else 1.0 / dt
            self._seq += 1
            self._frame = Frame(image, t, self._seq, jpeg)

    def get(self) -> Frame | None:
        with self._lock:
            return self._frame

    def age(self) -> float:
        with self._lock:
            return time.time() - self._frame.t if self._frame else float("inf")

    def clear(self) -> None:
        with self._lock:
            self._frame = None
            self.fps = 0.0


class RobotCamera:
    name = "robot"

    def __init__(self) -> None:
        self.slot = _Slot()
        self.error: str | None = None
        self._mini = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def attach(self, mini) -> None:
        self._mini = mini

    def detach(self) -> None:
        self._mini = None
        self.slot.clear()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="robot-camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self.slot.clear()

    def _run(self) -> None:
        last_sig = None
        while not self._stop.is_set():
            mini = self._mini
            if mini is None:
                time.sleep(0.05)
                continue
            try:
                frame = mini.media.get_frame()
            except Exception as e:  # noqa: BLE001 - connection hiccup; robot link handles reconnects
                self.error = str(e)
                time.sleep(0.05)
                continue
            if frame is None:
                time.sleep(0.005)
                continue
            sig = frame[::97, ::97].tobytes()
            if sig == last_sig:
                time.sleep(0.004)
                continue
            last_sig = sig
            self.error = None
            self.slot.put(frame.copy(), time.time())


class Webcam:
    name = "webcam"

    def __init__(self, index: int = 0) -> None:
        self.index = index
        self.slot = _Slot()
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="webcam", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.slot.clear()

    def _open(self) -> cv2.VideoCapture | None:
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self) -> None:
        cap = None
        try:
            while not self._stop.is_set():
                if cap is None:
                    cap = self._open()
                    if cap is None:
                        self.error = "Laptop webcam not available"
                        self._stop.wait(2.0)
                        continue
                    self.error = None
                ok, frame = cap.read()
                if not ok:
                    self.error = "Webcam stopped delivering frames"
                    cap.release()
                    cap = None
                    self._stop.wait(1.0)
                    continue
                self.slot.put(frame, time.time())
        finally:
            if cap is not None:
                cap.release()


PHONE_SHAKE_DPS = 35.0
PHONE_SHAKE_SETTLE_S = 0.4


class PhoneCamera:
    name = "phone"

    def __init__(self) -> None:
        self.slot = _Slot()
        self.error: str | None = None
        self.clients = 0
        self._decode_lock = threading.Lock()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def push(self, jpeg: bytes) -> None:
        self.slot.put(None, time.time(), jpeg)

    def set_motion(self, dps: float) -> None:
        """Phone rotation speed reported by its motion sensor (deg/s)."""
        self.motion_dps, self.motion_t = float(dps), time.time()
        if dps > PHONE_SHAKE_DPS:
            self.shake_until = self.motion_t + PHONE_SHAKE_SETTLE_S

    def shaking(self, t: float | None = None) -> bool:
        """True while (or just after) the hand-held phone was moving."""
        return (t if t is not None else time.time()) <= getattr(self, "shake_until", 0.0)

    def connected(self) -> bool:
        return self.clients > 0 and self.slot.age() < PHONE_CONNECTED_S

    def decoded(self, frame: Frame) -> Frame | None:
        with self._decode_lock:
            if frame.image is None:
                image = cv2.imdecode(np.frombuffer(frame.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    return None
                frame.image = image
            return frame


def encode_display(frame: Frame) -> bytes | None:
    """JPEG for the live view. Phone frames are already JPEG; others are
    downscaled and encoded once per frame."""
    if frame.jpeg is not None:
        return frame.jpeg
    image = frame.image
    if image is None:
        return None
    h, w = image.shape[:2]
    if w > DISPLAY_WIDTH:
        image = cv2.resize(image, (DISPLAY_WIDTH, int(h * DISPLAY_WIDTH / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return None
    frame.jpeg = buf.tobytes()
    return frame.jpeg


class CameraHub:
    def __init__(self, persist_path: str) -> None:
        self.persist_path = persist_path
        self.robot = RobotCamera()
        self.webcam = Webcam()
        self.phone = PhoneCamera()
        self._sources = {"robot": self.robot, "webcam": self.webcam, "phone": self.phone}
        self._lock = threading.Lock()
        self.selected = self._load()
        self._sources[self.selected].start()

    def _load(self) -> str:
        try:
            with open(self.persist_path) as f:
                source = json.load(f).get("source")
            if source in SOURCES:
                return source
        except (OSError, json.JSONDecodeError):
            pass
        return "robot"

    def select(self, name: str) -> None:
        if name not in SOURCES:
            raise ValueError(f"unknown camera source '{name}'")
        with self._lock:
            if name == self.selected:
                return
            self._sources[self.selected].stop()
            self.selected = name
            self._sources[name].start()
        try:
            with open(self.persist_path, "w") as f:
                json.dump({"source": name}, f)
        except OSError:
            pass

    def latest(self) -> Frame | None:
        src = self._sources[self.selected]
        frame = src.slot.get()
        if frame is None or time.time() - frame.t > STALE_S:
            return None
        if src is self.phone:
            return self.phone.decoded(frame)
        return frame

    def status(self) -> dict:
        src = self._sources[self.selected]
        live = src.slot.age() < STALE_S
        if live:
            state = "live"
        elif src is self.phone:
            state = "waiting_for_phone"
        elif src.error:
            state = "error"
        else:
            state = "connecting"
        return {
            "source": self.selected,
            "state": state,
            "fps": round(src.slot.fps, 1) if live else 0.0,
            "error": src.error,
            "phone_connected": self.phone.connected(),
            "phone_moving": self.phone.shaking(),
        }
