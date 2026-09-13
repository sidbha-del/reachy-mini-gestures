"""Robot motion and connection management.

MotionEngine: a single 50Hz thread is the only writer to the motors. It blends
a continuous layer (head follow with the body turning along, antennas and
body following two hands) with short, preemptible whole-body keyframe clips,
and drops any intent whose camera frame is older than MAX_INTENT_AGE_S, so
reactions happen promptly or not at all.

RobotLink: owns the ReachyMini connection, reconnects with backoff, and runs
wake/sleep as exclusive operations.
"""

from __future__ import annotations

import collections
import os
import queue
import sys
import threading
import time
import traceback
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini.vision.look_at import look_at_image_pose

RATE_HZ = 50
MAX_INTENT_AGE_S = 1.0
FOLLOW_TIMEOUT_S = 0.3
PREEMPT_BLEND_S = 0.15
UNHEALTHY_ERRORS = 100
HEAD_FOLLOW_GAIN = 0.7
EXTERNAL_MAX_YAW = 45.0
EXTERNAL_MAX_PITCH = 22.0
# Head poses are in the world frame, so turning the body under a tracking head
# keeps the gaze on target while the whole robot visibly turns toward it.
BODY_SHARE_OF_HEAD_YAW = 0.6
BODY_FOLLOW_MAX = 30.0
# Two-pointer mode: sign that maps finger tilt onto antenna angle so each
# antenna leans the same way as its finger (-1 verified on Reachy Mini Lite).
ANTENNA_TILT_SIGN = -1.0
# Robot-camera self-motion: faster than this, or during/just after a clip, the
# image is moving because of Reachy itself, so motion gestures are unreliable.
EGO_BUSY_SPEED = 90.0  # deg/s (translation counted as 1 mm ~ 1 deg)
EGO_SETTLE_S = 0.3
FALLBACK_HFOV_DEG = 65.0

X, Y, Z, ROLL, PITCH, YAW, ANT_R, ANT_L, BODY = range(9)
CHANNELS = {"x": X, "y": Y, "z": Z, "roll": ROLL, "pitch": PITCH, "yaw": YAW, "ar": ANT_R, "al": ANT_L, "body": BODY}
LIMITS = np.array(
    [[-15, 15], [-15, 15], [-15, 25], [-20, 20], [-25, 25], [-60, 60], [-55, 55], [-55, 55], [-45, 45]],
    dtype=float,
)
# Antennas are small servos: fast full-range swings tripped an Overload Error
# (motor latches off until power-cycled), so they get a gentler limit.
MAX_SPEED = np.array([60, 60, 60, 240, 240, 300, 250, 250, 200], dtype=float)  # mm/s, deg/s


def _vec(**channels: float) -> np.ndarray:
    v = np.zeros(9)
    for key, value in channels.items():
        v[CHANNELS[key]] = value
    return v


def _turn(angle: float, **channels: float) -> dict:
    """Whole-body turn: body and head yaw together."""
    return {"body": angle, "yaw": angle, **channels}


def _clip(*keys: tuple[float, dict], sound: str | None = None) -> dict:
    return {"keys": [(t, _vec(**ch)) for t, ch in keys], "sound": sound}


CLIPS = {
    "ok": _clip(
        (0.0, {}),
        (0.25, {"roll": 14, "pitch": -6, "ar": -20, "al": 20}),
        (0.7, {"roll": 14, "pitch": -6, "ar": -20, "al": 20}), (1.0, {}),
    ),
    # Waving back: both antennas swing wide in opposite directions while the
    # head tilts and the body turns toward the person. Key spacing keeps the
    # antenna swing under its gentle speed limit so the full range is reached.
    "wave": _clip(
        (0.0, {}),
        (0.3, _turn(15, roll=12, ar=55, al=-55)), (0.6, _turn(15, roll=12, ar=-35, al=35)),
        (0.9, _turn(15, roll=12, ar=55, al=-55)), (1.2, _turn(10, roll=8, ar=-35, al=35)),
        (1.5, _turn(5, roll=4, ar=40, al=-40)), (1.8, {}),
    ),
    "open_palm": None,  # Hello and Wave are one gesture now; filled in below
    "fist": _clip(
        (0.0, {}),
        (0.3, _turn(22, pitch=18, z=-8, ar=-30, al=30)),
        (0.8, _turn(22, pitch=18, z=-8, ar=-30, al=30)), (1.15, {}),
    ),
    "thumbs_up": _clip(
        (0.0, {}),
        (0.15, _turn(8, pitch=-12, ar=30, al=-30)), (0.3, _turn(-8, pitch=10)),
        (0.45, _turn(8, pitch=-8, ar=30, al=-30)), (0.6, _turn(-4, pitch=6)), (0.85, {}),
    ),
    "thumbs_down": _clip(
        (0.0, {}),
        (0.25, {"body": 10, "yaw": 22, "pitch": 8, "ar": -35, "al": 35}),
        (0.55, {"body": -10, "yaw": -22, "pitch": 8, "ar": -35, "al": 35}),
        (0.85, {"body": 6, "yaw": 14, "pitch": 8, "ar": -35, "al": 35}), (1.2, {}),
        sound="confused1.wav",
    ),
    "peace": _clip(
        (0.0, {}), (0.2, _turn(25)), (0.45, _turn(-25)), (0.7, _turn(12)), (0.95, {}),
    ),
    "love": _clip(
        (0.0, {}),
        (0.2, _turn(20, roll=8, ar=40, al=-40)), (0.4, _turn(-20, roll=-8, ar=-40, al=40)),
        (0.6, _turn(20, roll=8, ar=40, al=-40)), (0.8, _turn(-20, roll=-8, ar=-40, al=40)), (1.1, {}),
        sound="dance1.wav",
    ),
    "raise_both": _clip(
        (0.0, {}),
        (0.3, _turn(10, z=18, pitch=-10, ar=45, al=-45)),
        (0.6, _turn(-10, z=18, pitch=-10, ar=45, al=-45)), (0.8, _turn(0, z=18, pitch=-10, ar=45, al=-45)), (1.1, {}),
    ),
    "ta_da": _clip(
        (0.0, {}),
        (0.15, _turn(30, z=12, ar=55, al=-55)), (0.35, _turn(-30, z=-4, ar=-20, al=20)),
        (0.55, _turn(15, z=8, roll=10, ar=50, al=-50)), (0.75, _turn(0, roll=-10, ar=50, al=-50)), (1.0, {}),
        sound="dance1.wav",
    ),
    # Too close: a startled jolt back, then "no no no, don't come close!" with
    # the antennas wiggling and small head shakes in time with the uh-uh-uh sound.
    "approach": _clip(
        (0.0, {}),
        (0.12, _turn(-12, x=-10, pitch=-10, ar=40, al=-40)),
        (0.36, _turn(-12, x=-10, pitch=-10, yaw=-4, ar=-25, al=25)),
        (0.6, _turn(-12, x=-10, pitch=-10, yaw=-20, ar=40, al=-40)),
        (0.84, _turn(-12, x=-10, pitch=-10, yaw=-4, ar=-25, al=25)),
        (1.08, _turn(-12, x=-10, pitch=-10, yaw=-20, ar=35, al=-35)),
        (1.45, {}),
        sound="no_no_no.wav",
    ),
}


CLIPS["open_palm"] = CLIPS["wave"]


def _minjerk(u: float) -> float:
    return u * u * u * (10 - 15 * u + 6 * u * u)


def clip_offset(keys: list[tuple[float, np.ndarray]], t: float) -> np.ndarray | None:
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            u = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
            return v0 + (v1 - v0) * _minjerk(u)
    return None


APP_SOUNDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")


def _sound_path(filename: str) -> str | None:
    """App sounds first (e.g. no_no_no.wav), then the SDK's built-in ones."""
    import reachy_mini

    for folder in (APP_SOUNDS_DIR, os.path.join(os.path.dirname(reachy_mini.__file__), "assets")):
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            return path
    return None


SOUND_FILES = sorted({c["sound"] for c in CLIPS.values() if c["sound"]})


def clip_sound(name: str) -> str | None:
    clip = CLIPS.get(name)
    return clip["sound"] if clip else None


def sound_file_path(filename: str) -> str | None:
    """Path of a reaction sound, restricted to the files clips actually use."""
    return _sound_path(filename) if filename in SOUND_FILES else None


class Ego:
    """How the robot camera itself moved at a frame's capture time.
    du/dv: image shift (normalized) a static point gets from the current head
    yaw/pitch; subtract it to get a world-stable position."""

    __slots__ = ("du", "dv", "speed", "busy")

    def __init__(self, du: float = 0.0, dv: float = 0.0, speed: float = 0.0, busy: bool = False) -> None:
        self.du, self.dv, self.speed, self.busy = du, dv, speed, busy


class MotionEngine:
    def __init__(self, mini: ReachyMini, sound_enabled: Callable[[], bool] = lambda: True) -> None:
        self.mini = mini
        self.sound_enabled = sound_enabled
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.base = np.zeros(9)
        self.commanded = np.zeros(9)
        self._head_target: tuple[np.ndarray, float, float] | None = None
        self._ant_target: tuple[np.ndarray, float] | None = None
        self._body_target: tuple[float, float] | None = None
        self._clip: tuple[str, list, float] | None = None
        self._preempt = np.zeros(9)
        self._preempt_t = -1e9
        self._pending_capture_t: float | None = None
        self._history: collections.deque[tuple[float, np.ndarray]] = collections.deque(maxlen=RATE_HZ * 2)
        self._busy_spans: collections.deque[list[float]] = collections.deque(maxlen=16)
        self._last_sent: np.ndarray | None = None
        self._last_sent_t = 0.0
        self.paused = True
        self.sleeping = False
        self.backoff_until = 0.0
        self.consecutive_errors = 0
        self.fired = 0
        self.dropped_stale = 0
        self.motor_latency_ms: float | None = None
        self.last_error: str | None = None

    # ---- lifecycle ----
    def start(self) -> None:
        self.resync()
        self._thread = threading.Thread(target=self._run, name="motion-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    @property
    def healthy(self) -> bool:
        return self.consecutive_errors < UNHEALTHY_ERRORS

    @property
    def action(self) -> str | None:
        with self._lock:
            return self._clip[0] if self._clip else None

    def resync(self) -> None:
        """Adopt the robot's actual pose as the commanded pose so nothing snaps."""
        vec = np.zeros(9)
        try:
            pose = self.mini.get_current_head_pose()
            vec[[ROLL, PITCH, YAW]] = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
            vec[[X, Y, Z]] = pose[:3, 3] * 1000
            antennas = self.mini.get_present_antenna_joint_positions()
            vec[[ANT_R, ANT_L]] = np.rad2deg(antennas[:2])
        except Exception as e:  # noqa: BLE001 - fall back to neutral
            print(f"[motion] resync failed, assuming neutral: {e}", file=sys.stderr)
        with self._lock:
            self.base = vec
            self.commanded = vec.copy()
            self._clip = None
            self._head_target = None
            self._ant_target = None
            self._body_target = None
            self._preempt[:] = 0
            self._history.clear()
            self._last_sent = None

    def run_exclusive(self, fn: Callable[[], None], sleeping_after: bool) -> None:
        """Run a blocking SDK motion (wake_up / goto_sleep) with the engine
        paused, then adopt the resulting pose."""
        self.paused = True
        time.sleep(2.0 / RATE_HZ)
        span = [time.time(), float("inf")]
        with self._lock:
            self._busy_spans.append(span)
        try:
            fn()
        finally:
            self.sleeping = sleeping_after
            self.resync()
            span[1] = time.time()
            self.paused = False

    # ---- intents (called from the vision thread) ----
    def follow_head(self, u: float, v: float, source: str, t_capture: float, gain: float = 1.0) -> None:
        """gain 1.0 = look right at it (pointing); lower = a softer, partial glance."""
        now = time.time()
        if now - t_capture > MAX_INTENT_AGE_S or self.sleeping:
            return
        u = min(0.98, max(0.02, u))
        v = min(0.98, max(0.02, v))
        rpy = None
        if source == "robot":
            try:
                cam = self.mini.media.camera
                width, height = cam.resolution
                past = self._pose_at(t_capture)
                t_world_head = create_head_pose(
                    x=past[X], y=past[Y], z=past[Z], roll=past[ROLL], pitch=past[PITCH], yaw=past[YAW], mm=True
                )
                target = look_at_image_pose(
                    u * width, v * height, cam.K, cam.D, T_world_head=t_world_head, T_head_cam=self.mini.T_head_cam
                )
                aim = Rotation.from_matrix(target[:3, :3]).as_euler("xyz", degrees=True)
                current = past[[ROLL, PITCH, YAW]]
                rpy = gain * (current + HEAD_FOLLOW_GAIN * (aim - current))
                rpy[0] = 0.0
            except Exception:  # noqa: BLE001 - camera specs unavailable, use proportional mapping
                rpy = None
        if rpy is None:
            rpy = gain * np.array([0.0, (v - 0.5) * 2 * EXTERNAL_MAX_PITCH, (0.5 - u) * 2 * EXTERNAL_MAX_YAW])
        with self._lock:
            self._head_target = (np.asarray(rpy, dtype=float), now, gain)

    def follow_antennas(self, left_y: float, right_y: float, t_capture: float, center_x: float | None = None) -> None:
        now = time.time()
        if now - t_capture > MAX_INTENT_AGE_S or self.sleeping:
            return

        def level(y: float) -> float:
            return float(np.interp(y, [0.15, 0.85], [60.0, -60.0]))

        with self._lock:
            self._ant_target = (np.array([level(right_y), -level(left_y)]), now)
            if center_x is not None:
                self._body_target = ((0.5 - center_x) * 2 * BODY_FOLLOW_MAX, now)

    def follow_antenna_angles(
        self, left_deg: float, right_deg: float, t_capture: float, center_x: float | None = None
    ) -> None:
        """Antennas copy the tilt of each pointing finger (0 = finger straight
        up). left/right are the hands on the left/right of the camera image."""
        now = time.time()
        if now - t_capture > MAX_INTENT_AGE_S or self.sleeping:
            return
        target = ANTENNA_TILT_SIGN * np.array([right_deg, left_deg], dtype=float)
        with self._lock:
            self._ant_target = (np.clip(target, LIMITS[ANT_R, 0], LIMITS[ANT_R, 1]), now)
            if center_x is not None:
                self._body_target = ((0.5 - center_x) * 2 * BODY_FOLLOW_MAX, now)

    def trigger(self, name: str, t_capture: float) -> bool:
        now = time.time()
        clip = CLIPS.get(name)
        if clip is None or self.sleeping or self.paused:
            return False
        if now - t_capture > MAX_INTENT_AGE_S:
            self.dropped_stale += 1
            return False
        with self._lock:
            self._preempt = self._offset(now)
            self._preempt_t = now
            self._clip = (name, clip["keys"], now)
            self._pending_capture_t = t_capture
            if self._busy_spans and self._busy_spans[-1][1] > now:
                self._busy_spans[-1][1] = now
            self._busy_spans.append([now, now + clip["keys"][-1][0]])
        self.fired += 1
        if clip["sound"] and self.sound_enabled():
            threading.Thread(target=self._play_sound, args=(clip["sound"],), daemon=True).start()
        return True

    def _play_sound(self, filename: str) -> None:
        path = _sound_path(filename)
        if path is None:
            return
        try:
            self.mini.media.play_sound(path)
        except Exception as e:  # noqa: BLE001
            print(f"[motion] sound failed: {e}", file=sys.stderr)

    def ego_at(self, t: float) -> Ego:
        """Robot-camera self-motion at capture time t (see Ego)."""
        pose = self._pose_at(t)
        before = self._pose_at(t - 0.06)
        after = self._pose_at(t + 0.06)
        with self._lock:
            newest = self._history[-1][0] if self._history else t
            busy = any(start <= t <= end + EGO_SETTLE_S for start, end in self._busy_spans)
        span = max(1e-3, min(t + 0.06, newest) - (t - 0.06))
        delta = np.abs(after - before)[[X, Y, Z, ROLL, PITCH, YAW]]
        speed = float(np.max(delta)) / span
        width_px, fx, fy, height_px = None, None, None, None
        try:
            cam = self.mini.media.camera
            width_px, height_px = cam.resolution
            fx, fy = float(cam.K[0, 0]), float(cam.K[1, 1])
        except Exception:  # noqa: BLE001 - no specs, assume a typical wide camera
            pass
        if not (width_px and fx):
            k_u = 1.0 / np.deg2rad(FALLBACK_HFOV_DEG)
            k_v = k_u * 16 / 9
        else:
            k_u, k_v = fx / width_px, fy / height_px
        # Positive yaw turns left, so the scene slides right (+u); positive pitch
        # looks down, so the scene slides up (-v).
        du = float(np.deg2rad(pose[YAW]) * k_u)
        dv = float(-np.deg2rad(pose[PITCH]) * k_v)
        return Ego(du, dv, speed, busy or speed > EGO_BUSY_SPEED)

    # ---- internals ----
    def _pose_at(self, t: float) -> np.ndarray:
        with self._lock:
            best = self.commanded
            for ts, vec in reversed(self._history):
                best = vec
                if ts <= t:
                    break
            return best.copy()

    def _offset(self, now: float) -> np.ndarray:
        off = np.zeros(9)
        if self._clip is not None:
            _, keys, start = self._clip
            o = clip_offset(keys, now - start)
            if o is None:
                self._clip = None
            else:
                off += o
        fade = 1.0 - (now - self._preempt_t) / PREEMPT_BLEND_S
        if fade > 0:
            off += self._preempt * fade
        return off

    def _tick(self, now: float, dt: float) -> tuple[np.ndarray, float | None]:
        with self._lock:
            target = np.zeros(9)
            tau = np.full(9, 0.35)
            head_active = self._head_target is not None and now - self._head_target[1] <= FOLLOW_TIMEOUT_S
            if head_active:
                rpy = self._head_target[0]
                target[[ROLL, PITCH, YAW]] = rpy
                target[BODY] = BODY_SHARE_OF_HEAD_YAW * rpy[2]
                precise = self._head_target[2] >= 0.99
                tau[[ROLL, PITCH, YAW]] = 0.08 if precise else 0.18
                tau[BODY] = 0.2 if precise else 0.35
            if self._ant_target is not None and now - self._ant_target[1] <= FOLLOW_TIMEOUT_S:
                target[[ANT_R, ANT_L]] = self._ant_target[0]
                tau[[ANT_R, ANT_L]] = 0.07
            if not head_active and self._body_target is not None and now - self._body_target[1] <= FOLLOW_TIMEOUT_S:
                target[BODY] = target[YAW] = self._body_target[0]
                tau[BODY] = tau[YAW] = 0.25
            step = (target - self.base) * (1.0 - np.exp(-dt / tau))
            max_step = MAX_SPEED * dt
            self.base += np.clip(step, -max_step, max_step)
            out = np.clip(self.base + self._offset(now), LIMITS[:, 0], LIMITS[:, 1])
            self.commanded = out
            self._history.append((now, out.copy()))
            capture_t, self._pending_capture_t = self._pending_capture_t, None
        return out, capture_t

    def _send(self, out: np.ndarray) -> None:
        head = create_head_pose(
            x=out[X], y=out[Y], z=out[Z], roll=out[ROLL], pitch=out[PITCH], yaw=out[YAW], mm=True
        )
        self.mini.set_target(head=head, antennas=np.deg2rad([out[ANT_R], out[ANT_L]]), body_yaw=float(np.deg2rad(out[BODY])))

    def _run(self) -> None:
        period = 1.0 / RATE_HZ
        last = time.time()
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            now = time.time()
            dt = min(0.1, max(1e-3, now - last))
            last = now
            if not self.paused and not self.sleeping and now >= self.backoff_until:
                out, capture_t = self._tick(now, dt)
                changed = self._last_sent is None or float(np.max(np.abs(out - self._last_sent))) > 0.05
                if changed or now - self._last_sent_t > 0.5:
                    try:
                        self._send(out)
                        self._last_sent = out
                        self._last_sent_t = now
                        self.consecutive_errors = 0
                        if capture_t is not None:
                            self.motor_latency_ms = (time.time() - capture_t) * 1000
                    except Exception as e:  # noqa: BLE001 - keep ticking, back off on bus errors
                        self.consecutive_errors += 1
                        self.last_error = f"{type(e).__name__}: {e}"
                        lowered = str(e).lower()
                        if "communication error" in lowered or "no response" in lowered:
                            self.backoff_until = now + 3.0
            next_tick += period
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()


class RobotLink:
    def __init__(self, camera, sound_enabled: Callable[[], bool], stop_event: threading.Event) -> None:
        self.camera = camera
        self.sound_enabled = sound_enabled
        self.stop_event = stop_event
        self.state = "connecting"
        self.error: str | None = None
        self.engine: MotionEngine | None = None
        self._commands: queue.Queue[str] = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="robot-link", daemon=True)
        self._thread.start()

    def join(self, timeout: float) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)

    def command(self, name: str) -> None:
        self._commands.put(name)

    def status(self) -> dict:
        engine = self.engine
        return {
            "state": self.state,
            "error": self.error,
            "sleeping": bool(engine and engine.sleeping),
            "action": engine.action if engine else None,
            "motor_ms": round(engine.motor_latency_ms) if engine and engine.motor_latency_ms is not None else None,
            "fired": engine.fired if engine else 0,
            "dropped": engine.dropped_stale if engine else 0,
        }

    def _run(self) -> None:
        delay = 2.0
        while not self.stop_event.is_set():
            self.state = "connecting"
            try:
                # "default" lets the SDK pick local media for a USB robot and
                # WebRTC for a wireless one, so the same app runs on both.
                with ReachyMini(media_backend="default") as mini:
                    engine = MotionEngine(mini, self.sound_enabled)
                    try:
                        self.camera.attach(mini)
                        engine.start()
                        self.engine = engine
                        while not self._commands.empty():
                            self._commands.get_nowait()
                        engine.run_exclusive(mini.wake_up, sleeping_after=False)
                        self.state = "connected"
                        self.error = None
                        delay = 2.0
                        print("[robot] connected")
                        while not self.stop_event.is_set():
                            try:
                                cmd = self._commands.get(timeout=0.1)
                            except queue.Empty:
                                cmd = None
                            if cmd == "wake":
                                engine.run_exclusive(mini.wake_up, sleeping_after=False)
                            elif cmd == "sleep":
                                engine.run_exclusive(mini.goto_sleep, sleeping_after=True)
                            elif cmd == "reconnect":
                                raise ConnectionError("reconnecting to refresh the robot session")
                            if not engine.healthy:
                                raise ConnectionError(f"robot stopped responding ({engine.last_error})")
                        print("[robot] stop requested, going to sleep")
                        engine.run_exclusive(mini.goto_sleep, sleeping_after=True)
                        return
                    finally:
                        engine.stop()
                        self.engine = None
                        self.camera.detach()
            except Exception as e:  # noqa: BLE001 - reconnect on any connection-level failure
                self.state = "error"
                self.error = f"{type(e).__name__}: {e}"
                print(f"[robot] {self.error}; retrying in {delay:.0f}s", file=sys.stderr)
                traceback.print_exc()
                self.stop_event.wait(delay)
                delay = min(delay * 1.5, 15.0)
