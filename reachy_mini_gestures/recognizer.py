"""Hand-shape recognition.

Per frame: MediaPipe's pretrained GestureRecognizer (landmarks + canned
gesture scores) -> joint-angle rules for poses the model has no label for
(e.g. pointing sideways) -> optional user "teach" samples that can confirm a
label -> engaged-hand filter that marks resting / side / background hands as
ignored so they never drive the robot.
"""

from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

SHAPES = ["point", "open_palm", "fist", "thumbs_up", "thumbs_down", "peace", "love", "ok", "heart", "gun"]

CANNED_TO_SHAPE = {
    "Pointing_Up": "point",
    "Open_Palm": "open_palm",
    "Closed_Fist": "fist",
    "Thumb_Up": "thumbs_up",
    "Thumb_Down": "thumbs_down",
    "Victory": "peace",
    "ILoveYou": "love",
}
CANNED_MIN_SCORE = {
    "point": 0.55,
    "open_palm": 0.6,
    "fist": 0.6,
    "thumbs_up": 0.6,
    "thumbs_down": 0.6,
    "peace": 0.6,
    "love": 0.6,
}
RULE_CONF = 0.7
RULE_OVERRIDE_BELOW = 0.8  # a clear rule result beats a less sure model label
TEACH_CONFIRM_CONF = 0.85
TEACH_ONLY_CONF = 0.7
TEACH_OVERRIDE_DIST = 0.55  # a taught example this close overrides the model...
TEACH_OVERRIDE_BELOW = 0.9  # ...unless the model is at least this sure
# A match this close, with every other taught gesture at least MARGIN times
# further away, wins regardless of the model (measured on real taught OK signs:
# nearest OK 0.19-0.53, nearest other gesture >= 0.95).
TEACH_STRONG_DIST = 0.5
TEACH_STRONG_MARGIN = 2.0
PROCESS_WIDTH = 640

# Shapes shown palm-forward. A hand resting flat on a desk also tends to read
# as open_palm, so these additionally require the palm to face the camera.
FACING_REQUIRED = {"open_palm", "peace", "love"}

FINGERS = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
PALM_POINTS = [0, 5, 9, 13, 17]


@dataclass
class EngageConfig:
    min_area: float = 0.015
    # Compact shapes (finger heart, fist) have a small bounding box even up
    # close, so palm size (wrist -> middle knuckle) also counts as "near enough".
    min_scale: float = 0.06
    zone_bottom: float = 0.12
    facing_max_deg: float = 60.0
    edge_margin: float = 0.02
    min_inside: int = 17
    min_handedness: float = 0.7


@dataclass(eq=False)  # identity equality: fields hold numpy arrays
class Hand:
    pts: np.ndarray  # (21, 3) normalized image coords
    world: np.ndarray  # (21, 3) metric, camera-aligned axes
    side: str
    shape: str | None
    conf: float
    source: str  # "model" | "rules" | "teach" | "none"
    center: tuple[float, float]
    tip: tuple[float, float]
    area: float
    scale: float  # wrist -> middle knuckle length in the image; shape-independent size
    engaged: bool
    reject: str | None
    side_score: float = 1.0  # handedness confidence; low for hands shown back-first


def _joint_angle(p: np.ndarray, a: int, b: int, c: int) -> float:
    v1 = p[a] - p[b]
    v2 = p[c] - p[b]
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _finger_state(world: np.ndarray, mcp: int, pip: int, dip: int, tip: int) -> str:
    pip_ang = _joint_angle(world, mcp, pip, dip)
    dip_ang = _joint_angle(world, pip, dip, tip)
    if pip_ang > 155 and dip_ang > 140:
        return "extended"
    if pip_ang < 120:
        return "curled"
    return "ambiguous"


def _thumb_extended(world: np.ndarray) -> bool:
    palm = float(np.linalg.norm(world[9] - world[0])) or 1e-6
    straight = _joint_angle(world, 1, 2, 3) > 150 and _joint_angle(world, 2, 3, 4) > 145
    away = float(np.linalg.norm(world[4] - world[5])) > 0.55 * palm
    return straight and away


def _folded(world: np.ndarray, mcp: int, pip: int, dip: int, tip: int) -> bool:
    """Tip pulled in closer to the wrist than the finger's middle joint. Holds
    for a curled finger even when it is hidden behind the back of the hand and
    its joint angles come out unreliable."""
    return float(np.linalg.norm(world[tip] - world[0])) < float(np.linalg.norm(world[pip] - world[0]))


def _long_index(world: np.ndarray) -> bool:
    """Index clearly straight and reaching much farther than the other fingers:
    a point in any direction, palm or back toward the camera."""
    reach = [float(np.linalg.norm(world[f[3]] - world[0])) for f in FINGERS]
    return _joint_angle(world, 5, 6, 7) > 150 and all(reach[0] >= 1.7 * r for r in reach[1:])


def rule_shape(pts: np.ndarray, world: np.ndarray) -> str | None:
    """Rotation-invariant shape rules from 3D joint angles. Returns None unless
    every finger is clearly extended or clearly folded, so half-relaxed hands
    never produce a label."""
    states = [_finger_state(world, *f) for f in FINGERS]
    palm = float(np.linalg.norm(world[9] - world[0])) or 1e-6
    # OK sign: thumb and index tips touching, the other three fingers up. The
    # closed circle keeps it apart from an open palm (index is not extended).
    # Middle/ring/pinky are often slightly bent in a relaxed OK, so "not curled"
    # with at least two clearly straight is enough.
    others = states[1:]
    circle = float(np.linalg.norm(world[4] - world[8])) < 0.45 * palm
    if circle and states[0] != "extended" and "curled" not in others and others.count("extended") >= 2:
        return "ok"
    # Korean finger heart: thumb tip crossed against a bent index finger (hand
    # tracking places it anywhere along the index's last two segments), middle/
    # ring/pinky folded. The index sticks out further than the folded fingers
    # (unlike a fist) but is bent (unlike a point with the thumb resting on it).
    def dist(a: int, b: int) -> float:
        return float(np.linalg.norm(world[a] - world[b]))

    crossed = min(dist(4, j) for j in (6, 7, 8)) < 0.35 * palm
    folded = all(s == "curled" or _folded(world, *f) for s, f in zip(others, FINGERS[1:]))
    index_out = dist(8, 0) > 1.1 * max(dist(12, 0), dist(16, 0), dist(20, 0))
    index_bent = _joint_angle(world, 5, 6, 7) < 150
    if crossed and folded and index_out and index_bent:
        return "heart"
    states = [
        "curled" if s == "ambiguous" and _folded(world, *f) else s
        for s, f in zip(states, FINGERS)
    ]
    if "ambiguous" in states:
        return "point" if _long_index(world) else None
    idx, mid, ring, pinky = (s == "extended" for s in states)
    thumb = _thumb_extended(world)

    if idx and not mid and not ring and not pinky:
        # Finger gun: thumb also straight and out, roughly at a right angle to the
        # index (an "L"). A pointing hand keeps its thumb tucked or alongside.
        if thumb:
            t_dir = world[4] - world[2]
            i_dir = world[8] - world[5]
            cos = float(np.dot(t_dir, i_dir) / (np.linalg.norm(t_dir) * np.linalg.norm(i_dir) + 1e-9))
            if math.degrees(math.acos(max(-1.0, min(1.0, cos)))) >= 50:
                return "gun"
        return "point"
    if idx and mid and not ring and not pinky:
        return "peace"
    if idx and not mid and not ring and pinky and thumb:
        return "love"
    if idx and mid and ring and pinky:
        return "open_palm"
    if not idx and not mid and not ring and not pinky:
        if not thumb:
            return "fist"
        dx = float(pts[4, 0] - pts[2, 0])
        dy = float(pts[4, 1] - pts[2, 1])
        palm_img = float(np.linalg.norm(pts[9, :2] - pts[0, :2])) or 1e-6
        if abs(dy) > 1.2 * abs(dx) and abs(dy) > 0.5 * palm_img:
            return "thumbs_up" if dy < 0 else "thumbs_down"
    return None


def palm_facing_deg(world: np.ndarray) -> float:
    """Angle between the palm plane normal and the camera axis (0 = palm or
    back of hand squarely facing the camera, 90 = edge-on)."""
    n = np.cross(world[5] - world[0], world[17] - world[0])
    norm = float(np.linalg.norm(n))
    if norm < 1e-9:
        return 90.0
    return math.degrees(math.acos(min(1.0, abs(float(n[2])) / norm)))


def assess_engagement(
    pts: np.ndarray, world: np.ndarray, side_score: float, shape: str | None, cfg: EngageConfig, scale: float = 0.0
) -> tuple[bool, str | None, float]:
    # side_score is not a gate: handedness confidence collapses for a hand shown
    # back-first (common when pointing), which is still a deliberate gesture.
    xs, ys = pts[:, 0], pts[:, 1]
    area = float((xs.max() - xs.min()) * (ys.max() - ys.min()))
    m = cfg.edge_margin
    inside = int(np.sum((xs >= m) & (xs <= 1 - m) & (ys >= m) & (ys <= 1 - m)))
    if inside < cfg.min_inside:
        return False, "cut off", area
    if area < cfg.min_area and scale < cfg.min_scale:
        return False, "too far", area
    if float(pts[0, 1]) > 1 - cfg.zone_bottom:
        return False, "resting", area
    if shape in FACING_REQUIRED and palm_facing_deg(world) > cfg.facing_max_deg:
        return False, "not facing", area
    # A hand someone is showing Reachy makes a shape; a relaxed hand on a lap
    # or armrest usually doesn't, so it shouldn't attract the robot's gaze.
    if shape is None:
        return False, "relaxed", area
    return True, None, area


def _canonical(world: np.ndarray, side: str) -> np.ndarray:
    """Hand-local frame: wrist origin, palm-size scale, wrist->middle MCP as +y,
    palm normal as +z. Left hands are mirrored before alignment so both hands
    share one template space."""
    w = np.asarray(world, dtype=np.float64).copy()
    if side == "Left":
        w[:, 0] *= -1
    w -= w[0]
    scale = float(np.linalg.norm(w[9])) or 1e-6
    w /= scale
    y = w[9] / (float(np.linalg.norm(w[9])) or 1e-6)
    n = np.cross(w[5], w[17])
    n -= y * float(np.dot(n, y))
    n /= float(np.linalg.norm(n)) or 1e-6
    x = np.cross(y, n)
    return (w @ np.stack([x, y, n]).T).flatten()


class TeachStore:
    """User-taught samples, persisted as raw world landmarks (not features) so
    the matching math can evolve without invalidating saved training."""

    def __init__(self, path: str, max_dist: float = 0.9, ratio: float = 0.8) -> None:
        self.path = path
        self.max_dist = max_dist
        self.ratio = ratio
        self._lock = threading.Lock()
        self._samples: dict[str, list[dict]] = {}
        self._features: dict[str, list[np.ndarray]] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            self._samples = data.get("samples", {})
        except (OSError, json.JSONDecodeError):
            self._samples = {}
        self._rebuild()

    def _rebuild(self) -> None:
        self._features = {
            label: [_canonical(np.array(s["world"]), s["side"]) for s in items]
            for label, items in self._samples.items()
        }

    def _save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"version": 2, "samples": self._samples}, f)
        os.replace(tmp, self.path)

    def add(self, label: str, world: np.ndarray, side: str) -> int:
        with self._lock:
            self._samples.setdefault(label, []).append({"world": np.asarray(world).tolist(), "side": side})
            self._features.setdefault(label, []).append(_canonical(world, side))
            self._save()
            return len(self._samples[label])

    def clear(self, label: str | None) -> None:
        with self._lock:
            if label is None:
                self._samples = {}
            else:
                self._samples.pop(label, None)
            self._rebuild()
            self._save()

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {label: len(items) for label, items in self._samples.items() if items}

    def match(self, world: np.ndarray, side: str) -> str | None:
        return self.match_scored(world, side)[0]

    def match_scored(self, world: np.ndarray, side: str) -> tuple[str | None, float, float]:
        """Best taught label, its distance (smaller = closer) and the distance
        to the nearest *other* label, or (None, inf, inf)."""
        with self._lock:
            if not self._features:
                return None, math.inf, math.inf
            vec = _canonical(world, side)
            best = {
                label: min(float(np.linalg.norm(vec - f)) for f in feats)
                for label, feats in self._features.items()
                if feats
            }
        if not best:
            return None, math.inf, math.inf
        ranked = sorted(best.items(), key=lambda kv: kv[1])
        label, d1 = ranked[0]
        d2 = ranked[1][1] if len(ranked) > 1 else math.inf
        if d1 > self.max_dist:
            return None, d1, d2
        if d1 > self.ratio * d2:
            return None, d1, d2
        return label, d1, d2


class HandRecognizer:
    def __init__(self, model_path: str, teach: TeachStore | None = None, cfg: EngageConfig | None = None) -> None:
        self.teach = teach
        self.cfg = cfg or EngageConfig()
        self._last_ts = -1
        self._rec = vision.GestureRecognizer.create_from_options(
            vision.GestureRecognizerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.5,
            )
        )

    def process(self, frame_bgr: np.ndarray, t_s: float) -> list[Hand]:
        h, w = frame_bgr.shape[:2]
        if w > PROCESS_WIDTH:
            frame_bgr = cv2.resize(frame_bgr, (PROCESS_WIDTH, int(h * PROCESS_WIDTH / w)), interpolation=cv2.INTER_AREA)
        aspect = frame_bgr.shape[1] / frame_bgr.shape[0]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ts = int(t_s * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        res = self._rec.recognize_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        hands: list[Hand] = []
        for i, lms in enumerate(res.hand_landmarks):
            pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float64)
            if res.hand_world_landmarks and i < len(res.hand_world_landmarks):
                world = np.array([[lm.x, lm.y, lm.z] for lm in res.hand_world_landmarks[i]], dtype=np.float64)
            else:
                world = pts
            hd = res.handedness[i][0] if res.handedness and i < len(res.handedness) and res.handedness[i] else None
            side = hd.category_name if hd else "Unknown"
            side_score = float(hd.score) if hd else 0.0

            shape, conf, source = None, 0.0, "none"
            canned = res.gestures[i][0] if res.gestures and i < len(res.gestures) and res.gestures[i] else None
            if canned is not None and canned.category_name in CANNED_TO_SHAPE:
                mapped = CANNED_TO_SHAPE[canned.category_name]
                if canned.score >= CANNED_MIN_SCORE[mapped]:
                    shape, conf, source = mapped, float(canned.score), "model"

            rule = rule_shape(pts, world)
            if shape is None and rule is not None:
                shape, conf, source = rule, RULE_CONF, "rules"
            elif shape is not None and rule is not None and rule != shape:
                # A sideways point with the thumb out looks like a thumb gesture to
                # the model; a clearly extended index finger settles it.
                gun = (rule == "point" and shape in ("thumbs_up", "thumbs_down")) or (rule == "ok" and shape == "open_palm") or (
                    (rule == "heart" and shape in ("fist", "thumbs_up", "point", "love"))
                    or (rule == "gun" and shape in ("point", "thumbs_up", "thumbs_down", "love"))
                )
                if gun or conf < RULE_OVERRIDE_BELOW:
                    shape, conf, source = rule, max(RULE_CONF, conf), "rules"
                else:
                    conf *= 0.8

            if self.teach is not None:
                taught, taught_dist, runner_up = self.teach.match_scored(world, side)
                # Very close to the user's own examples of one gesture and far from
                # every other taught gesture: trust the examples even over a
                # confident built-in label (e.g. OK signs the model calls Open_Palm).
                strong = (
                    taught is not None
                    and taught_dist < TEACH_STRONG_DIST
                    and runner_up > TEACH_STRONG_MARGIN * taught_dist
                )
                if taught is not None and taught == shape:
                    conf = max(conf, TEACH_CONFIRM_CONF)
                elif strong:
                    shape, conf, source = taught, TEACH_CONFIRM_CONF, "teach"
                elif taught is not None and shape is None:
                    shape, conf, source = taught, TEACH_ONLY_CONF, "teach"
                elif taught is not None and taught_dist < TEACH_OVERRIDE_DIST and conf < TEACH_OVERRIDE_BELOW:
                    # A close match to the user's own examples beats a less sure
                    # built-in label: that's what teaching a gesture is for.
                    shape, conf, source = taught, TEACH_CONFIRM_CONF, "teach"

            scale = math.hypot((pts[9, 0] - pts[0, 0]) * aspect, pts[9, 1] - pts[0, 1])
            engaged, reject, area = assess_engagement(pts, world, side_score, shape, self.cfg, scale)
            palm = pts[PALM_POINTS]
            hands.append(
                Hand(
                    pts=pts,
                    world=world,
                    side=side,
                    shape=shape,
                    conf=conf,
                    source=source,
                    center=(float(palm[:, 0].mean()), float(palm[:, 1].mean())),
                    tip=(float(pts[8, 0]), float(pts[8, 1])),
                    area=area,
                    scale=float(scale),
                    engaged=engaged,
                    reject=reject,
                    side_score=side_score,
                )
            )
        return hands
