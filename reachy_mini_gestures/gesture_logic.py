"""Time-based gesture state machine.

Turns per-frame hand shapes into robot intents:
- per-hand tracks with confidence smoothing and onset/release hysteresis, so a
  single bad frame neither triggers nor cancels anything;
- discrete reactions fire once per hold, after a short hold that lets
  transitional shapes (fist on the way to thumbs up) pass, and re-arm only
  after release;
- continuous modes (head follow, antennas) latch while the pose is held;
- motion gestures (waving hello, hands up, ta-da, approach) from a short history.
With the robot camera, positions are stabilized against the head's own
rotation and motion gestures are refused while Reachy itself is moving, so the
robot never reacts to its own movement.
All timing is in seconds, so behavior is the same at 10fps and 30fps.
"""

from __future__ import annotations

import collections
import itertools
import math
from collections import deque
from dataclasses import dataclass, field

from .recognizer import SHAPES, Hand

# emoji: compact icon (pill, toast). steps: how to make it, shown on the card as
# icons joined by arrows. hands: how many hands the gesture needs.
GESTURES = [
    {"key": "point", "emoji": "☝️", "steps": ["☝️"], "hands": 1, "name": "Point", "reaction": "Head follows your finger", "kind": "continuous", "teach": "point", "hint": "Point one finger at something, palm or back of hand"},
    {"key": "point_both", "emoji": "2x☝️", "steps": ["2x☝️"], "hands": 2, "name": "Two pointers", "reaction": "Antennas copy your finger angles", "kind": "continuous", "teach": None, "hint": "Point up with both hands, then tilt your fingers"},
    {"key": "open_palm", "emoji": "\U0001f44b", "steps": ["✋", "\U0001f44b"], "hands": 1, "name": "Hello", "reaction": "Waves back", "kind": "pose", "teach": "open_palm", "hint": "Show an open palm, still or waving"},
    {"key": "fist", "emoji": "✊", "steps": ["✊"], "hands": 1, "name": "Fist", "reaction": "Gets shy", "kind": "pose", "teach": "fist", "hint": "Make a fist"},
    {"key": "thumbs_up", "emoji": "\U0001f44d", "steps": ["\U0001f44d"], "hands": 1, "name": "Thumbs up", "reaction": "Happy nod", "kind": "pose", "teach": "thumbs_up", "hint": "Thumb up"},
    {"key": "thumbs_down", "emoji": "\U0001f44e", "steps": ["\U0001f44e"], "hands": 1, "name": "Thumbs down", "reaction": "Sad head shake", "kind": "pose", "teach": "thumbs_down", "hint": "Thumb down"},
    {"key": "peace", "emoji": "✌️", "steps": ["✌️"], "hands": 1, "name": "Peace", "reaction": "Body wiggle", "kind": "pose", "teach": "peace", "hint": "Index and middle finger up"},
    {"key": "love", "emoji": "\U0001f91f", "steps": ["\U0001f91f", "or", "\U0001faf0"], "hands": 1, "name": "Love you", "reaction": "Little dance", "kind": "pose", "teach": "love", "hint": "Thumb, index and pinky out, or a finger heart"},
    # Finger heart is a second way to show Love you: its own hand shape (so it can
    # be recognized and taught) but no card of its own, and it fires as "love".
    {"key": "heart", "emoji": "\U0001faf0", "steps": ["\U0001faf0"], "hands": 1, "name": "Finger heart", "reaction": "Little dance", "kind": "pose", "teach": "heart", "hidden": True, "hint": "Cross thumb and index finger into a small heart, other fingers folded"},
    {"key": "ok", "emoji": "\U0001f44c", "steps": ["\U0001f44c"], "hands": 1, "name": "Okay", "reaction": "Curious head tilt", "kind": "pose", "teach": "ok", "hint": "Touch thumb and index tip in a circle, other fingers up"},
    {"key": "gun", "emoji": "\U0001f449", "steps": ["\U0001f449", "\U0001f4a5"], "hands": 1, "name": "Bang!", "reaction": "Falls down dead, then comes back", "kind": "pose", "teach": "gun", "hint": "Finger gun: index out, thumb up, other fingers folded"},
    {"key": "raise_both", "emoji": "\U0001f64c", "steps": ["2x✋", "⬆️"], "hands": 2, "name": "Hands up", "reaction": "Stretches up tall", "kind": "motion", "teach": None, "hint": "Show both open palms, then raise them"},
    {"key": "ta_da", "emoji": "2x✋", "steps": ["2x✊", "2x✋"], "hands": 2, "name": "Ta-da", "reaction": "Surprise pop", "kind": "motion", "teach": None, "hint": "Two fists, then open both hands"},
    {"key": "namaste", "emoji": "\U0001f64f", "steps": ["\U0001f64f"], "hands": 2, "name": "Namaste", "reaction": "Bows with antennas together", "kind": "pose", "teach": None, "hint": "Press your palms together in front of your chest, fingers up"},
    {"key": "approach", "emoji": "✋", "steps": ["✋", "\U0001f4f7"], "hands": 1, "name": "Too close", "reaction": "Startled: “no no no, don't come close!”", "kind": "motion", "teach": None, "hint": "Hold your hand right in front of the camera, or push a palm toward it"},
]

# train: the hand shapes Teach captures for a gesture, in order (two-hand
# gestures capture both hands at each step).
_TRAIN = {"point_both": ["point"], "raise_both": ["open_palm"], "ta_da": ["fist", "open_palm"], "approach": ["open_palm"], "love": ["love", "heart"], "heart": []}
# Shapes that trigger another gesture's reaction.
FIRE_AS = {"heart": "love"}
for _g in GESTURES:
    _g["train"] = _TRAIN.get(_g["key"], [_g["teach"]] if _g["teach"] else [])

POSE_EVENTS = {"open_palm", "fist", "thumbs_up", "thumbs_down", "peace", "love", "ok", "heart", "gun"}


@dataclass
class LogicConfig:
    onset_conf: float = 0.65
    release_conf: float = 0.35
    onset_s: float = 0.15
    ema_tau_s: float = 0.08
    lost_s: float = 0.25
    min_track_s: float = 0.12
    pair_scale_ratio: float = 0.5
    match_dist: float = 0.25
    history_s: float = 1.2
    pose_hold_s: float = 0.3
    palm_still_s: float = 0.35
    palm_still_move: float = 0.05
    wave_window_s: float = 1.0
    wave_amp: float = 0.035
    raise_window_s: float = 1.0
    raise_dist: float = 0.10
    tada_window_s: float = 1.5
    approach_window_s: float = 0.5
    approach_growth: float = 1.3
    approach_min_scale: float = 0.16
    approach_min_samples: int = 3
    approach_min_span_s: float = 0.2
    approach_max_move: float = 0.08
    approach_max_dip: float = 0.95
    approach_baseline_s: float = 0.2
    # "Too close": a hand filling much of the view is Come close, not Hello.
    close_scale: float = 0.3
    close_area: float = 0.22
    close_hold_s: float = 0.25
    # Namaste: two hands pressed together (palm centers within this many palm
    # sizes of each other), fingertips above the wrists, held this long.
    namaste_gap: float = 1.1
    namaste_hold_s: float = 0.5
    refire_gap_s: float = 0.6
    predict_s: float = 0.08

    def set_sensitivity(self, value: float) -> None:
        """0 = strict, 1 = eager. Moves the onset threshold and hold times."""
        v = max(0.0, min(1.0, value))
        self.onset_conf = 0.8 - 0.3 * v
        self.onset_s = 0.25 - 0.17 * v
        self.pose_hold_s = 0.4 - 0.18 * v


@dataclass
class Sample:
    t: float
    cx: float  # world-stable center (image center minus robot-camera rotation)
    cy: float
    scale: float
    stable: str | None
    engaged: bool = True
    busy: bool = False  # the robot camera itself was moving
    fistish: bool = False  # looked like a fist (for ta-da's first half)


class Track:
    def __init__(self, tid: int, hand: Hand, t: float) -> None:
        self.id = tid
        self.side = hand.side
        self.hand = hand
        self.first_seen = t
        self.last_seen = t - 1 / 30
        self.ema = {s: 0.0 for s in SHAPES}
        self.stable: str | None = None
        self.stable_since = t
        self.candidate: str | None = None
        self.candidate_since = t
        self.episode_fired = False
        self.approach_peak: float | None = None
        self.close_since: float | None = None
        self.close_fired = False
        self.vel = (0.0, 0.0)
        self.history: deque[Sample] = deque()

    def age(self, t: float) -> float:
        return t - self.first_seen

    def update(self, hand: Hand, t: float, cfg: LogicConfig, ego=None) -> None:
        dt = max(1e-3, t - self.last_seen)
        alpha = 1.0 - math.exp(-dt / cfg.ema_tau_s)
        for s in SHAPES:
            target = hand.conf if (hand.engaged and hand.shape == s) else 0.0
            self.ema[s] += alpha * (target - self.ema[s])

        prev = self.stable
        if self.stable is not None and self.ema[self.stable] <= cfg.release_conf:
            self.stable = None

        best = max(SHAPES, key=self.ema.__getitem__)
        if best != self.stable and self.ema[best] >= cfg.onset_conf:
            if self.candidate != best:
                self.candidate, self.candidate_since = best, t
            if t - self.candidate_since >= cfg.onset_s and (
                self.stable is None or self.ema[self.stable] < cfg.onset_conf
            ):
                self.stable = best
        else:
            self.candidate = None

        if self.stable != prev:
            self.stable_since = t
            self.episode_fired = False

        vx = (hand.center[0] - self.hand.center[0]) / dt
        vy = (hand.center[1] - self.hand.center[1]) / dt
        k = 1.0 - math.exp(-dt / 0.1)
        self.vel = (self.vel[0] + k * (vx - self.vel[0]), self.vel[1] + k * (vy - self.vel[1]))

        du, dv, busy = (ego.du, ego.dv, ego.busy) if ego is not None else (0.0, 0.0, False)
        engaged = hand.engaged or (hand.reject == "relaxed" and self.stable is not None)
        self.hand = hand
        self.side = hand.side
        self.last_seen = t
        fistish = self.stable == "fist" or self.ema["fist"] >= 0.5
        self.history.append(Sample(t, hand.center[0] - du, hand.center[1] - dv, hand.scale, self.stable, engaged, busy, fistish))
        while self.history and t - self.history[0].t > cfg.history_s:
            self.history.popleft()

    def window(self, t: float, seconds: float) -> list[Sample]:
        return [s for s in self.history if t - s.t <= seconds]

    def had_stable(self, shape: str, t: float, seconds: float) -> bool:
        return any(s.stable == shape for s in self.window(t, seconds))

    def predict(self, point: tuple[float, float], lead_s: float) -> tuple[float, float]:
        """Where this point will be after lead_s, to cancel camera + processing lag."""
        return (
            min(1.0, max(0.0, point[0] + self.vel[0] * lead_s)),
            min(1.0, max(0.0, point[1] + self.vel[1] * lead_s)),
        )

    def display_conf(self) -> float:
        key = self.stable or max(SHAPES, key=self.ema.__getitem__)
        return self.ema[key]


@dataclass
class Decision:
    mode: str = "idle"
    active: str | None = None
    fire: str | None = None
    detail: str = ""
    head_uv: tuple[float, float] | None = None
    head_gain: float = 1.0
    antennas_deg: tuple[float, float] | None = None  # finger tilt of the left/right image hand
    body_x: float | None = None
    tracks: list[Track] = field(default_factory=list)


def _reversals(samples: list[Sample], amp: float) -> int:
    if len(samples) < 3:
        return 0
    direction = 0
    extreme = samples[0].cx
    count = 0
    for s in samples[1:]:
        x = s.cx
        if direction >= 0 and x < extreme - amp:
            count += direction == 1
            direction, extreme = -1, x
        elif direction <= 0 and x > extreme + amp:
            count += direction == -1
            direction, extreme = 1, x
        elif (direction == 1 and x > extreme) or (direction == -1 and x < extreme):
            extreme = x
    return count


def _quiet(samples: list[Sample]) -> bool:
    return bool(samples) and not any(s.busy for s in samples)


# Hands that still count toward two-hand mode although they aren't making a
# clear shape this frame (a fist still closing, a finger still rising).
TWO_HAND_OK_REJECTS = {"relaxed", "not facing"}


def finger_tilt_deg(hand: Hand) -> float:
    """Tilt of the index finger (knuckle -> tip) in the camera plane:
    0 = straight up, negative = leaning toward image left, positive = right."""
    d = hand.world[8] - hand.world[5]
    if abs(float(d[0])) + abs(float(d[1])) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(float(d[0]), float(-d[1])))


class GestureLogic:
    def __init__(self, cfg: LogicConfig | None = None) -> None:
        self.cfg = cfg or LogicConfig()
        self.tracks: dict[int, Track] = {}
        self._ids = itertools.count(1)
        self._latched: dict[str, bool] = {}
        self._last_fire: dict[str, float] = {}
        self._primary_id: int | None = None
        self._namaste_since: float | None = None
        # Motion-gesture candidates we refused, by reason; shown in /status.
        self.suppressed: collections.Counter[str] = collections.Counter()

    def _match(self, hands: list[Hand], t: float, ego) -> None:
        cfg = self.cfg
        remaining = list(hands)
        for tr in sorted(self.tracks.values(), key=lambda x: x.last_seen, reverse=True):
            if not remaining:
                break
            best, best_d = None, cfg.match_dist
            for h in remaining:
                # Handedness flips for a hand shown back-first; only trust a confident label.
                side_penalty = 0.08 if (h.side != tr.side and h.side_score >= 0.7) else 0.0
                d = math.dist(tr.hand.center, h.center) + side_penalty
                if d < best_d:
                    best, best_d = h, d
            if best is not None:
                tr.update(best, t, cfg, ego)
                remaining = [h for h in remaining if h is not best]
        for h in remaining:
            tr = Track(next(self._ids), h, t)
            tr.update(h, t, cfg, ego)
            self.tracks[tr.id] = tr
        for tid in [tid for tid, tr in self.tracks.items() if t - tr.last_seen > cfg.lost_s]:
            del self.tracks[tid]

    def _latch_ok(self, name: str, cond: bool) -> bool:
        if not cond:
            self._latched[name] = False
            return False
        return not self._latched.get(name, False)

    def _gap_ok(self, name: str, t: float) -> bool:
        return t - self._last_fire.get(name, -1e9) >= self.cfg.refire_gap_s

    def _approach(self, tr: Track, t: float, in_pair: bool) -> str | None:
        """Returns "" when an approach should fire, a suppression reason when a
        growing hand was refused, or None when nothing approach-like happened."""
        cfg = self.cfg
        win = tr.window(t, cfg.approach_window_s)
        if not win:
            return None
        if tr.approach_peak is not None and tr.hand.scale < 0.8 * tr.approach_peak and not win[-1].busy:
            tr.approach_peak = None
        low_i = min(range(len(win)), key=lambda i: win[i].scale)
        low = win[low_i]
        growing = tr.hand.scale >= cfg.approach_growth * low.scale and tr.hand.scale >= cfg.approach_min_scale
        if not growing or tr.approach_peak is not None:
            return None
        rise = win[low_i:]
        if not _quiet(win):
            return "ego"
        if in_pair:
            return "pair"
        if tr.stable != "open_palm":
            return "shape"
        if tr.age(t) < cfg.approach_window_s or not all(s.engaged for s in win):
            return "edge"
        # A push starts from a hand already held up in view, not one still arriving.
        before = [s for s in tr.history if s.t <= low.t and low.t - s.t <= cfg.approach_baseline_s]
        if before[0].t > low.t - cfg.approach_baseline_s + 0.05 or not all(s.engaged and not s.busy for s in before):
            return "edge"
        if len(rise) < cfg.approach_min_samples or rise[-1].t - rise[0].t < cfg.approach_min_span_s:
            return "jump"
        if any(b.scale < cfg.approach_max_dip * a.scale for a, b in zip(rise, rise[1:])):
            return "jump"
        if max(math.dist((s.cx, s.cy), (rise[-1].cx, rise[-1].cy)) for s in rise) > cfg.approach_max_move:
            return "moved"
        return ""

    def update(self, hands: list[Hand], t: float, ego=None) -> Decision:
        cfg = self.cfg
        self._match(hands, t, ego)
        decision = Decision(tracks=list(self.tracks.values()))

        # A latched gesture survives a blurry frame with no recognizable shape.
        engaged = [
            tr for tr in self.tracks.values()
            if (tr.hand.engaged or (tr.hand.reject == "relaxed" and tr.stable is not None))
            and tr.last_seen == t
            and tr.age(t) >= cfg.min_track_s
        ]
        engaged.sort(key=lambda tr: tr.hand.scale, reverse=True)
        # Two hands up in view = two-hand mode: no single-hand reaction fires from
        # either hand, so ta-da or two pointers never leak a fist or a point.
        visible = [
            tr for tr in self.tracks.values()
            if tr.last_seen == t
            and tr.age(t) >= cfg.min_track_s
            and (tr.hand.engaged or tr.hand.reject in TWO_HAND_OK_REJECTS)
        ]
        visible.sort(key=lambda tr: tr.hand.scale, reverse=True)
        pair = None
        if len(visible) >= 2 and visible[1].hand.scale >= cfg.pair_scale_ratio * visible[0].hand.scale:
            pair = (visible[0], visible[1])
        fire: str | None = None
        detail = ""

        # --- two-hand motion gestures ---
        def palmish(tr: Track) -> bool:
            return tr.stable == "open_palm" or tr.ema["open_palm"] >= 0.5

        both_palms = pair is not None and all(palmish(tr) for tr in pair)
        tada_ready = both_palms and all(any(s.fistish for s in tr.window(t, cfg.tada_window_s)) for tr in pair)
        if self._latch_ok("ta_da", both_palms) and tada_ready and self._gap_ok("ta_da", t):
            fire = "ta_da"
            self._latched["ta_da"] = True
            self._latched["raise_both"] = True

        def rose(tr: Track) -> bool:
            win = tr.window(t, cfg.raise_window_s)
            return _quiet(win) and max(s.cy for s in win) - win[-1].cy > cfg.raise_dist

        if fire is None and self._latch_ok("raise_both", both_palms) and pair and all(rose(tr) for tr in pair):
            if self._gap_ok("raise_both", t):
                fire = "raise_both"
                self._latched["raise_both"] = True

        # --- namaste: palms pressed together, fingers up ---
        # Palms pressed together are edge-on to the camera, so their shape is
        # often unreadable; this uses only where the two hands are and which way
        # the fingers point.
        def fingers_up(tr: Track) -> bool:
            p = tr.hand.pts
            return float(p[12, 1]) < float(p[0, 1]) - 0.5 * tr.hand.scale

        namaste = (
            pair is not None
            and math.dist(pair[0].hand.center, pair[1].hand.center)
            < cfg.namaste_gap * max(pair[0].hand.scale, pair[1].hand.scale)
            and all(fingers_up(tr) for tr in pair)
        )
        if not namaste:
            self._namaste_since = None
        elif self._namaste_since is None:
            self._namaste_since = t
        namaste_held = namaste and t - self._namaste_since >= cfg.namaste_hold_s
        if fire is None and self._latch_ok("namaste", namaste_held) and self._gap_ok("namaste", t):
            fire, detail = "namaste", "palms together"
            self._latched["namaste"] = True

        # --- too close: a hand right in front of the camera means Come close ---
        # Counts hands cut off by the frame edge too, since a very close hand
        # rarely fits in view. Fires once per approach; re-arms when it backs off.
        for tr in self.tracks.values():
            if tr.last_seen != t or tr.age(t) < cfg.min_track_s:
                continue
            hd = tr.hand
            if not (hd.engaged or hd.reject in ("cut off", "relaxed", "not facing")):
                continue
            if hd.scale >= cfg.close_scale or hd.area >= cfg.close_area:
                if tr.close_since is None:
                    tr.close_since = t
                if fire is None and not tr.close_fired and pair is None and t - tr.close_since >= cfg.close_hold_s:
                    if not _quiet(tr.window(t, cfg.close_hold_s)):
                        self.suppressed["approach:ego"] += 1
                    elif self._gap_ok("approach", t):
                        fire, detail = "approach", f"too close scale {hd.scale:.2f} area {hd.area:.2f}"
                        tr.close_fired = True
                        tr.episode_fired = True
            elif hd.scale < 0.8 * cfg.close_scale and hd.area < 0.8 * cfg.close_area:
                tr.close_since, tr.close_fired = None, False

        # --- single-hand motion gestures (one hand in view only) ---
        for tr in engaged:
            if fire is not None or pair is not None:
                break
            # Waving is just an eager Hello: one reaction per open-palm episode.
            if tr.stable == "open_palm" and pair is None and not tr.episode_fired and tr.close_since is None:
                win = tr.window(t, cfg.wave_window_s)
                if _reversals(win, cfg.wave_amp) >= 2:
                    if not _quiet(win):
                        self.suppressed["wave:ego"] += 1
                    elif self._gap_ok("open_palm", t):
                        fire, detail = "open_palm", "waving"
                        tr.episode_fired = True
                        break
            verdict = self._approach(tr, t, pair is not None)
            if verdict == "" and self._gap_ok("approach", t):
                win = tr.window(t, cfg.approach_window_s)
                fire = "approach"
                detail = f"scale {min(s.scale for s in win):.2f}->{tr.hand.scale:.2f}"
                tr.approach_peak = tr.hand.scale
                tr.episode_fired = True
            elif verdict:
                self.suppressed[f"approach:{verdict}"] += 1

        # --- single-hand pose reactions: once per hold, one hand in view only ---
        for tr in engaged:
            if fire is not None or pair is not None:
                break
            shape = tr.stable
            if shape not in POSE_EVENTS or tr.episode_fired or tr.close_since is not None:
                continue
            held = t - tr.stable_since
            others = [o for o in engaged if o is not tr]
            if shape == "open_palm":
                if pair is not None:
                    continue
                recent = tr.window(t, cfg.palm_still_s)
                still = _quiet(recent) and (
                    max(s.cx for s in recent) - min(s.cx for s in recent) < cfg.palm_still_move
                    and max(s.cy for s in recent) - min(s.cy for s in recent) < cfg.palm_still_move
                )
                if held < cfg.palm_still_s or not still:
                    continue
            else:
                if shape == "fist" and any(o.stable == "fist" or o.candidate == "fist" for o in others):
                    continue
                if held < cfg.pose_hold_s:
                    continue
            if not self._gap_ok(FIRE_AS.get(shape, shape), t):
                continue
            fire = FIRE_AS.get(shape, shape)
            detail = f"held {held:.2f}s"
            tr.episode_fired = True

        if fire is not None:
            self._last_fire[fire] = t
            decision.fire = fire
            decision.detail = detail

        # --- continuous modes ---
        def pointing(tr: Track) -> bool:
            return tr.stable == "point" or tr.ema["point"] >= cfg.release_conf - 0.05

        pointers = [tr for tr in engaged if tr.stable == "point"]
        # One confirmed pointer plus a second hand that is at least mostly pointing
        # is two pointers; waiting for both to be perfect felt like one-finger mode.
        if pair is not None and all(pointing(tr) for tr in pair) and any(tr.stable == "point" for tr in pair):
            left, right = sorted(pair, key=lambda tr: tr.hand.center[0])
            decision.mode = "antennas"
            decision.active = "point_both"
            decision.antennas_deg = (finger_tilt_deg(left.hand), finger_tilt_deg(right.hand))
            decision.body_x = (left.hand.center[0] + right.hand.center[0]) / 2
            self._primary_id = None
        elif pointers and pair is None:
            primary = next((tr for tr in pointers if tr.id == self._primary_id), None)
            if primary is None:
                primary = pointers[0]
                self._primary_id = primary.id
            decision.mode = "head_follow"
            decision.active = "point"
            decision.head_uv = primary.predict(primary.hand.tip, cfg.predict_s)
        else:
            self._primary_id = None
            if pair is not None:
                decision.mode = "watching"
                decision.head_uv = (
                    (pair[0].hand.center[0] + pair[1].hand.center[0]) / 2,
                    (pair[0].hand.center[1] + pair[1].hand.center[1]) / 2,
                )
                decision.head_gain = 0.5
            elif engaged:
                lead = engaged[0]
                decision.mode = "watching"
                decision.head_uv = lead.predict(lead.hand.center, cfg.predict_s)
                decision.head_gain = 0.5

        if decision.active is None:
            if fire:
                decision.active = fire
            elif pair is not None:
                # Two hands in view: name the two-hand gesture being formed, never
                # a single-hand one (two open palms are Hands up, not Hello).
                if namaste:
                    decision.active = "namaste"
                elif both_palms:
                    decision.active = "raise_both"
                elif all(tr.stable == "fist" or tr.ema["fist"] >= 0.5 for tr in pair):
                    decision.active = "ta_da"
            elif engaged:
                decision.active = engaged[0].stable
        decision.active = FIRE_AS.get(decision.active, decision.active)
        return decision
