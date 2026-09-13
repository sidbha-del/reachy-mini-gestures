"""Offline checks for gesture_logic and motion_engine (no robot, no camera)."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from reachy_mini_gestures.gesture_logic import GestureLogic  # noqa: E402
from reachy_mini_gestures.recognizer import Hand  # noqa: E402

FAILS = []


def check(name, cond, detail=None):
    cond = bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail is not None else ""))
    if not cond:
        FAILS.append(name)


def hand(shape, cx=0.5, cy=0.45, conf=0.9, engaged=True, scale=0.12, side="Right"):
    return Hand(
        pts=np.zeros((21, 3)), world=np.zeros((21, 3)), side=side, shape=shape, conf=conf, source="model",
        center=(cx, cy), tip=(cx, cy - 0.1), area=scale * scale * 2, scale=scale, engaged=engaged,
        reject=None if engaged else "resting",
    )


def run(frames, fps=30.0, logic=None):
    """frames: list of (list_of_hands). Returns (logic, fires [(t, name)], decisions)."""
    logic = logic or GestureLogic()
    fires, decisions = [], []
    for i, hands in enumerate(frames):
        t = 1000.0 + i / fps
        d = logic.update(hands, t)
        decisions.append(d)
        if d.fire:
            fires.append((round(t - 1000.0, 3), d.fire))
    return logic, fires, decisions


# 1. Held fist fires exactly once, reasonably fast
_, fires, _ = run([[hand("fist")]] * 60)
check("held fist fires once", [f[1] for f in fires] == ["fist"], fires)
check("fist fires within 0.6s", fires and fires[0][0] <= 0.6, fires)

# 2. Release and re-show fires again
_, fires, _ = run([[hand("fist")]] * 30 + [[]] * 15 + [[hand("fist")]] * 30)
check("fist re-arms after release", [f[1] for f in fires] == ["fist", "fist"], fires)

# 3. Flickering noise never fires
rng = np.random.default_rng(1)
shapes = ["fist", "peace", "open_palm", "thumbs_up", None, "love"]
noise = [[hand(shapes[rng.integers(len(shapes))], conf=0.7)] for _ in range(150)]
_, fires, _ = run(noise)
check("single-frame noise never fires", fires == [], fires)

# 4. Not-engaged hand never fires
_, fires, _ = run([[hand("thumbs_up", engaged=False)]] * 90)
check("ignored (resting) hand never fires", fires == [], fires)

# 5. Still open palm -> hello once; waving palm -> wave, no hello
_, fires, _ = run([[hand("open_palm")]] * 60)
check("still palm says hello once", [f[1] for f in fires] == ["open_palm"], fires)
wave_frames = [[hand("open_palm", cx=0.5 + 0.08 * np.sin(2 * np.pi * 2.0 * i / 30))] for i in range(60)]
_, fires, _ = run(wave_frames)
names = [f[1] for f in fires]
check("waving palm says hello once (wave merged)", names == ["open_palm"], fires)

# 6. Two fists -> two palms = ta-da, no shy fist
tada = [[hand("fist", cx=0.3, side="Left"), hand("fist", cx=0.7)]] * 20 + [[hand("open_palm", cx=0.3, side="Left"), hand("open_palm", cx=0.7)]] * 20
_, fires, _ = run(tada)
check("ta-da fires once without fist", [f[1] for f in fires] == ["ta_da"], fires)

# 7. Two pointers -> antennas mode with each hand's height
_, _, decs = run([[hand("point", cx=0.3, cy=0.3, side="Left"), hand("point", cx=0.7, cy=0.7)]] * 20)
d = decs[-1]
check("two pointers -> antennas", d.mode == "antennas" and d.antennas_deg == (0.0, 0.0), (d.mode, d.antennas_deg))


def tilted_pointer(cx, tilt_deg, side):
    w = np.zeros((21, 3))
    w[8] = [0.06 * np.sin(np.radians(tilt_deg)), -0.06 * np.cos(np.radians(tilt_deg)), 0.0]
    return Hand(pts=np.zeros((21, 3)), world=w, side=side, shape="point", conf=0.9, source="model",
                center=(cx, 0.45), tip=(cx, 0.35), area=0.03, scale=0.12, engaged=True, reject=None)


_, _, decs = run([[tilted_pointer(0.3, -40, "Left"), tilted_pointer(0.7, 25, "Right")]] * 20)
ang = decs[-1].antennas_deg
check("antennas copy each finger's tilt", ang is not None and abs(ang[0] + 40) < 1 and abs(ang[1] - 25) < 1, ang)

# Two hands in view: a fist while the other hand is still opening is not a single fist
relaxed2 = Hand(pts=np.zeros((21, 3)), world=np.zeros((21, 3)), side="Left", shape=None, conf=0.0, source="none",
                center=(0.3, 0.45), tip=(0.3, 0.35), area=0.03, scale=0.12, engaged=False, reject="relaxed")
_, fires, _ = run([[hand("fist", cx=0.7), relaxed2]] * 40)
check("two hands in view: no single fist", "fist" not in [f[1] for f in fires], fires)
_, _, decs = run([[hand("point", cx=0.7), relaxed2]] * 20)
check("two hands in view: no single head follow", decs[-1].mode != "head_follow", decs[-1].mode)

# 8. Pointer + small background pointer -> head follow, not antennas
_, _, decs = run([[hand("point", cx=0.4, scale=0.14), hand("point", cx=0.8, scale=0.05, side="Left")]] * 20)
check("small background hand can't pair", decs[-1].mode == "head_follow", decs[-1].mode)

# 9. Resting second hand doesn't pair
_, _, decs = run([[hand("point", cx=0.3, side="Left"), hand("point", cx=0.7, engaged=False)]] * 20)
check("resting second hand can't pair", decs[-1].mode == "head_follow", decs[-1].mode)

# 10. 10 fps robot camera still recognizes quickly
_, fires, _ = run([[hand("peace")]] * 20, fps=10.0)
check("10fps peace fires once within 0.6s", [f[1] for f in fires] == ["peace"] and fires[0][0] <= 0.6, fires)

# 11. Opening a fist does not trigger approach (scale is shape independent)
_, fires, _ = run([[hand("fist")]] * 20 + [[hand("open_palm")]] * 20)
check("opening a hand is not 'approach'", "approach" not in [f[1] for f in fires], fires)

# 12. Pushing a hand toward the camera triggers approach once
def push_frames(n_still=30, n_push=15, n_after=20, **kw):
    return (
        [[hand("open_palm", scale=0.12, **kw)]] * n_still
        + [[hand("open_palm", scale=0.12 + 0.12 * i / n_push, **kw)] for i in range(1, n_push + 1)]
        + [[hand("open_palm", scale=0.24, **kw)]] * n_after
    )


_, fires, _ = run(push_frames())
check("approach fires once", [f[1] for f in fires].count("approach") == 1, fires)


# ---- robustness: robot self-motion, look-alikes, back of hand ----
class E:  # stand-in for motion_engine.Ego
    def __init__(self, du=0.0, dv=0.0, speed=0.0, busy=False):
        self.du, self.dv, self.speed, self.busy = du, dv, speed, busy


def run_ego(frames, egos, fps=30.0):
    logic = GestureLogic()
    fires = []
    for i, (hands, ego) in enumerate(zip(frames, egos)):
        t = 1000.0 + i / fps
        d = logic.update(hands, t, ego)
        if d.fire:
            fires.append((round(t - 1000.0, 3), d.fire))
    return logic, fires


# R1. Robot turning: the hand slides across the image, but it is still in the world
n = 90
turn = [0.15 * np.sin(2 * np.pi * 1.5 * i / 30) for i in range(n)]
frames = [[hand("open_palm", cx=0.5 + u)] for u in turn]
_, fires = run_ego(frames, [E(du=u, speed=10) for u in turn])
names = [f[1] for f in fires]
check("head panning is not a wave", "wave" not in names, fires)
check("compensated still palm still says hello", names == ["open_palm"], fires)

# R2. Scale grows while a clip runs -> no approach; the same push after settling -> once
frames = push_frames()
logic, fires = run_ego(frames, [E(busy=True)] * len(frames))
check("growth during own clip is not approach", "approach" not in [f[1] for f in fires], fires)
check("suppression counted as ego", logic.suppressed.get("approach:ego", 0) > 0, dict(logic.suppressed))
frames = push_frames(n_still=40) + [[hand("open_palm", scale=0.24)]] * 10
egos = [E(busy=i < 20) for i in range(len(frames))]
_, fires = run_ego(frames, egos)
check("real push after robot settles fires once", [f[1] for f in fires].count("approach") == 1, fires)

# R3. Robot leans back and returns (hand shrinks then grows) during the clip -> no re-fire
frames = push_frames() + [[hand("open_palm", scale=0.24 - 0.08 * min(1, i / 8))] for i in range(15)] + \
    [[hand("open_palm", scale=0.16 + 0.08 * min(1, i / 8))] for i in range(15)]
egos = [E(busy=i >= 45) for i in range(len(frames))]
_, fires = run_ego(frames, egos)
check("robot's own lean-back/return doesn't re-fire approach", [f[1] for f in fires].count("approach") == 1, fires)

# R4. Hand sliding in from the edge (not engaged at first) -> no approach
frames = [[hand("open_palm", scale=0.08 + 0.01 * i, engaged=False)] for i in range(8)] + \
    [[hand("open_palm", scale=0.16 + 0.01 * i)] for i in range(8)] + [[hand("open_palm", scale=0.24)]] * 20
_, fires, _ = run(frames)
check("hand entering the frame is not approach", "approach" not in [f[1] for f in fires], fires)

# R5. Fist or pointing (not a palm) growing -> no approach
_, fires, _ = run([[hand("fist", scale=0.12 + 0.12 * min(1, i / 12))] for i in range(40)])
check("growing fist is not approach", "approach" not in [f[1] for f in fires], fires)

# R6. Sweeping hand that also grows -> no approach
frames = [[hand("open_palm", scale=0.12)]] * 30 + [[hand("open_palm", cx=0.5 - 0.02 * i, scale=0.12 + 0.008 * i)] for i in range(1, 16)]
_, fires, _ = run(frames)
check("sideways sweep that grows is not approach", "approach" not in [f[1] for f in fires], fires)

# R7. Two palms rising and growing -> only hands up
frames = [[hand("open_palm", cx=0.3, cy=0.8, side="Left"), hand("open_palm", cx=0.7, cy=0.8)]] * 15 + [
    [hand("open_palm", cx=0.3, cy=0.8 - 0.02 * i, scale=0.12 + 0.008 * i, side="Left"),
     hand("open_palm", cx=0.7, cy=0.8 - 0.02 * i, scale=0.12 + 0.008 * i)] for i in range(1, 16)]
_, fires, _ = run(frames)
check("hands up doesn't also fire approach", [f[1] for f in fires] == ["raise_both"], fires)

# R8. Head pitching down makes still hands appear to rise -> no hands up
frames, egos = [], []
for i in range(40):
    dv = -0.015 * max(0, i - 15)
    frames.append([hand("open_palm", cx=0.3, cy=0.8 + dv, side="Left"), hand("open_palm", cx=0.7, cy=0.8 + dv)])
    egos.append(E(dv=dv, speed=15))
_, fires = run_ego(frames, egos)
check("robot pitching is not hands up", "raise_both" not in [f[1] for f in fires], fires)

# R9. Fist on the way to thumbs up -> only thumbs up
_, fires, _ = run([[hand("fist")]] * 6 + [[hand("thumbs_up")]] * 30)
check("brief fist before thumbs up doesn't fire fist", [f[1] for f in fires] == ["thumbs_up"], fires)
_, fires, _ = run([[hand("point")]] * 7 + [[hand("peace")]] * 30)
check("point on the way to peace fires only peace", [f[1] for f in fires] == ["peace"], fires)

# A1. Two open palms are shown as Hands up (or Ta-da for two fists), never Hello / Fist
_, _, decs = run([[hand("open_palm", cx=0.3, side="Left"), hand("open_palm", cx=0.7)]] * 20)
check("two palms show Hands up, not Hello", decs[-1].active == "raise_both", decs[-1].active)
_, _, decs = run([[hand("fist", cx=0.3, side="Left"), hand("fist", cx=0.7)]] * 20)
check("two fists show Ta-da, not Fist", decs[-1].active == "ta_da", decs[-1].active)

# L1. A held finger heart triggers Love you (one merged gesture)
_, fires, decs = run([[hand("heart")]] * 40)
check("finger heart fires love", [f[1] for f in fires] == ["love"] and decs[-1].active == "love", (fires, decs[-1].active))

# R10. Low handedness confidence (back of hand) doesn't break track matching or engagement
from reachy_mini_gestures.recognizer import EngageConfig, assess_engagement, rule_shape  # noqa: E402
flip = [[Hand(pts=np.zeros((21, 3)), world=np.zeros((21, 3)), side=("Left" if i % 2 else "Right"), shape="point",
              conf=0.9, source="rules", center=(0.5, 0.45), tip=(0.5, 0.35), area=0.03, scale=0.12,
              engaged=True, reject=None, side_score=0.5)] for i in range(20)]
logic, _, decs = run(flip)
check("flickering handedness keeps one track", len(logic.tracks) == 1 and decs[-1].mode == "head_follow", (len(logic.tracks), decs[-1].mode))
pts = np.zeros((21, 3)); pts[:, 0] = np.linspace(0.4, 0.6, 21); pts[:, 1] = np.linspace(0.4, 0.65, 21)
ok, reason, _ = assess_engagement(pts, np.zeros((21, 3)), 0.3, "point", EngageConfig())
check("low handedness score still engaged", ok, reason)
small = np.zeros((21, 3)); small[:, 0] = np.linspace(0.45, 0.53, 21); small[:, 1] = np.linspace(0.40, 0.50, 21)
ok, reason, _ = assess_engagement(small, np.zeros((21, 3)), 0.9, "heart", EngageConfig(), scale=0.09)
check("compact finger heart at normal distance is not 'too far'", ok, reason)
ok, reason, _ = assess_engagement(small, np.zeros((21, 3)), 0.9, "heart", EngageConfig(), scale=0.03)
check("genuinely distant small hand is still 'too far'", not ok and reason == "too far", reason)


def synth_hand(extended, folded_ambiguous=()):
    """World landmarks of a flat hand pointing +x; fingers either straight,
    fully curled, or bent ~135deg at the PIP with the tip tucked in (as when
    hidden behind the back of the hand)."""
    w = np.zeros((21, 3))
    bases = {1: -0.03, 5: -0.015, 9: 0.0, 13: 0.015, 17: 0.03}
    for mcp, y in bases.items():
        if mcp == 1:
            w[1:5] = [[0.02, -0.02 - 0.01 * k, 0] for k in range(4)]
            continue
        w[mcp] = [0.08, y, 0]
        if mcp in extended:
            for k in range(1, 4):
                w[mcp + k] = [0.08 + 0.025 * k, y, 0]
        elif mcp in folded_ambiguous:
            w[mcp + 1] = [0.105, y, 0]
            w[mcp + 2] = [0.105 - 0.013, y, 0.013]
            w[mcp + 3] = [0.105 - 0.03, y, 0.0]
        else:
            w[mcp + 1] = [0.1, y, 0.01]
            w[mcp + 2] = [0.085, y, 0.02]
            w[mcp + 3] = [0.075, y, 0.01]
    return w


w = synth_hand({5}, folded_ambiguous={9, 13, 17})
check("back-of-hand point (occluded fingers) -> point", rule_shape(w, w) == "point", rule_shape(w, w))
w = synth_hand({5})
check("clear point -> point", rule_shape(w, w) == "point", rule_shape(w, w))
w = synth_hand({5, 9, 13, 17})
check("open hand -> open_palm", rule_shape(w, w) == "open_palm", rule_shape(w, w))
w = synth_hand({9, 13, 17}); w[8] = w[4] + [0.005, 0, 0]
check("thumb-index circle, others up -> ok", rule_shape(w, w) == "ok", rule_shape(w, w))
# Korean finger heart: bent index sticking out past the folded fingers, thumb tip against it
w = synth_hand(set())
w[6] = [0.1, -0.015, 0.0]; w[7] = [0.105, -0.015, 0.02]; w[8] = [0.1, -0.02, 0.035]
w[4] = w[7] + [0.004, -0.008, 0.004]
check("finger heart -> heart", rule_shape(w, w) == "heart", rule_shape(w, w))
w = synth_hand(set())
check("plain fist is not a finger heart", rule_shape(w, w) != "heart", rule_shape(w, w))
import tempfile  # noqa: E402

from reachy_mini_gestures.recognizer import TeachStore  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    store = TeachStore(os.path.join(tmp, "teach.json"))
    heart = synth_hand(set())
    heart[6] = [0.1, -0.015, 0.0]; heart[7] = [0.105, -0.015, 0.02]; heart[8] = [0.1, -0.02, 0.035]
    store.add("heart", heart, "Right")
    store.add("fist", synth_hand(set()), "Right")
    label, d, other = store.match_scored(heart, "Right")
    check("taught example matches itself closely", label == "heart" and d < 0.05 and other > d, (label, d, other))
    check("taught heart persists", TeachStore(os.path.join(tmp, "teach.json")).counts() == {"heart": 1, "fist": 1})

w = synth_hand({5}); w[4] = w[6] + [0.0, -0.008, 0.0]
check("point with thumb resting on index is not a heart", rule_shape(w, w) == "point", rule_shape(w, w))

# N1. Quick ta-da at 10fps (short fists, then palms)
tada10 = [[hand("fist", cx=0.3, side="Left"), hand("fist", cx=0.7)]] * 5 + \
    [[hand("open_palm", cx=0.3, side="Left"), hand("open_palm", cx=0.7)]] * 12
_, fires, _ = run(tada10, fps=10.0)
check("quick ta-da at 10fps fires", "ta_da" in [f[1] for f in fires], fires)
# N2. Two pointers at different distances -> antennas, not one-finger follow
_, _, decs = run([[hand("point", cx=0.3, scale=0.14, side="Left"), hand("point", cx=0.7, scale=0.08)]] * 20)
check("two pointers at different distances -> antennas", decs[-1].mode == "antennas", decs[-1].mode)
# N3. Second pointer only weakly recognized still counts
frames = [[hand("point", cx=0.3, side="Left"), hand("point" if i % 3 else None, cx=0.7, conf=0.7)] for i in range(30)]
_, _, decs = run(frames)
check("flaky second pointer still -> antennas", sum(d.mode == "antennas" for d in decs[-10:]) >= 8, [d.mode for d in decs[-10:]])
# N4. Slower hands up (rise over ~0.9s)
frames = [[hand("open_palm", cx=0.3, cy=0.8, side="Left"), hand("open_palm", cx=0.7, cy=0.8)]] * 15 + [
    [hand("open_palm", cx=0.3, cy=0.8 - 0.012 * i, side="Left"), hand("open_palm", cx=0.7, cy=0.8 - 0.012 * i)] for i in range(1, 28)]
_, fires, _ = run(frames)
check("slower hands up fires", "raise_both" in [f[1] for f in fires], fires)
# N5. Head gazing along with rising hands (moderate speed) doesn't block hands up
egos = [E(dv=-0.004 * max(0, i - 15), speed=40) for i in range(len(frames))]
frames_g = [[hand("open_palm", cx=h.center[0], cy=h.center[1] - 0.004 * max(0, i - 15), side=h.side) for h in f] for i, f in enumerate(frames)]
_, fires = run_ego(frames_g, egos)
check("hands up works while Reachy's gaze follows", "raise_both" in [f[1] for f in fires], fires)


# 13. Any engaged hand gets a soft gaze; a moving pointer is led ahead
_, _, decs = run([[hand("fist", cx=0.2)]] * 20)
d = decs[-1]
check("soft gaze follows a non-pointing hand", d.mode == "watching" and d.head_gain == 0.5 and d.head_uv is not None, (d.mode, d.head_uv))
_, _, decs = run([[hand("point", cx=0.3 + 0.02 * i)] for i in range(20)])
check("moving pointer target is predicted ahead", decs[-1].head_uv[0] > 0.3 + 0.02 * 19, decs[-1].head_uv)
_, _, decs = run([[hand("fist", engaged=False)]] * 20)
check("resting hand gets no gaze", decs[-1].head_uv is None, decs[-1].mode)

# 14. Relaxed (shapeless) hand is ignored; a latched point survives one blurry frame
from reachy_mini_gestures.recognizer import EngageConfig, assess_engagement  # noqa: E402
pts = np.zeros((21, 3)); pts[:, 0] = np.linspace(0.4, 0.6, 21); pts[:, 1] = np.linspace(0.4, 0.65, 21)
ok, reason, _ = assess_engagement(pts, np.zeros((21, 3)), 0.95, None, EngageConfig())
check("shapeless hand is 'relaxed', not engaged", not ok and reason == "relaxed", reason)
relaxed = Hand(pts=np.zeros((21, 3)), world=np.zeros((21, 3)), side="Right", shape=None, conf=0.0, source="none",
               center=(0.5, 0.45), tip=(0.5, 0.35), area=0.03, scale=0.12, engaged=False, reject="relaxed")
_, _, decs = run([[hand("point")]] * 15 + [[relaxed]] + [[hand("point")]] * 2)
check("blurry frame doesn't drop head follow", all(d.mode == "head_follow" for d in decs[-3:]), [d.mode for d in decs[-3:]])
_, _, decs = run([[relaxed]] * 30)
check("relaxed hand alone gets no gaze", decs[-1].head_uv is None, decs[-1].mode)

# C1. A palm held right in front of the camera is Come close, not Hello
_, fires, _ = run([[hand("open_palm", scale=0.35)]] * 40)
check("too-close palm -> come close, not hello", [f[1] for f in fires] == ["approach"], fires)
# C2. A very close hand cut off by the frame edge still counts
cut = Hand(pts=np.zeros((21, 3)), world=np.zeros((21, 3)), side="Right", shape=None, conf=0.0, source="none",
           center=(0.5, 0.6), tip=(0.5, 0.3), area=0.3, scale=0.4, engaged=False, reject="cut off")
_, fires, _ = run([[cut]] * 30)
check("cut-off close hand -> come close once", [f[1] for f in fires] == ["approach"], fires)
# C3. Close during the robot's own movement is ignored, then fires once when still
frames = [[hand("open_palm", scale=0.35)]] * 40
_, fires = run_ego(frames, [E(busy=i < 15) for i in range(len(frames))])
check("too close while robot moves waits, then fires once", [f[1] for f in fires] == ["approach"], fires)



# ---- motion engine: stale intents are dropped, fresh ones play ----
class FakeMedia:
    camera = None


class FakeMini:
    def __init__(self):
        self.media = FakeMedia()
        self.T_head_cam = np.eye(4)
        self.calls = []
        self.body = []

    def get_current_head_pose(self):
        return np.eye(4)

    def get_present_antenna_joint_positions(self):
        return [0.0, 0.0]

    def set_target(self, head=None, antennas=None, body_yaw=None):
        self.calls.append((time.time(), np.array(antennas)))
        self.body.append(body_yaw)


from reachy_mini_gestures.motion_engine import MotionEngine  # noqa: E402

mini = FakeMini()
engine = MotionEngine(mini, sound_enabled=lambda: False)
engine.start()
engine.paused = False
check("stale intent (1.5s old) dropped", engine.trigger("open_palm", time.time() - 1.5) is False and engine.dropped_stale == 1)
check("fresh intent accepted", engine.trigger("open_palm", time.time() - 0.05) is True)
time.sleep(0.35)
peak = max((float(np.max(np.abs(a))) for _, a in mini.calls), default=0.0)
check("clip moves antennas", peak > np.deg2rad(15), f"peak={np.rad2deg(peak):.1f}deg")
check("motor latency measured", engine.motor_latency_ms is not None and engine.motor_latency_ms < 200, engine.motor_latency_ms)
time.sleep(1.6)  # let the wave-back clip (1.8s) finish
for _ in range(10):  # the vision loop refreshes follow targets every frame
    engine.follow_antennas(0.2, 0.8, time.time())
    time.sleep(0.05)
last = mini.calls[-1][1]
check("antenna follow reaches targets", np.rad2deg(last[0]) < -30 and np.rad2deg(last[1]) < -30, np.rad2deg(last))
for _ in range(12):
    engine.follow_antennas(0.5, 0.5, time.time(), center_x=0.2)
    time.sleep(0.05)
check("two-hand center turns the body", np.rad2deg(mini.body[-1]) > 12, np.rad2deg(mini.body[-1]))
for _ in range(15):
    engine.follow_antenna_angles(-40.0, 40.0, time.time())
    time.sleep(0.05)
last = np.rad2deg(mini.calls[-1][1])
check("antennas reach finger angles", abs(last[0] + 40) < 5 and abs(last[1] - 40) < 5, last)
engine.follow_head(0.15, 0.5, "webcam", time.time())
for _ in range(10):
    engine.follow_head(0.15, 0.5, "webcam", time.time())
    time.sleep(0.05)
check("pointing to the side turns body with head", np.rad2deg(mini.body[-1]) > 10, np.rad2deg(mini.body[-1]))
engine.follow_head(0.15, 0.5, "webcam", time.time(), gain=0.5)
time.sleep(0.05)
check("soft gaze stores a partial target", abs(engine._head_target[0][2] - 0.5 * (0.5 - 0.15) * 2 * 45) < 1e-6, engine._head_target[0])
engine.trigger("peace", time.time())
time.sleep(0.25)
check("peace clip swings the body", abs(np.rad2deg(mini.body[-1])) > 10, np.rad2deg(mini.body[-1]))
ego = engine.ego_at(time.time() - 0.05)
check("ego is busy during a clip", ego.busy, (ego.speed, ego.busy))
time.sleep(0.75 + 0.35)  # clip (0.95s) + settle
engine.follow_head(0.3, 0.5, "webcam", time.time())
for _ in range(20):
    engine.follow_head(0.3, 0.5, "webcam", time.time())
    time.sleep(0.05)
ego = engine.ego_at(time.time() - 0.1)
yaw = engine._pose_at(time.time() - 0.1)[5]
expected = np.deg2rad(yaw) / np.deg2rad(65.0)
check("ego settles after the clip", not ego.busy, (ego.speed, ego.busy))
check("ego du follows head yaw (left turn -> scene right)", yaw > 5 and abs(ego.du - expected) < 1e-3, (yaw, ego.du, expected))
engine.stop()

print()
print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
