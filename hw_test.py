"""Reachy Mini hardware verification script.

Run one test at a time:
    python hw_test.py movement
    python hw_test.py camera
    python hw_test.py speaker
    python hw_test.py mic
    python hw_test.py touch
    python hw_test.py imu
"""

import os
import sys
import time

import numpy as np

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose


def test_movement(mini: ReachyMini) -> None:
    print("Watch the robot: antennas -> head nod/shake/lift -> body rotate.")

    print("Antennas...")
    mini.goto_target(antennas=np.deg2rad([45, -45]), duration=0.6)
    mini.goto_target(antennas=np.deg2rad([-45, 45]), duration=0.6)
    mini.goto_target(antennas=np.deg2rad([0, 0]), duration=0.6)

    print("Head nod (pitch)...")
    mini.goto_target(head=create_head_pose(pitch=15), duration=0.8)
    mini.goto_target(head=create_head_pose(pitch=-15), duration=0.8)
    mini.goto_target(head=create_head_pose(pitch=0), duration=0.8)

    print("Head shake (yaw)...")
    mini.goto_target(head=create_head_pose(yaw=20), duration=0.8)
    mini.goto_target(head=create_head_pose(yaw=-20), duration=0.8)
    mini.goto_target(head=create_head_pose(yaw=0), duration=0.8)

    print("Head lift (z)...")
    mini.goto_target(head=create_head_pose(z=15, mm=True), duration=0.8)
    mini.goto_target(head=create_head_pose(z=-5, mm=True), duration=0.8)
    mini.goto_target(head=create_head_pose(z=0, mm=True), duration=0.8)

    print("Body rotation...")
    mini.goto_target(body_yaw=np.deg2rad(30), duration=1.0)
    mini.goto_target(body_yaw=np.deg2rad(-30), duration=1.0)
    mini.goto_target(body_yaw=0, duration=1.0)

    print("Movement test done.")


def test_camera(mini: ReachyMini) -> None:
    frame_bytes = mini.media.get_frame_jpeg()
    if not frame_bytes:
        raise RuntimeError("No frame returned by camera.")

    out_path = os.path.join(os.path.dirname(__file__), "camera_snapshot.jpg")
    with open(out_path, "wb") as f:
        f.write(frame_bytes)

    print(f"Saved snapshot to {out_path}")
    if sys.platform == "win32":
        os.startfile(out_path)  # noqa: S606 - opens with default viewer for user to see
    print("Opened snapshot in your default image viewer. Check what it shows.")


def test_speaker(mini: ReachyMini) -> None:
    # Locate a bundled sound asset shipped with the reachy_mini package.
    import reachy_mini

    assets_dir = os.path.join(os.path.dirname(reachy_mini.__file__), "assets")
    wav_path = os.path.join(assets_dir, "wake_up.wav")
    if not os.path.exists(wav_path):
        raise RuntimeError(f"Expected bundled sound not found: {wav_path}")

    print(f"Playing {wav_path} through the robot's speaker...")
    mini.media.start_playing()
    mini.media.play_sound(wav_path)
    time.sleep(3.0)
    mini.media.stop_playing()
    print("Speaker test done. Did you hear it?")


def test_mic(mini: ReachyMini) -> None:
    from scipy.signal import resample

    duration = 5.0

    print("Get ready to speak...")
    for n in (3, 2, 1):
        print(n)
        time.sleep(1.0)

    print(">>> RECORDING NOW - SPEAK <<<")
    mini.goto_target(antennas=np.deg2rad([45, 45]), duration=0.2)  # ears up = recording
    mini.media.start_recording()
    mini.media.start_playing()

    chunks = []
    start = time.time()
    remaining_last = None
    while time.time() - start < duration:
        remaining = int(duration - (time.time() - start))
        if remaining != remaining_last:
            print(f"  ...{remaining}s left")
            remaining_last = remaining
        chunk = mini.media.get_audio_sample()
        if chunk is not None and len(chunk) > 0:
            chunks.append(chunk)
        time.sleep(0.02)

    mini.goto_target(antennas=np.deg2rad([0, 0]), duration=0.2)  # ears down = stopped
    print(">>> RECORDING STOPPED <<<")

    samples = np.concatenate(chunks, axis=0)
    print(f"Captured {samples.shape[0]} samples, {samples.shape[1]} channel(s).")

    in_rate = mini.media.get_input_audio_samplerate()
    out_rate = mini.media.get_output_audio_samplerate()
    if in_rate != out_rate:
        samples = resample(samples, int(out_rate * len(samples) / in_rate))

    time.sleep(0.5)
    print(">>> PLAYING BACK NOW - LISTEN <<<")
    mini.goto_target(antennas=np.deg2rad([-45, -45]), duration=0.2)  # ears down-out = playback
    mini.media.push_audio_sample(samples)
    time.sleep(len(samples) / out_rate + 0.5)
    mini.goto_target(antennas=np.deg2rad([0, 0]), duration=0.2)
    print(">>> PLAYBACK FINISHED <<<")

    mini.media.stop_recording()
    mini.media.stop_playing()
    print("Mic test done. Did you hear your own recording played back?")


def test_touch(mini: ReachyMini) -> None:
    duration = 15.0
    print("Watch for 3 quick antenna flicks -> that's your signal the robot is about to go limp.")
    print("Get ready...")
    for n in (3, 2, 1):
        print(n)
        time.sleep(1.0)

    for _ in range(3):
        mini.goto_target(antennas=np.deg2rad([60, 60]), duration=0.15)
        mini.goto_target(antennas=np.deg2rad([-60, -60]), duration=0.15)
    mini.goto_target(antennas=np.deg2rad([0, 0]), duration=0.15)

    mini.disable_motors()
    print("")
    print("#" * 60)
    print("#  LIMP NOW - GRAB AND MOVE THE HEAD / ANTENNAS BY HAND  #")
    print("#" * 60)

    start = time.time()
    last_print = 0.0
    while time.time() - start < duration:
        now = time.time() - start
        if now - last_print >= 0.3:
            antennas = mini.get_present_antenna_joint_positions()
            remaining = duration - now
            print(f"t={now:5.1f}s ({remaining:4.1f}s left)  antennas={np.round(antennas, 3)}")
            last_print = now
        time.sleep(0.03)

    mini.enable_motors()
    print("#" * 60)
    print("#  MOTORS RE-ENABLED - robot back under control          #")
    print("#" * 60)
    print("Touch test done. Look at the antennas numbers above while you moved it - did they change?")


def test_imu(mini: ReachyMini) -> None:
    try:
        imu_data = mini.imu
    except Exception as e:
        print(f"IMU not available on this platform (expected on Reachy Mini Lite): {e}")
        return
    print("IMU data:", imu_data)


TESTS = {
    "movement": test_movement,
    "camera": test_camera,
    "speaker": test_speaker,
    "mic": test_mic,
    "touch": test_touch,
    "imu": test_imu,
}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in TESTS:
        print(f"Usage: python hw_test.py <{'|'.join(TESTS)}>")
        sys.exit(1)

    name = sys.argv[1]
    with ReachyMini() as mini:
        TESTS[name](mini)


if __name__ == "__main__":
    main()
