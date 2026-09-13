---
title: Reachy Mini Gestures
emoji: 👋
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
license: apache-2.0
short_description: Control Reachy Mini with hand gestures
tags:
 - reachy_mini
 - reachy_mini_python_app
 - gestures
 - mediapipe
---

# Reachy Mini Gestures

Wave, point, give a thumbs up, and Reachy Mini reacts with its head, body and
antennas. Hands are recognized with MediaPipe through the robot's own camera, a
laptop webcam, or a phone you stand in front of the robot.

> **Where it runs:** this app drives a real robot, so it runs on the computer
> connected to your Reachy Mini (or on the robot itself), not on Hugging Face's
> servers. The Hugging Face Space is where the Reachy Mini dashboard downloads
> it from; GitHub hosts the same source code.
>
> - Hugging Face: <https://huggingface.co/spaces/sidbha1980/reachy-mini-gestures>
> - GitHub: <https://github.com/sidbha-del/reachy-mini-gestures>

## Gestures

| Gesture | How | Reachy |
|---|---|---|
| ☝️ Point | Point one finger (palm or back of hand) | Head and body follow your finger |
| ☝️☝️ Two pointers | Point up with both hands, tilt your fingers | Antennas copy your finger angles, body turns to you |
| ✋ Hello | Open palm, still or waving | Waves back |
| ✊ Fist | Make a fist | Gets shy |
| 👍 / 👎 Thumbs | Thumb up or down | Happy nod / sad head shake |
| ✌️ Peace | Index and middle finger up | Body wiggle |
| 🤟 / 🫰 Love you | Thumb, index and pinky out, or a Korean finger heart | Little dance |
| 👌 Okay | Thumb and index circle, other fingers up | Curious head tilt |
| 👉 💥 Bang! | Finger gun: index out, thumb up, other fingers folded | Falls down dead, then slowly comes back |
| 🙌 Hands up | Both open palms, then raise them | Stretches up tall |
| ✊✊ → ✋✋ Ta-da | Two fists, then open both hands | Surprise pop |
| 🙏 Namaste | Press your palms together in front of your chest, fingers up | Bows with antennas leaning together |
| ✋ → 📷 Too close | Hold a hand right in front of the camera, or push a palm toward it | Startled jolt back, antennas wiggle, “uh-uh-uh” no-no-no sound |

With two hands in view the app is in two-hand mode, so single-hand reactions
never leak out of a ta-da or two-pointer gesture. Motion gestures are ignored
while Reachy itself (or a hand-held phone) is moving, so the robot never reacts
to its own movement.

## What you need

- A **Reachy Mini** (developed and tested on **Reachy Mini Lite** over USB;
  Wireless should work but is not tested yet, see [Troubleshooting](#troubleshooting)).
- A computer with **Python 3.10+** and the **Reachy Mini SDK / daemon**
  (`pip install reachy-mini`), connected to the robot.
- A modern browser (Chrome, Edge, Safari or Firefox).
- Optional: a phone on the same Wi-Fi to use as a faster camera.

## Install

### Option 1: from Hugging Face, using the Reachy Mini dashboard (easiest)

1. Turn on Reachy Mini and open the Reachy Mini dashboard (it starts the daemon).
2. Open the app list and search for **reachy_mini_gestures**.
3. Press **Install**, then **Start**. The first install downloads MediaPipe and
   OpenCV, so it can take a few minutes.
4. Open <https://localhost:8765> on the same computer (see [First run](#first-run)).

### Option 2: from GitHub (source)

```bash
git clone https://github.com/sidbha-del/reachy-mini-gestures
cd reachy-mini-gestures
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

pip install -e .
```

Start the robot daemon in one terminal, then the app in another:

```bash
reachy-mini-daemon            # talks to the robot (leave it running)
reachy-mini-gestures          # starts the gesture app
```

On Windows you can instead run `.\manage.ps1` from the project folder for a
small menu that checks, starts, stops and restarts both. It expects the virtual
environment in a folder named `reachy_mini_env`; edit the paths at the top of
the script if yours is different.

### Option 3: from the Hugging Face repository with git

The Space is also a normal git repository (the gesture model is stored with
Git LFS, so install [git-lfs](https://git-lfs.com) first):

```bash
git lfs install
git clone https://huggingface.co/spaces/sidbha1980/reachy-mini-gestures
pip install ./reachy-mini-gestures
reachy-mini-gestures
```

`pip install git+https://huggingface.co/spaces/...` does **not** work, because
Hugging Face doesn't support the partial clone pip uses. Clone first, as above.

## First run

1. Open <https://localhost:8765>.
2. Your browser warns about the certificate. This is expected: the app creates
   its own self-signed certificate on first run, because phone cameras only work
   over HTTPS. Choose **Advanced → Proceed** (Chrome/Edge) or **Show details →
   visit this website** (Safari).
3. The top bar shows **Connected** once the app reaches the robot. Pick a camera
   and try pointing at something.
4. On Windows, allow Python through the firewall on **private networks** when
   asked, or phones won't be able to connect.

## Cameras

- **Robot**: no setup. On Reachy Mini Lite the camera, speaker and motors share
  one USB 2.0 connection, so it runs at about 10 fps.
- **Laptop**: a built-in or USB webcam at up to 30 fps.
- **Phone**:
  1. Select **Phone** in the app; a QR code appears.
  2. Scan it with the phone (same Wi-Fi as the computer) and accept the
     certificate warning on the phone too.
  3. Allow camera access.
  4. Stand the phone **just in front of Reachy, facing you, in landscape, tilted
     up a little**; the pairing screen shows an animation of the best spot.
  5. Optional: in **Settings**, turn on **Pause robot camera** to free the
     Lite's USB link for smoother motion and sound.

Reaction sounds can play on Reachy's speaker or on the computer (including any
Bluetooth speaker paired with it): **Settings → Play sounds on**.

## Teach (optional)

Built-in recognition works without training. **Teach** lets you add examples of
your own hands for any gesture (two-hand gestures capture both hands; Ta-da and
Love you have two steps). Close matches to your examples win over the built-in
model, which helps with gestures like Okay or the finger heart. Save 4–5
examples per gesture, facing the camera the same way you'll use it.

### Your data stays on your computer

Taught examples, settings and the HTTPS certificate are stored locally and are
never uploaded:

- running from a source checkout: in the project folder
  (`teach_samples.json`, `app_settings.json`, `camera_source.json`, `cert.pem`, `key.pem`);
- installed from the dashboard or pip: in `~/.reachy_mini_gestures/`;
- or any folder you choose with the `REACHY_GESTURES_DATA` environment variable.

These files are listed in `.gitignore`. If you publish your own copy, keep it
that way: they contain recordings of your hand shapes and your private key.

## Make your own copy

**On GitHub:** press **Fork**, clone your fork, and push your changes. Before
your first commit, set an identity you're happy to make public, for example
GitHub's private no-reply email (GitHub → Settings → Emails → *Keep my email
address private*):

```bash
git config user.name "your-github-username"
git config user.email "ID+your-github-username@users.noreply.github.com"
```

**On Hugging Face:** open the Space, choose **⋮ → Duplicate this Space**, or
publish from your own checkout with the Reachy Mini app assistant:

```bash
hf auth login                                  # once
reachy-mini-app-assistant check .              # validates the app layout
reachy-mini-app-assistant publish . "My changes"
```

The app store needs `pyproject.toml` (with the `reachy_mini_apps` entry
point), the `reachy_mini_gestures/` package with `main.py`, `README.md` with the
`reachy_mini` and `reachy_mini_python_app` tags, and `index.html` + `style.css`
at the root; keep those when you rename or change things. On Windows run the
checker with `PYTHONUTF8=1` set, or it fails to read this README's emoji.

## Development

```bash
python tests/test_logic.py      # offline tests, no robot needed
```

Every reaction logs a `[gesture]` line, and <https://localhost:8765/status>
reports how often motion gestures were suppressed and why, which helps tuning.

## Troubleshooting

| Problem | What to do |
|---|---|
| Browser says the connection isn't private | Expected on first visit: proceed past the warning (see [First run](#first-run)). |
| Page doesn't open | Make sure the app is running and nothing else uses port 8765. |
| "Robot not connected" | Start the Reachy Mini daemon (or the dashboard) and check the USB cable and power. The app reconnects on its own. |
| Phone can't open the link | Same Wi-Fi as the computer, allow Python on private networks in the firewall, accept the certificate on the phone. |
| Phone video freezes or is slow | Move closer to the router, or use the laptop camera. Keep the phone app in the foreground. |
| Reactions feel slow on the robot camera | Use the laptop or phone camera and turn on **Pause robot camera**. |
| A gesture isn't recognized | Good light, hand clearly in view and not at the very bottom of the frame; then add 4–5 examples in **Teach**. |
| Hands are shown as "Ignored" | The label says why: too far, cut off, resting at the bottom, or turned sideways. |
| No sound from Reachy | **Settings → Play sounds on → This computer** plays through the computer or a Bluetooth speaker. |
| An antenna stopped moving | A servo overload switches the motor off. Free the antenna, power-cycle the robot, then restart the app. |
| Reachy Mini Wireless | Not tested yet. Apps run on the robot's own computer there, which may be too slow for MediaPipe; reports are welcome in GitHub issues. |

## License

Apache-2.0. Built on the [Reachy Mini SDK](https://github.com/pollen-robotics/reachy_mini)
by Pollen Robotics and [MediaPipe](https://ai.google.dev/edge/mediapipe) by Google.
