"""Local launcher kept for manage.ps1: runs the packaged gesture app.

Installed users can run `reachy-mini-gestures` or start it from the Reachy
Mini dashboard instead.
"""

from reachy_mini_gestures.gesture_web_app import main

if __name__ == "__main__":
    main()
