"""Reachy Mini app-store entry point.

The dashboard starts this module (`python -m reachy_mini_gestures.main`) and
stops it with Ctrl+C / SIGINT. The gesture app manages its own robot
connection (with automatic reconnects) and serves its own HTTPS page on port
8765, because phone cameras only work over HTTPS. So instead of the SDK's
default wrapper, which opens a second robot connection and a plain-HTTP
settings server, it simply runs the app until asked to stop.
"""

from __future__ import annotations

import threading

from reachy_mini import ReachyMini, ReachyMiniApp


class ReachyMiniGestures(ReachyMiniApp):
    # Opened by the dashboard's app button. HTTPS with a self-signed
    # certificate: the browser asks once to trust it.
    custom_app_url: str | None = "https://localhost:8765"
    dont_start_webserver: bool = True

    def wrapped_run(self, *args, **kwargs) -> None:
        self.run(None, self.stop_event)

    def run(self, reachy_mini: ReachyMini | None, stop_event: threading.Event) -> None:
        from . import gesture_web_app as app

        def relay_stop() -> None:
            stop_event.wait()
            app.stop_event.set()

        threading.Thread(target=relay_stop, name="app-stop-relay", daemon=True).start()
        app.main()


if __name__ == "__main__":
    gestures = ReachyMiniGestures()
    try:
        gestures.wrapped_run()
    except KeyboardInterrupt:
        gestures.stop()
