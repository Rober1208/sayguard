"""Camera sources.  Each source runs a reader thread and keeps only the latest
frame, so a slow consumer drops frames instead of building up latency.

Three kinds, told apart by the source string, because they fail differently:
- camera index ("0"): live; a read failure means the device is gone, so the
  source goes offline and retries with a growing back-off;
- network stream (anything with "://": rtsp://, http://, ...): live like a
  camera.  It is read as fast as it delivers (pacing would make FFmpeg queue
  frames, i.e. latency), the capture buffer is kept to one frame, and open/read
  timeouts stop a dead host from blocking the reader for FFmpeg's default 30 s.
  That matters beyond the one camera: OpenCV serialises every FFmpeg open in the
  process behind one lock, so a dead host's open stalls the other sources' opens;
- video file: the demo's stand-in for a camera.  It loops forever at its own
  fps, rewinding in place rather than reopening (no trip through that lock); a
  file that opens but yields no frame (corrupt, still being written) backs off
  instead of reopening in a tight loop."""
from __future__ import annotations

import logging
import re
import threading
import time

import cv2

from .types import FramePacket

log = logging.getLogger("sayguard.sources")

STREAM_TIMEOUT_MS = 5000     # open and read timeout for network streams
MAX_BACKOFF_S = 10.0         # live sources: 1, 2, 4, 8, 10, 10 ... s between reconnects
FILE_RETRY_S = 2.0           # a file that does not open (missing, locked)
FILE_EMPTY_RETRY_S = 1.0     # a file that opens but gives no frame


def is_network(source: str | int) -> bool:
    return "://" in str(source)


def is_camera_index(source: str | int) -> bool:
    return isinstance(source, int) or str(source).isdigit()


def redact(source: str | int) -> str:
    """Hide credentials (rtsp://user:pass@host/...) before a URL goes into a log."""
    return re.sub(r"(://)[^/@]*@", r"\1***@", str(source))


def _stream_open_args() -> tuple[int, list[int]]:
    """(api, params) for network streams, limited to what the installed cv2 offers."""
    try:
        api = cv2.CAP_FFMPEG if cv2.videoio_registry.hasBackend(cv2.CAP_FFMPEG) else cv2.CAP_ANY
    except (AttributeError, cv2.error):
        api = cv2.CAP_ANY
    params: list[int] = []
    for name in ("CAP_PROP_OPEN_TIMEOUT_MSEC", "CAP_PROP_READ_TIMEOUT_MSEC"):
        if hasattr(cv2, name):
            params += [getattr(cv2, name), STREAM_TIMEOUT_MS]
    return api, params


class VideoSource:
    def __init__(self, cam_id: str, source: str | int):
        self.cam_id = cam_id
        self.source = source
        self.kind = "camera" if is_camera_index(source) else "stream" if is_network(source) else "file"
        self._latest: FramePacket | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"src-{cam_id}", daemon=True)
        self.online = False
        self.native_fps = 0.0

    @property
    def live(self) -> bool:
        return self.kind != "file"

    def start(self) -> "VideoSource":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def latest(self) -> FramePacket | None:
        with self._lock:
            return self._latest

    def _open(self) -> cv2.VideoCapture:
        if self.kind == "camera":
            cap = cv2.VideoCapture(int(self.source))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap
        if self.kind == "file":
            return cv2.VideoCapture(str(self.source))
        api, params = _stream_open_args()
        cap = None
        if params:
            try:
                cap = cv2.VideoCapture(str(self.source), api, params)
            except cv2.error:   # backend rejects the parameters: open without timeouts
                cap = None
        if cap is None:
            cap = cv2.VideoCapture(str(self.source), api)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # newest frame only, where the backend supports it
        return cap

    def _wait(self, seconds: float) -> None:
        self._stop.wait(seconds)    # returns at once on stop()

    def _run(self) -> None:
        idx = 0
        backoff = 1.0
        down = False                # log an outage once, not on every retry
        while not self._stop.is_set():
            cap = self._open()
            if not cap.isOpened():
                cap.release()
                self.online = False
                if not down:
                    log.warning("%s: cannot open %s source %s", self.cam_id, self.kind, redact(self.source))
                    down = True
                if self.live:
                    self._wait(backoff)
                    backoff = min(MAX_BACKOFF_S, backoff * 2)
                else:
                    self._wait(FILE_RETRY_S)
                continue
            self.online = True
            fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
            self.native_fps = fps if 1 <= fps <= 120 else 10.0
            period = 1.0 / self.native_fps
            nxt = time.perf_counter()
            got = run = 0           # frames since open / since the last rewind
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    if self.kind == "file" and run and cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                        run = 0     # end of file: loop by rewinding
                        continue
                    break  # file cannot rewind -> reopen; live source error -> reconnect
                got += 1
                run += 1
                idx += 1
                if down:
                    log.info("%s: %s source is delivering frames again", self.cam_id, self.kind)
                    down = False
                with self._lock:
                    self._latest = FramePacket(self.cam_id, idx, time.time(), frame)
                if self.kind == "file":
                    nxt += period
                    delay = nxt - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        nxt = time.perf_counter()
            cap.release()
            if self._stop.is_set():
                break
            if got == 0 and not down:
                log.warning("%s: %s source %s opens but yields no frame", self.cam_id, self.kind, redact(self.source))
                down = True
            if self.live:
                self.online = False
                if got:
                    backoff = 1.0       # it was delivering: try again soon
                    log.warning("%s: %s source stopped delivering frames; reconnecting", self.cam_id, self.kind)
                self._wait(backoff)
                backoff = min(MAX_BACKOFF_S, backoff * 2)
            elif got == 0:
                self.online = False
                self._wait(FILE_EMPTY_RETRY_S)
