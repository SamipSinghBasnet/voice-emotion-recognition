import time
import threading
from collections import deque, Counter
import numpy as np
import sounddevice as sd

# We will call your predictor like:
# label, conf, probs = predictor.predict(audio_np, sr=16000)

class LiveStreamer:
    """
    True streaming microphone capture using sounddevice callback.
    Maintains a ring buffer and runs inference on sliding windows.
    """

    def __init__(
        self,
        predictor,
        sr: int = 16000,
        window_sec: float = 1.5,
        hop_sec: float = 0.5,
        smooth_n: int = 5,
        uncertain_thresh: float = 0.5,
        history_len: int = 60,
        device=None,
    ):
        self.predictor = predictor
        self.sr = sr
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.smooth_n = smooth_n
        self.uncertain_thresh = uncertain_thresh
        self.history_len = history_len
        self.device = device

        self._buffer = deque(maxlen=int(sr * max(3.0, window_sec * 2)))  # enough room
        self._lock = threading.Lock()

        self._stop = threading.Event()
        self._thread = None
        self._stream = None

        self.recent = deque(maxlen=smooth_n)          # (label, conf)
        self.history = deque(maxlen=history_len)      # (timestamp, label, conf)
        self.last_probs = None
        self.last_raw = None
        self.last_smoothed = None
        self.last_conf = None

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            # you can log status if you want
            pass
        # indata shape: (frames, channels)
        x = indata[:, 0].astype(np.float32)
        with self._lock:
            self._buffer.extend(x.tolist())

    def _get_window(self) -> np.ndarray | None:
        n_win = int(self.window_sec * self.sr)
        with self._lock:
            if len(self._buffer) < n_win:
                return None
            # take latest n_win samples
            buf = list(self._buffer)
        window = np.array(buf[-n_win:], dtype=np.float32)
        return window

    @staticmethod
    def _mode_label(labels):
        return Counter(labels).most_common(1)[0][0]

    def _loop(self):
        next_t = time.time()
        while not self._stop.is_set():
            now = time.time()
            if now < next_t:
                time.sleep(min(0.05, next_t - now))
                continue

            audio = self._get_window()
            if audio is not None:
                label, conf, probs = self.predictor.predict(audio, sr=self.sr)

                self.last_raw = (label, conf)
                self.last_probs = probs

                self.recent.append((label, conf))
                labels = [x[0] for x in self.recent]
                confs = [x[1] for x in self.recent]

                smoothed = self._mode_label(labels)
                avg_conf = float(np.mean(confs))

                display_label = smoothed if avg_conf >= self.uncertain_thresh else "uncertain"

                self.last_smoothed = display_label
                self.last_conf = avg_conf

                self.history.append((time.time(), display_label, avg_conf))

            next_t += self.hop_sec

    def start(self):
        if self._stream is not None:
            return

        self._stop.clear()

        self._stream = sd.InputStream(
            samplerate=self.sr,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
            device=self.device,
            blocksize=0,  # let sounddevice choose
        )
        self._stream.start()

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        finally:
            self._stream = None
            self._thread = None
