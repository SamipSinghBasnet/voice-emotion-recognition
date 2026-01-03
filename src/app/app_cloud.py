import time
from collections import deque, Counter

import numpy as np
import streamlit as st
import av
import librosa
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

from src.infer.predict import EmotionPredictor

st.set_page_config(page_title="Voice Emotion (Web Demo)", layout="centered")
st.title("🎙️ Voice Emotion Recognition (Web Demo)")
st.caption("In-browser recording via WebRTC • 3-class (Angry / Neutral / Sad) • CPU inference")

SR = 16000  # model expects 16k

@st.cache_resource
def load_predictor():
    return EmotionPredictor("models/emotion_clf.joblib")

predictor = load_predictor()

st.sidebar.header("Streaming Settings")
win_sec = st.sidebar.slider("Window length (seconds)", 0.8, 3.0, 1.5, 0.1)
hop_sec = st.sidebar.slider("Update every (seconds)", 0.2, 1.0, 0.5, 0.1)
smooth_n = st.sidebar.slider("Smoothing window (#updates)", 1, 15, 5, 1)
uncertain_thresh = st.sidebar.slider("Uncertain threshold", 0.30, 0.90, 0.50, 0.05)
history_len = st.sidebar.slider("History length (#updates)", 10, 120, 60, 5)

def mode_label(labels):
    return Counter(labels).most_common(1)[0][0]

# Shared state (lives across reruns)
if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=history_len)  # (t, label, conf)
if "recent" not in st.session_state:
    st.session_state.recent = deque(maxlen=smooth_n)      # (label, conf)
if "last_update" not in st.session_state:
    st.session_state.last_update = 0.0

# --- Audio Processor ---
class EmotionAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.buffer = deque(maxlen=int(SR * 6))  # keep last ~6s
        self.last_pred = None

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Convert incoming audio frame -> mono float32 numpy
        audio = frame.to_ndarray()  # shape: (channels, samples) or (samples,) depending
        if audio.ndim == 2:
            # (channels, samples) -> take first channel
            audio = audio[0]
        audio = audio.astype(np.float32)

        # WebRTC often gives 48kHz; we resample to 16k for the model
        in_sr = frame.sample_rate if frame.sample_rate else 48000
        if in_sr != SR:
            audio = librosa.resample(audio, orig_sr=in_sr, target_sr=SR)

        # push into ring buffer
        self.buffer.extend(audio.tolist())
        return frame

audio_processor = EmotionAudioProcessor()

# --- WebRTC component ---
ctx = webrtc_streamer(
    key="voice-emotion-webrtc",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=256,
    media_stream_constraints={"audio": True, "video": False},
    rtc_configuration={  # public STUN server
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    audio_processor_factory=lambda: audio_processor,
)

status = st.empty()
current = st.empty()
probs_box = st.empty()
timeline = st.empty()

if ctx.state.playing:
    status.success("Mic connected. Speak and watch predictions update.")
else:
    status.info("Click Start to enable microphone.")

def get_window_from_buffer(buf: deque, seconds: float) -> np.ndarray | None:
    n = int(seconds * SR)
    if len(buf) < n:
        return None
    x = np.array(list(buf)[-n:], dtype=np.float32)
    return x

# Inference loop (runs on rerun cadence)
if ctx.state.playing:
    now = time.time()
    if now - st.session_state.last_update >= hop_sec:
        st.session_state.last_update = now

        window = get_window_from_buffer(audio_processor.buffer, win_sec)
        if window is not None:
            label, conf, probs = predictor.predict(window, sr=SR)

            st.session_state.recent.append((label, conf))
            labels = [x[0] for x in st.session_state.recent]
            confs = [x[1] for x in st.session_state.recent]

            smoothed = mode_label(labels)
            avg_conf = float(np.mean(confs))
            display = smoothed if avg_conf >= uncertain_thresh else "uncertain"

            st.session_state.history.append((time.time(), display, avg_conf))

            # UI
            current.subheader("Current")
            current.write(f"**Raw:** {label} (conf {conf:.2f})")
            current.write(f"**Smoothed:** {display} (avg conf {avg_conf:.2f})")

            probs_box.subheader("Probabilities (raw)")
            probs_box.write({
                "angry": float(probs[0]),
                "neutral": float(probs[1]),
                "sad": float(probs[2]),
            })

# History visualization
timeline.subheader("Recent history")
if len(st.session_state.history) == 0:
    timeline.write("No predictions yet.")
else:
    labels = [x[1] for x in st.session_state.history]
    counts = Counter(labels)
    timeline.write(dict(counts))

# Auto-refresh while playing to keep inference going
if ctx.state.playing:
    time.sleep(0.2)
    st.rerun()
