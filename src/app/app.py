import time
import streamlit as st
import numpy as np

from src.infer.predict import EmotionPredictor
from src.app.live_stream import LiveStreamer

st.set_page_config(page_title="Voice Emotion (True Live)", layout="centered")
st.title("🎙️ Voice Emotion Recognition — True Live Streaming")

sr = 16000

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

# Create / update streamer in session state
if "streamer" not in st.session_state:
    st.session_state.streamer = LiveStreamer(
        predictor=predictor,
        sr=sr,
        window_sec=win_sec,
        hop_sec=hop_sec,
        smooth_n=smooth_n,
        uncertain_thresh=uncertain_thresh,
        history_len=history_len,
    )

# If settings changed, recreate streamer (safe)
if st.button("Apply settings (recreate stream)"):
    try:
        st.session_state.streamer.stop()
    except Exception:
        pass
    st.session_state.streamer = LiveStreamer(
        predictor=predictor,
        sr=sr,
        window_sec=win_sec,
        hop_sec=hop_sec,
        smooth_n=smooth_n,
        uncertain_thresh=uncertain_thresh,
        history_len=history_len,
    )
    st.success("Settings applied. Click Start.")

col1, col2 = st.columns(2)

with col1:
    if st.button("▶ Start"):
        st.session_state.streamer.start()
        st.session_state.running = True

with col2:
    if st.button("⏹ Stop"):
        st.session_state.streamer.stop()
        st.session_state.running = False

status = st.empty()
current = st.empty()
probs_box = st.empty()
timeline = st.empty()

running = st.session_state.get("running", False)

if running:
    status.info("Live streaming ON (callback-based). Speak into the mic.")
else:
    status.warning("Live streaming OFF. Click Start.")

# Auto-refresh UI while running
if running:
    # refresh rate for UI only (not audio). 5 updates/sec looks smooth.
    time.sleep(0.2)
    st.rerun()

# Display latest predictions (works even when stopped)
s = st.session_state.streamer
current.subheader("Current")
if s.last_raw is None:
    current.write("Waiting for audio… (talk for a second)")
else:
    raw_label, raw_conf = s.last_raw
    current.write(f"**Raw:** {raw_label} (conf {raw_conf:.2f})")
    current.write(f"**Smoothed:** {s.last_smoothed} (avg conf {s.last_conf:.2f})")

probs_box.subheader("Probabilities (raw)")
if s.last_probs is None:
    probs_box.write({"angry": None, "neutral": None, "sad": None})
else:
    probs_box.write({
        "angry": float(s.last_probs[0]),
        "neutral": float(s.last_probs[1]),
        "sad": float(s.last_probs[2]),
    })

timeline.subheader("Recent history")
if len(s.history) == 0:
    timeline.write("No history yet.")
else:
    labels = [x[1] for x in s.history]
    unique, counts = np.unique(labels, return_counts=True)
    timeline.write(dict(zip(unique.tolist(), counts.tolist())))
