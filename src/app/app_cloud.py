import os, sys
sys.path.insert(0, os.path.abspath("."))


import streamlit as st
import numpy as np
import soundfile as sf
import librosa
import io
from collections import Counter, deque

from src.infer.predict import EmotionPredictor

st.set_page_config(page_title="Voice Emotion Recognition", layout="centered")
st.title("🎙️ Voice Emotion Recognition (Browser Recording)")

SR = 16000

@st.cache_resource
def load_model():
    return EmotionPredictor("models/emotion_clf.joblib")

model = load_model()

st.sidebar.header("Settings")
win_sec = st.sidebar.slider("Window length (seconds)", 1.0, 3.0, 1.5, 0.1)
smooth_n = st.sidebar.slider("Smoothing window", 1, 10, 5)
uncertain_thresh = st.sidebar.slider("Uncertain threshold", 0.3, 0.9, 0.5)

if "recent" not in st.session_state:
    st.session_state.recent = deque(maxlen=smooth_n)

audio_file = st.audio_input("Record your voice")

if audio_file is not None:
    audio_bytes = audio_file.read()
    audio, sr = sf.read(io.BytesIO(audio_bytes))
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)

    label, conf, probs = model.predict(audio, SR)
    st.session_state.recent.append((label, conf))

    labels = [x[0] for x in st.session_state.recent]
    confs = [x[1] for x in st.session_state.recent]

    final_label = Counter(labels).most_common(1)[0][0]
    avg_conf = float(np.mean(confs))

    if avg_conf < uncertain_thresh:
        final_label = "uncertain"

    st.subheader("Prediction")
    st.write(f"**Emotion:** {final_label}")
    st.write(f"**Confidence:** {avg_conf:.2f}")

    st.subheader("Class probabilities")
    st.json({
        "angry": float(probs[0]),
        "neutral": float(probs[1]),
        "sad": float(probs[2])
    })
