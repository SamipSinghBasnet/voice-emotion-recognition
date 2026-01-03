import os, sys
sys.path.insert(0, os.path.abspath("."))

import io
import json
import time
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import streamlit as st
import soundfile as sf
import librosa
import joblib

from src.infer.predict import EmotionPredictor

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Voice Emotion Recognition", layout="centered")

SR = 16000
LABELS = ["angry", "neutral", "sad"]

# -----------------------------
# Helpers
# -----------------------------
def softmax_like(probs: np.ndarray) -> dict:
    return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

def safe_read_audio(file_bytes: bytes):
    audio, sr = sf.read(io.BytesIO(file_bytes))
    # to mono
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        sr = SR
    return audio, sr

def compute_metrics_if_possible():
    """
    Tries to compute metrics from committed feature arrays.
    If not present, tries to load precomputed reports/metrics.json
    """
    # Option A: precomputed metrics JSON (recommended for cloud)
    metrics_json = Path("reports/metrics.json")
    if metrics_json.exists():
        with open(metrics_json, "r", encoding="utf-8") as f:
            return json.load(f), "Loaded precomputed reports/metrics.json"

    # Option B: compute from data/features if you committed them
    feat_dir = Path("data/features")
    model_path = Path("models/emotion_clf.joblib")
    required = [
        feat_dir / "X_test.npy",
        feat_dir / "y_test.npy",
        model_path,
    ]
    if all(p.exists() for p in required):
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

        X_test = np.load(feat_dir / "X_test.npy")
        y_test = np.load(feat_dir / "y_test.npy")
        clf = joblib.load(model_path)

        pred = clf.predict(X_test)
        acc = float(accuracy_score(y_test, pred))
        macro_f1 = float(f1_score(y_test, pred, average="macro"))
        cm = confusion_matrix(y_test, pred).tolist()
        report = classification_report(
            y_test,
            pred,
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        )

        payload = {
            "test_accuracy": acc,
            "test_macro_f1": macro_f1,
            "confusion_matrix": cm,
            "classification_report": report,
            "note": "Computed from data/features/*_test.npy",
        }
        return payload, "Computed metrics from data/features (X_test/y_test)"

    return None, "No metrics found"

# -----------------------------
# Load model (cached)
# -----------------------------
st.title("🎙️ Voice Emotion Recognition (Cloud Demo)")
st.caption("3-class: Angry / Neutral / Sad • Browser recording • CPU inference")

@st.cache_resource
def load_predictor():
    return EmotionPredictor("models/emotion_clf.joblib")

predictor = load_predictor()

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Settings")
smooth_n = st.sidebar.slider("Smoothing window (#recent recordings)", 1, 10, 5, 1)
uncertain_thresh = st.sidebar.slider("Uncertain threshold", 0.30, 0.90, 0.50, 0.05)

if "history" not in st.session_state:
    # store last N predictions (timestamp, label, conf)
    st.session_state.history = deque(maxlen=30)

# smoothing buffer
if "recent" not in st.session_state:
    st.session_state.recent = deque(maxlen=smooth_n)
else:
    st.session_state.recent = deque(st.session_state.recent, maxlen=smooth_n)

# -----------------------------
# Tabs
# -----------------------------
tab_demo, tab_how, tab_metrics = st.tabs(["✅ Demo", "🧠 How it works", "📊 Metrics"])

# ========== TAB 1: DEMO ==========
with tab_demo:
    st.subheader("Record and Predict")

    audio_file = st.audio_input("Record your voice (1–5 seconds recommended)")

    if audio_file is None:
        st.info("Click the microphone above, record a short clip, then the app will predict emotion.")
    else:
        audio_bytes = audio_file.read()
        audio, sr = safe_read_audio(audio_bytes)

        # show playback
        st.audio(audio_bytes, format="audio/wav")

        # predict
        label, conf, probs = predictor.predict(audio, sr=sr)
        st.session_state.recent.append((label, conf))

        # smooth across recent recordings
        recent_labels = [x[0] for x in st.session_state.recent]
        recent_confs = [x[1] for x in st.session_state.recent]

        # mode label
        smoothed = max(set(recent_labels), key=recent_labels.count)
        avg_conf = float(np.mean(recent_confs))

        final_label = smoothed if avg_conf >= uncertain_thresh else "uncertain"

        # history
        st.session_state.history.append((time.time(), final_label, avg_conf))

        # UI blocks
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("### Prediction")
            st.markdown(f"**Raw:** `{label}`  (conf: **{conf:.2f}**)")
            st.markdown(f"**Smoothed:** `{final_label}`  (avg conf: **{avg_conf:.2f}**)")
            st.progress(min(max(avg_conf, 0.0), 1.0))

        with c2:
            st.markdown("### Probabilities (raw)")
            prob_dict = softmax_like(probs)
            prob_df = pd.DataFrame({"probability": prob_dict}).T
            st.bar_chart(prob_df.T)

        st.markdown("### Waveform preview")
        show_n = min(len(audio), SR * 3)  # up to 3 seconds
        st.line_chart(audio[:show_n])

        st.markdown("### Recent history")
        if len(st.session_state.history) == 0:
            st.write("No history yet.")
        else:
            hist_df = pd.DataFrame(
                list(st.session_state.history),
                columns=["timestamp", "label", "confidence"]
            )
            hist_df["time"] = pd.to_datetime(hist_df["timestamp"], unit="s").dt.strftime("%H:%M:%S")
            st.dataframe(hist_df[["time", "label", "confidence"]], use_container_width=True)

            # download report
            report = {
                "last_prediction": {
                    "raw_label": label,
                    "raw_confidence": float(conf),
                    "raw_probabilities": prob_dict,
                    "smoothed_label": final_label,
                    "smoothed_confidence": float(avg_conf),
                },
                "settings": {
                    "smoothing_window": smooth_n,
                    "uncertain_threshold": float(uncertain_thresh),
                },
            }
            st.download_button(
                "⬇️ Download prediction report (JSON)",
                data=json.dumps(report, indent=2).encode("utf-8"),
                file_name="prediction_report.json",
                mime="application/json",
            )

# ========== TAB 2: HOW IT WORKS ==========
with tab_how:
    st.subheader("System overview (research-style)")

    st.markdown(
        """
**Goal:** Predict emotion from speech audio in a **speaker-independent** setting.

**Pipeline**
1. **Audio Input** (browser recording → resampled to 16 kHz mono)
2. **Feature Extraction:** `Wav2Vec2 (facebook/wav2vec2-base)` generates deep speech embeddings  
   - We average over time to get a fixed vector (768-d)
3. **Classifier:** Logistic Regression predicts **Angry / Neutral / Sad**
4. **Uncertainty Handling:** If confidence < threshold → output `uncertain`
5. **Temporal Smoothing:** majority vote over the last N recordings (reduces flicker)

**Why this is credible**
- Uses **speaker-independent split** to avoid leakage (important for real research)
- Uses **self-supervised speech representation (wav2vec2)**, strong baseline on CPU
- Provides confidence + uncertainty handling (responsible ML behavior)
        """
    )

    st.markdown("---")
    st.subheader("Local vs Cloud")
    st.markdown(
        """
**Cloud demo:** uses Streamlit browser recording (`st.audio_input`) for maximum portability.  
**Local demo:** can run true continuous streaming with `sounddevice` callback.

This is an engineering tradeoff: cloud environments cannot access the user's microphone device directly.
        """
    )

# ========== TAB 3: METRICS ==========
with tab_metrics:
    st.subheader("Evaluation Metrics")

    metrics, msg = compute_metrics_if_possible()
    st.info(msg)

    if metrics is None:
        st.warning(
            """
No metrics file found in this deployment.

**Best practice (recommended):** generate metrics locally and commit `reports/metrics.json`.
This keeps the cloud app lightweight while still showing your real evaluation results.
            """
        )

        st.code(
            """# Create reports/metrics.json locally (recommended)
python src/train/train_classifier.py  # or create a small metrics script
# then save metrics to: reports/metrics.json and git push""",
            language="bash",
        )
    else:
        # Show headline numbers
        c1, c2 = st.columns(2)
        with c1:
            if "test_accuracy" in metrics:
                st.metric("Test Accuracy", f"{metrics['test_accuracy']:.3f}")
        with c2:
            if "test_macro_f1" in metrics:
                st.metric("Test Macro-F1", f"{metrics['test_macro_f1']:.3f}")

        # Confusion matrix
        cm = metrics.get("confusion_matrix")
        if cm is not None:
            st.markdown("### Confusion Matrix (test)")
            cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
            st.dataframe(cm_df, use_container_width=True)

        # Classification report
        report = metrics.get("classification_report")
        if isinstance(report, dict):
            st.markdown("### Classification Report (test)")
            # show per-class precision/recall/f1
            rows = []
            for lab in LABELS:
                if lab in report:
                    rows.append({
                        "class": lab,
                        "precision": report[lab].get("precision"),
                        "recall": report[lab].get("recall"),
                        "f1": report[lab].get("f1-score"),
                        "support": report[lab].get("support"),
                    })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.markdown("### Download metrics")
        st.download_button(
            "⬇️ Download metrics JSON",
            data=json.dumps(metrics, indent=2).encode("utf-8"),
            file_name="metrics.json",
            mime="application/json",
        )
