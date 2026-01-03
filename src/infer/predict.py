import numpy as np
import torch
import librosa
import joblib
from transformers import Wav2Vec2Processor, Wav2Vec2Model

MODEL_NAME = "facebook/wav2vec2-base"
ID2LABEL = {0: "angry", 1: "neutral", 2: "sad"}

class EmotionPredictor:
    def __init__(self, clf_path="models/emotion_clf.joblib"):
        self.device = torch.device("cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        self.wav2vec = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(self.device)
        self.wav2vec.eval()
        self.clf = joblib.load(clf_path)

    def _audio_to_embedding(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        # Ensure mono float32 and correct SR
        if audio.ndim > 1:
            audio = audio.squeeze()
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        audio = audio.astype(np.float32)
        if len(audio) == 0:
            audio = np.zeros(sr, dtype=np.float32)

        with torch.inference_mode():
            inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
            x = inputs.input_values.to(self.device)
            out = self.wav2vec(x)
            emb = out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()  # (768,)
        return emb

    def predict(self, audio: np.ndarray, sr: int = 16000):
        emb = self._audio_to_embedding(audio, sr=sr).reshape(1, -1)
        probs = self.clf.predict_proba(emb)[0]
        pred_id = int(np.argmax(probs))
        return ID2LABEL[pred_id], float(probs[pred_id]), probs
