from pathlib import Path
import numpy as np
import pandas as pd
import torch
import librosa
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model

MODEL_NAME = "facebook/wav2vec2-base"

LABEL2ID = {"angry": 0, "neutral": 1, "sad": 2}

def load_audio(path, sr=16000):
    y, _ = librosa.load(path, sr=sr, mono=True)
    # avoid empty
    if y is None or len(y) == 0:
        y = np.zeros(sr, dtype=np.float32)
    return y.astype(np.float32)

@torch.inference_mode()
def main():
    device = torch.device("cpu")

    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    out_dir = Path("data/features")
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        csv_path = Path(f"data/{split}.csv")
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing {csv_path}. Run make_splits.py first.")

        df = pd.read_csv(csv_path)
        X_list = []
        y_list = []

        print(f"\nProcessing split: {split} ({len(df)} files)")

        for _, row in tqdm(df.iterrows(), total=len(df)):
            wav_path = row["path"]
            label = row["label"]
            if label not in LABEL2ID:
                continue

            y = load_audio(wav_path, sr=16000)

            inputs = processor(y, sampling_rate=16000, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(device)

            outputs = model(input_values)
            # outputs.last_hidden_state: [B, T, H]
            emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()  # [H]

            X_list.append(emb)
            y_list.append(LABEL2ID[label])

        X = np.stack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.int64)

        np.save(out_dir / f"X_{split}.npy", X)
        np.save(out_dir / f"y_{split}.npy", y)

        print(f"✅ Saved {split}: X_{split}.npy {X.shape}, y_{split}.npy {y.shape}")

if __name__ == "__main__":
    main()
