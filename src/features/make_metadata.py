from pathlib import Path
import pandas as pd

# We will keep ONLY these 3 emotions
EMO_MAP = {
    "ANG": "angry",
    "NEU": "neutral",
    "SAD": "sad",
}

def parse_cremad_filename(name: str):
    """
    CREMA-D filename format:
    SpeakerID_SentenceID_Emotion_Intensity.wav
    Example: 1001_DFA_ANG_XX.wav
    """
    parts = name.split("_")
    if len(parts) < 4:
        return None
    speaker_id = parts[0]
    emotion_code = parts[2]
    if emotion_code not in EMO_MAP:
        return None
    return speaker_id, EMO_MAP[emotion_code], emotion_code

def main():
    base = Path("data/raw/CREMA-D")

    if not base.exists():
        raise FileNotFoundError(f"Folder not found: {base.resolve()}")

    # Find wav files even if you accidentally have AudioWAV/AudioWAV nested
    wav_files = sorted(base.rglob("*.wav"))

    if not wav_files:
        raise FileNotFoundError(
            f"No .wav files found under {base.resolve()}. "
            "Check your dataset path."
        )

    rows = []
    skipped = 0

    for f in wav_files:
        parsed = parse_cremad_filename(f.stem)
        if parsed is None:
            skipped += 1
            continue
        speaker_id, label, emotion_code = parsed
        rows.append(
            {
                "path": str(f.as_posix()),
                "label": label,
                "emotion_code": emotion_code,
                "speaker_id": speaker_id,
            }
        )

    df = pd.DataFrame(rows)

    out_path = Path("data/metadata.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"✅ Saved: {out_path.resolve()}")
    print(f"Total wav files found: {len(wav_files)}")
    print(f"Kept (ANG/NEU/SAD): {len(df)}")
    print(f"Skipped (other emotions or bad names): {skipped}")

    print("\nClass counts:")
    print(df["label"].value_counts())

    print("\nUnique speakers kept:", df["speaker_id"].nunique())

if __name__ == "__main__":
    main()
