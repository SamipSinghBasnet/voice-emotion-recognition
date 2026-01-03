from pathlib import Path
import pandas as pd
import numpy as np

def main(
    metadata_path="data/metadata.csv",
    out_dir="data",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
):
    metadata_path = Path(metadata_path)
    out_dir = Path(out_dir)

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found at: {metadata_path.resolve()}")

    df = pd.read_csv(metadata_path)

    # Basic checks
    required_cols = {"path", "label", "speaker_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv missing columns: {missing}")

    # Ensure ratios sum to 1.0
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    # Unique speakers
    speakers = df["speaker_id"].astype(str).unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(speakers)

    n = len(speakers)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    # rest goes to test to avoid rounding issues
    n_test = n - n_train - n_val

    train_speakers = set(speakers[:n_train])
    val_speakers = set(speakers[n_train:n_train + n_val])
    test_speakers = set(speakers[n_train + n_val:])

    # Safety check: no overlap
    assert train_speakers.isdisjoint(val_speakers)
    assert train_speakers.isdisjoint(test_speakers)
    assert val_speakers.isdisjoint(test_speakers)

    train_df = df[df["speaker_id"].astype(str).isin(train_speakers)].reset_index(drop=True)
    val_df   = df[df["speaker_id"].astype(str).isin(val_speakers)].reset_index(drop=True)
    test_df  = df[df["speaker_id"].astype(str).isin(test_speakers)].reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Print summary
    print("✅ Speaker-independent split created")
    print(f"Speakers total: {n}")
    print(f"Train speakers: {len(train_speakers)} | Val speakers: {len(val_speakers)} | Test speakers: {len(test_speakers)}")
    print(f"Rows  total: {len(df)}")
    print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)} | Test rows: {len(test_df)}")

    print("\nClass counts (train):")
    print(train_df["label"].value_counts())
    print("\nClass counts (val):")
    print(val_df["label"].value_counts())
    print("\nClass counts (test):")
    print(test_df["label"].value_counts())

    # Final verification: no speaker leakage
    leak = (set(train_df["speaker_id"]) & set(val_df["speaker_id"])) | \
           (set(train_df["speaker_id"]) & set(test_df["speaker_id"])) | \
           (set(val_df["speaker_id"]) & set(test_df["speaker_id"]))
    print("\nLeakage check (should be empty set):", leak)

if __name__ == "__main__":
    main()
