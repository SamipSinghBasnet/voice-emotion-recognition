from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib

ID2LABEL = {0: "angry", 1: "neutral", 2: "sad"}

def main():
    feat_dir = Path("data/features")
    X_train = np.load(feat_dir / "X_train.npy")
    y_train = np.load(feat_dir / "y_train.npy")
    X_val = np.load(feat_dir / "X_val.npy")
    y_val = np.load(feat_dir / "y_val.npy")
    X_test = np.load(feat_dir / "X_test.npy")
    y_test = np.load(feat_dir / "y_test.npy")

    # Simple, strong baseline
    clf = LogisticRegression(max_iter=2000, n_jobs=1)
    clf.fit(X_train, y_train)

    def eval_split(name, X, y):
        pred = clf.predict(X)
        acc = accuracy_score(y, pred)
        f1 = f1_score(y, pred, average="macro")
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro-F1:  {f1:.4f}")
        print("Confusion matrix:\n", confusion_matrix(y, pred))
        print(classification_report(y, pred, target_names=[ID2LABEL[i] for i in sorted(ID2LABEL)]))

    eval_split("VAL", X_val, y_val)
    eval_split("TEST", X_test, y_test)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(clf, "models/emotion_clf.joblib")
    print("\n✅ Saved model: models/emotion_clf.joblib")

if __name__ == "__main__":
    main()
