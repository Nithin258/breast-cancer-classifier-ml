"""Use the trained model.

  python predict.py --sample 5          # predict row 5 of the built-in dataset
  python predict.py --csv patients.csv  # CSV with the 30 feature columns

Educational project only. NOT a medical device.
"""
import argparse
import os

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer

HERE = os.path.dirname(os.path.abspath(__file__))


def load_model(path=os.path.join(HERE, "model.joblib")):
    return joblib.load(path)


def predict_frame(bundle, frame):
    """Return a DataFrame with the predicted class (and probability if available)."""
    missing = [c for c in bundle["features"] if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} required columns, e.g. {missing[:3]}")
    X = frame[bundle["features"]]
    model = bundle["model"]
    out = pd.DataFrame({"prediction": [bundle["classes"][i] for i in model.predict(X)]}, index=frame.index)
    if hasattr(model, "predict_proba"):
        out["confidence"] = model.predict_proba(X).max(axis=1).round(3)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, help="row index of the built-in dataset")
    ap.add_argument("--csv", help="CSV file containing the feature columns")
    args = ap.parse_args()
    bundle = load_model()

    if args.csv:
        frame = pd.read_csv(args.csv)
    elif args.sample is not None:
        data = load_breast_cancer(as_frame=True)
        frame = data.data.iloc[[args.sample]]
        print("Actual label:", data.target_names[data.target.iloc[args.sample]])
    else:
        ap.error("give --sample N or --csv FILE")
    print(predict_frame(bundle, frame).to_string())
    print("\nReminder: educational project, not medical advice.")


if __name__ == "__main__":
    main()
