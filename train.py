"""Breast Cancer Diagnosis Classifier - Final AI & ML Project.

End-to-end pipeline: load -> split -> compare models (cross-validation)
-> tune the best -> evaluate once on the untouched test set -> save.

Educational project only. NOT a medical device.
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42


def load_data():
    data = load_breast_cancer(as_frame=True)
    return data.data, data.target, list(data.target_names)  # 0=malignant, 1=benign


def candidate_models():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(random_state=SEED)),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
    }


PARAM_GRIDS = {
    "Logistic Regression": {"logisticregression__C": [0.01, 0.1, 1, 10, 100]},
    "SVM (RBF)": {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.001]},
    "Random Forest": {"max_depth": [None, 5, 10], "min_samples_leaf": [1, 3]},
}


def main():
    X, y, class_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"Samples: {len(X)} | features: {X.shape[1]} | train {len(X_train)} / test {len(X_test)}")
    print("Class counts (malignant, benign):", np.bincount(y).tolist())

    # 1. Compare models with cross-validation on TRAINING data only
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cv_scores = {}
    for name, model in candidate_models().items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="recall_macro")
        cv_scores[name] = (scores.mean(), scores.std())
        print(f"CV macro-recall  {name:<20} {scores.mean():.3f} +/- {scores.std():.3f}")
    best_name = max(cv_scores, key=lambda n: cv_scores[n][0])
    print("Best by CV:", best_name)

    # 2. Tune the best model (still only using training data)
    search = GridSearchCV(candidate_models()[best_name], PARAM_GRIDS[best_name],
                          cv=cv, scoring="recall_macro")
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print("Best params:", search.best_params_, "| CV score:", round(search.best_score_, 3))

    # 3. Final evaluation - the test set is used exactly once
    y_pred = model.predict(X_test)
    scores_test = (model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba")
                   else model.decision_function(X_test))
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, scores_test)
    print("\n" + classification_report(y_test, y_pred, target_names=class_names))
    print("Confusion matrix [rows=actual, cols=predicted]:\n", cm)
    print(f"ROC-AUC: {auc:.3f}")
    missed = int(cm[0, 1])
    print(f"Malignant cases missed as benign: {missed} of {int(cm[0].sum())}")

    # 4. Figures
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.savefig(os.path.join(HERE, "confusion_matrix.png"), dpi=130, bbox_inches="tight")
    plt.close()

    rf = RandomForestClassifier(n_estimators=300, random_state=SEED).fit(X_train, y_train)
    imp = (rf.feature_importances_)
    top = np.argsort(imp)[-10:]
    plt.figure(figsize=(7, 4.5))
    plt.barh(X.columns[top], imp[top], color="#3b6ea8")
    plt.title("Top 10 features (Random Forest importance)")
    plt.xlabel("Importance")
    plt.savefig(os.path.join(HERE, "feature_importance.png"), dpi=130, bbox_inches="tight")
    plt.close()

    # 5. Save model + metrics
    joblib.dump({"model": model, "features": list(X.columns), "classes": class_names},
                os.path.join(HERE, "model.joblib"))
    metrics = {
        "model": best_name, "best_params": search.best_params_,
        "cv_macro_recall": round(search.best_score_, 4),
        "test_accuracy": round(report["accuracy"], 4),
        "test_recall_malignant": round(report["malignant"]["recall"], 4),
        "test_precision_malignant": round(report["malignant"]["precision"], 4),
        "test_roc_auc": round(auc, 4),
        "malignant_missed": missed, "test_size": len(y_test),
    }
    with open(os.path.join(HERE, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved model.joblib, metrics.json, confusion_matrix.png, feature_importance.png")


if __name__ == "__main__":
    main()
