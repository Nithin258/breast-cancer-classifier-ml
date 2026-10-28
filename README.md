# Breast Cancer Diagnosis Classifier

**Final AI & ML Project – Data Science Internship, Codomax Digital Solutions (Module 06)**

An end-to-end machine learning project that predicts whether a breast tumour is **malignant** or **benign** from 30 measurements of cell nuclei, using the Wisconsin Diagnostic Breast Cancer dataset (569 samples, bundled with scikit-learn, so nothing needs downloading).

> ⚠️ **Educational project only. This is not a medical device and must not be used for real diagnosis.**

## Pipeline
1. **Load and check** the data (569 samples, 30 features, no missing values, 212 malignant / 357 benign)
2. **Split** 80/20 with stratification, so both sets keep the same class balance
3. **Compare models** using 5-fold cross-validation on the training data only: Logistic Regression, SVM (RBF), Random Forest
4. **Tune** the best model with `GridSearchCV`
5. **Evaluate once** on the untouched test set
6. **Save** the model (`model.joblib`) and metrics (`metrics.json`)
7. **Predict** with `predict.py`

The models are scored on *macro recall* instead of plain accuracy. In a screening context, missing a malignant case is worse than a false alarm, and macro recall treats both classes equally.

## Results (test set: 114 patients)
| Metric | Value |
|---|---|
| Model chosen | Logistic Regression (C = 0.1, with feature scaling) |
| Cross-validation macro recall | 97.8% |
| Test accuracy | 97.4% |
| Malignant recall | 95.2% (2 of 42 malignant cases missed) |
| Malignant precision | 97.6% |
| ROC-AUC | 0.996 |

| Confusion matrix | Top features |
|---|---|
| ![confusion matrix](confusion_matrix.png) | ![feature importance](feature_importance.png) |

Features describing tumour **size and shape irregularity** (worst perimeter, worst area, concave points) matter most.

## Honest limitations
- **Small test set.** 114 samples means each missed case moves recall by about 2 points, so treat the numbers as estimates and not guarantees.
- **One clean, well-known dataset.** Real clinical data is messier, and results would likely be lower on data from different labs or equipment.
- **Missing 2 of 42 malignant cases is not acceptable for real use.** A real system would need a lower decision threshold, external validation and clinical oversight.
- **Feature importance** comes from a separate Random Forest used for explanation, not from the final Logistic Regression model.

## Run it
```
pip install -r requirements.txt
python train.py                  # trains, evaluates, saves the model
python predict.py --sample 5     # predict one row of the dataset
python predict.py --csv my.csv   # predict a CSV with the 30 feature columns
python -m unittest test_project.py -v
```

## Files
| File | Purpose |
|---|---|
| `train.py` | full training and evaluation pipeline |
| `predict.py` | load the saved model and predict |
| `test_project.py` | unit tests |
| `model.joblib`, `metrics.json` | saved model and its test metrics |
| `confusion_matrix.png`, `feature_importance.png` | result charts |
| `requirements.txt` | dependencies |

## What I learned
- Keep the test set untouched until the very end, and compare models with cross-validation instead of the test score.
- Choose the metric to match the real-world cost of errors.
- Report limitations honestly alongside the results.

## Author
Karthik – B.Tech Artificial Intelligence & Data Science, Dhanalakshmi Srinivasan University
