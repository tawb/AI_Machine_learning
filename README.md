# Heart Disease Classification — ENCS5141 Assignment 1

Data preprocessing, feature engineering, and a comparative study of three classifiers
(Random Forest, SVM, MLP) on the Heart Disease dataset (918 patients, 12 columns).

This repository contains the full assignment submission: the notebook that runs the pipeline
end to end, the written report, the original assignment brief, and the dataset.

## Contents

| File | What it is |
|---|---|
| `Assignment1_Heart_Disease.ipynb` | The full solution — exploration, cleaning, encoding, scaling, feature selection, SMOTE, RF raw-vs-preprocessed comparison, default and tuned RF/SVM/MLP, plots and summary tables. |
| `Assignment1_Report.pdf` | The written report (abstract, dataset description, procedure, results, discussion, conclusion). |
| `First_Assignment-2026.pdf` | The original assignment brief from the instructor. |
| `Heart+DataSet.csv` | The dataset — 918 rows, 11 features + `HeartDisease` target. |

## What the project does

**Part 1 — Preprocessing and feature engineering.** Loads the raw data, finds three
hidden quality issues (10 invalid `Sex` codes, 173 medically impossible zeros in
`Cholesterol`/`RestingBP`, 13 negative `Oldpeak` values), imputes missing values with
median/mode, caps outliers with the IQR rule, encodes categoricals (binary mapping for
`Sex` and `ExerciseAngina`, one-hot for `ChestPainType`/`RestingECG`/`ST_Slope`), splits
80/20 with stratification, scales with `StandardScaler` fit on train only, runs ANOVA
F-test + Random Forest importance + PCA for feature selection, and applies SMOTE only on
the training fold. Then trains a Random Forest on the raw data and on the fully
preprocessed data and compares them on the same test set.

**Part 2 — Comparative classification.** Trains Random Forest, SVM (RBF), and MLP first
with default hyper-parameters, then with `GridSearchCV` (5-fold stratified CV, scoring
F1). Reports accuracy / precision / recall / F1 / ROC-AUC on the held-out test set, plus
training time, prediction time, and total tuning time. Adds confusion matrices, ROC
curves, and a hyper-parameter sweep.

## How to run

The notebook expects the CSV next to it as `Heart+DataSet.csv` (note the `+` is part of
the filename).

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn jupyter
jupyter notebook Assignment1_Heart_Disease.ipynb
```

Then run all cells top to bottom. The random seed is fixed at 42 throughout, the split
is 80/20 stratified, so the metric numbers are reproducible across runs. Wall-clock
times (training, tuning) will vary by machine — this is normal and expected.

## Headline results

| | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| RF (default) | 0.848 | 0.878 | 0.843 | 0.860 | 0.917 |
| RF (tuned)   | 0.832 | 0.866 | 0.824 | 0.844 | 0.919 |
| SVM (default)| 0.842 | 0.876 | 0.833 | 0.854 | 0.918 |
| SVM (tuned)  | 0.842 | 0.876 | 0.833 | 0.854 | 0.918 |
| MLP (default)| 0.815 | 0.854 | 0.804 | 0.828 | 0.863 |
| MLP (tuned)  | 0.821 | 0.856 | 0.814 | 0.834 | 0.897 |

Tuned SVM has the best test-set F1 (0.854) and tuned RF has the best ROC-AUC (0.919) —
the gap between them is well within run-to-run noise on a 184-sample test set. The MLP
trails by a more meaningful margin, which is the expected behaviour of neural nets on
small tabular data.

One notable finding from Part 1: a Random Forest trained on the *raw* data
(F1 = 0.889, AUC = 0.924) actually beat the same model trained on the fully preprocessed
data (F1 = 0.860, AUC = 0.917). The discussion section of the report explains this — the
IQR outlier capping clipped clinically meaningful extreme values (e.g. very high
cholesterol) that were strong disease signals, and SMOTE added artificial samples to a
class imbalance that was barely there to begin with. The cleaning steps that fixed real
errors (invalid `Sex` codes, hidden zeros, negative `Oldpeak`) were essential; the
generic "always cap outliers, always SMOTE" steps were not appropriate here.

## Stack

Python 3 · numpy · pandas · matplotlib · seaborn · scikit-learn · imbalanced-learn

## Course

ENCS5141 — Intelligent Systems Lab · Birzeit University · Second Semester 2025/2026

Author: Tawba Abdallah (1221002) · Instructor: Dr. Aziz Qaroush · Assistant: Rami Majadbeh
