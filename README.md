# Week 3: Building and Tuning an AI Model

An end-to-end support vector machine classifier for the **Wisconsin Diagnostic Breast Cancer** dataset. The project compares 50 hyperparameter combinations with five-fold stratified cross-validation, then evaluates the selected model on a held-out test set. This is an educational exercise, **not a clinical diagnostic tool**.

## Project files

- `week3_svm_model.py` — complete, commented, executable Python code.
- `week3_detailed_report.docx` — detailed report in the format required by the upload form.
- `week3_detailed_report.pdf` — the same report in PDF format, as requested in the assignment.
- `submission_description.txt` — a ready-to-paste description exceeding 200 words.
- `results/` — generated metrics, all 50 cross-validation candidates, and three visualizations from the recorded run.
- `requirements.txt` — exact primary library versions from the reference environment.

## Run the code

Python **3.13** was used for the reference run. From the project directory:

```bash
python -m pip install -r requirements.txt
python week3_svm_model.py --output-dir results
```

Alternatively, for other supported Python versions, install compatible releases with `python -m pip install numpy scikit-learn matplotlib`. The dataset is bundled with scikit-learn, so no input file or network access is needed after dependencies are installed. Different package or numerical-library versions may produce slightly different results.

The program writes `metrics.json`, `cv_results.csv`, `confusion_matrix.png`, `roc_curve.png`, and `cv_top_candidates.png` to the chosen output directory. The validation chart's whiskers show fold standard deviations, not confidence intervals.

## Method and measured results

- Data: 569 observations, 30 numeric features; malignant is remapped to positive class 1.
- Split: 455 training and 114 held-out test cases, stratified with seed 42.
- Preprocessing: `StandardScaler` fitted inside an SVC pipeline, separately within each fold.
- Tuning: 50 linear/RBF settings, five-fold stratified cross-validation, malignant F1 scoring.
- Selected setting: RBF kernel; `C=10`, `gamma=0.01`, `class_weight='balanced'`.
- Held-out accuracy **99.12%**; malignant precision **100.00%**, recall **97.62%**, F1 **98.80%**; ROC AUC **0.9970**.
- Confusion counts: 72 true negatives, 0 false positives, 1 false negative, 41 true positives.

These results describe one fixed held-out split and are not evidence of clinical reliability. See the report for leakage controls, uncertainty and limitations.

## Data attribution

Wolberg, W., Mangasarian, O., Street, N. and Street, W. (1995). *Breast Cancer Wisconsin (Diagnostic)* [dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5DW2B>. Dataset licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This script uses the copy bundled with [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).
