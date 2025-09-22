# Titanic – Machine Learning from Disaster

A clean, modular starter repo for the classic [Kaggle Titanic competition](https://www.kaggle.com/c/titanic).  
This repo contains reproducible pipelines, strong feature engineering, and model tuning scripts to reach competitive leaderboard scores.

---

## 🚀 Features

- **Modular structure**: `src/` with clear separation of features, models, tuning, and utilities.  
- **Feature engineering**:
  - `Title` extraction + rare grouping  
  - `FamilySize`, `IsAlone`, and **FamilyGroup buckets**  
  - `TicketPrefix`, **TicketGroupSize**  
  - `CabinDeck` extraction  
  - `FareLog` transformation  
  - Interaction features: `Sex×Pclass`, `IsAlone×Sex`  
  - Quantile binning for `Age` and `Fare`  
- **Models supported**:
  - Logistic Regression (with tuning for C, solver, penalty, class weight)  
  - Random Forest (randomized hyperparameter search)  
  - HistGradientBoostingClassifier (grid search, dense preprocessing)  
  - Soft-voting ensemble of LR + RF  
- **Cross-validation**: Stratified 5-fold CV for reliable estimates.  
- **Automated submission**: Best model predictions saved as `submission_*.csv`.  
- **Reproducible tuning logs**: Saves `tuning_summary.json` with model, params, and CV score.

---

## 📂 Repository structure

```
titanic-ml/
├── data/                  # raw CSVs (not tracked in git)
├── src/
│   ├── features.py        # feature engineering + preprocessors
│   ├── train.py           # baseline CV + submission
│   ├── tune.py            # compare Logistic, RF, HGB and pick best
│   ├── ensemble.py        # soft-vote LR + RF with weight sweep
│   ├── hgb_tune.py        # standalone tuner for HGB
│   ├── utils.py           # paths + helpers
│   └── __init__.py
├── environment.yml        # conda environment
├── README.md
└── .gitignore
```
---

## 🏃 Usage

### Baseline training
Runs Logistic Regression + Random Forest with CV, picks the better, and writes `submission.csv`:

```bash
python -m src.train
```

### Model tuning (Logit vs RF vs HGB)
Performs hyperparameter search for all three and saves the best:

```bash
python -m src.tune
```

Outputs:
- `submission_<Model>_<CV>.csv` → upload to Kaggle
- `tuning_summary.json` → stores params + CV score

### Ensemble sweep
Trains a soft-vote LR + RF ensemble with weight grid and submits the best:

```bash
python -m src.ensemble
```

### Optional: dedicated HGB tuning
```bash
python -m src.hgb_tune
```

---

## 📈 Results

On cross-validation (5-fold stratified):

- Logistic Regression: ~0.82–0.83  
- Random Forest: ~0.83–0.84 (tuned)  
- HistGradientBoosting: competitive, often 0.83–0.84  
- **Soft-vote Ensemble (LR + RF)**: **~0.8395 CV**
