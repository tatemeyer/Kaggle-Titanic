from __future__ import annotations
import json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import FunctionTransformer

from src.utils import data_path, out_path
from src.features import add_basic_features, build_preprocessor

SEED = 42

def load():
    train = pd.read_csv(data_path("train.csv"))
    test = pd.read_csv(data_path("test.csv"))
    return train, test

def prepare(train: pd.DataFrame, test: pd.DataFrame):
    train_proc = add_basic_features(train)
    test_proc  = add_basic_features(test)
    y = train_proc["Survived"].astype(int)
    X = train_proc.drop(columns=["Survived"])
    return X, y, test_proc

def main():
    train, test = load()
    X, y, test_proc = prepare(train, test)

    pre = build_preprocessor()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # --- Logistic Regression (grid) ---
    logit = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs"))
    ])
    logit_grid = {
        "clf__C": [0.05, 0.1, 0.5, 1, 2, 5, 10],
        "clf__solver": ["lbfgs", "saga"],
        "clf__penalty": ["l2"],
        # Optional: include class balancing pass
        "clf__class_weight": [None, "balanced"],
    }
    gs_logit = GridSearchCV(logit, logit_grid, scoring="accuracy", cv=cv, n_jobs=-1, refit=True, verbose=1)
    gs_logit.fit(X, y)
    res = [("LogisticRegression", float(gs_logit.best_score_), gs_logit.best_estimator_, gs_logit.best_params_)]

    # --- Random Forest (randomized) ---
    rf = Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(random_state=SEED, n_jobs=-1))
    ])
    rf_space = {
        "clf__n_estimators": [300, 600, 900, 1200],
        "clf__max_depth": [None, 5, 7, 9, 12],
        "clf__min_samples_split": [2, 4, 6, 8, 10],
        "clf__min_samples_leaf": [1, 2, 3, 4],
        "clf__max_features": ["sqrt", "log2"],
        "clf__bootstrap": [True, False],
    }
    rs_rf = RandomizedSearchCV(rf, rf_space, n_iter=40, scoring="accuracy", cv=cv,
                               random_state=SEED, n_jobs=-1, refit=True, verbose=1)
    rs_rf.fit(X, y)
    res.append(("RandomForest", float(rs_rf.best_score_), rs_rf.best_estimator_, rs_rf.best_params_))

    # --- HistGradientBoosting (grid) ---
    hgb = Pipeline([
    ("pre", pre),
    ("to_dense", FunctionTransformer(
        lambda X: X.toarray() if hasattr(X, "toarray") else X
    )),
    ("clf", HistGradientBoostingClassifier(
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.1,
    ))
])
    hgb_grid = {
        "clf__learning_rate": [0.03, 0.05, 0.08],
        "clf__max_depth": [None, 3, 4],
        "clf__max_leaf_nodes": [15, 31, 63],
        "clf__min_samples_leaf": [10, 20, 30],
        "clf__l2_regularization": [0.0, 0.1, 0.5],
    }
    gs_hgb = GridSearchCV(hgb, hgb_grid, scoring="accuracy", cv=cv, n_jobs=-1, refit=True, verbose=1)
    gs_hgb.fit(X, y)
    res.append(("HistGradientBoosting", float(gs_hgb.best_score_), gs_hgb.best_estimator_, gs_hgb.best_params_))

    # --- Pick winner ---
    res.sort(key=lambda t: t[1], reverse=True)
    best_name, best_cv, best_est, best_params = res[0]
    print("\n=== RESULTS ===")
    for name, s, _, p in res:
        print(f"{name:>22} best CV: {s:.4f}  params: {p}")
    print(f"\n>>> Selected: {best_name}  (CV={best_cv:.4f})")

    # Predict test set and write submission
    preds = best_est.predict(test_proc)
    sub = pd.DataFrame({"PassengerId": test_proc["PassengerId"], "Survived": preds})
    sub_name = f"submission_{best_name}_{best_cv:.4f}.csv"
    sub.to_csv(out_path(sub_name), index=False)
    print(f"Wrote {sub_name}")

    # Save summary
    meta = {
        "selected_model": best_name,
        "cv_score": best_cv,
        "best_params": best_params,
        "cv_folds": cv.get_n_splits(),
        "seed": SEED,
    }
    with open(out_path("tuning_summary.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Saved tuning_summary.json")

if __name__ == "__main__":
    main()
