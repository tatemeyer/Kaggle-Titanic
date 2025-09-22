from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline

from src.features import add_basic_features, build_preprocessor
from src.utils import data_path, out_path

SEED = 42

def load_prepare():
    train = pd.read_csv(data_path("train.csv"))
    test  = pd.read_csv(data_path("test.csv"))
    tr = add_basic_features(train)
    te = add_basic_features(test)
    X = tr.drop(columns=["Survived"])
    y = tr["Survived"].astype(int)
    return X, y, te

def main():
    X, y, te = load_prepare()
    pre = build_preprocessor()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    hgb = Pipeline([
        ("pre", pre),
        ("clf", HistGradientBoostingClassifier(
            random_state=SEED,
            early_stopping=True,
            validation_fraction=0.1,
        ))
    ])

    param_grid = {
        "clf__learning_rate": [0.03, 0.05, 0.08],
        "clf__max_depth": [None, 3, 4],
        "clf__max_leaf_nodes": [15, 31, 63],
        "clf__min_samples_leaf": [10, 20, 30],
        "clf__l2_regularization": [0.0, 0.1, 0.5],
    }

    gs = GridSearchCV(
        hgb, param_grid=param_grid, scoring="accuracy",
        cv=cv, n_jobs=-1, refit=True, verbose=1
    )
    gs.fit(X, y)

    best_cv = float(gs.best_score_)
    best_params = gs.best_params_
    print(f"[CV] HGB best: {best_cv:.4f}  params: {best_params}")

    preds = gs.predict(te)
    sub = pd.DataFrame({"PassengerId": te["PassengerId"], "Survived": preds})
    name = f"submission_HGB_{best_cv:.4f}.csv"
    sub.to_csv(out_path(name), index=False)
    print(f"Wrote {name}")

    with open(out_path("tuning_summary_hgb.json"), "w", encoding="utf-8") as f:
        json.dump({"cv_score": best_cv, "best_params": best_params}, f, indent=2)

if __name__ == "__main__":
    main()
