from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from joblib import dump

from src.utils import data_path, out_path
from src.features import add_basic_features, build_preprocessor

SEED = 42

def load_data():
    train = pd.read_csv(data_path("train.csv"))
    test = pd.read_csv(data_path("test.csv"))
    return train, test

def prepare(train: pd.DataFrame, test: pd.DataFrame):
    train_proc = add_basic_features(train)
    test_proc  = add_basic_features(test)
    y = train_proc["Survived"].astype(int)
    X = train_proc.drop(columns=["Survived"])
    return X, y, test_proc

def cv_score(model: Pipeline, X, y, cv=5) -> float:
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, X, y, scoring="accuracy", cv=skf, n_jobs=-1)
    return float(np.mean(scores))

def main():
    train, test = load_data()
    X, y, test_proc = prepare(train, test)

    pre = build_preprocessor()

    logit = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", C=2))
    ])

    rf = Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=SEED,
            n_jobs=-1
        ))
    ])

    logit_acc = cv_score(logit, X, y, cv=5)
    rf_acc    = cv_score(rf, X, y, cv=5)

    print(f"[CV] LogisticRegression accuracy: {logit_acc:.4f}")
    print(f"[CV] RandomForest accuracy     : {rf_acc:.4f}")

    best = rf if rf_acc >= logit_acc else logit
    best.fit(X, y)

    dump(best, out_path("model.joblib"))

    preds = best.predict(test_proc)
    sub = pd.DataFrame({"PassengerId": test_proc["PassengerId"], "Survived": preds})
    sub.to_csv(out_path("submission.csv"), index=False)
    print("Wrote submission.csv")

if __name__ == "__main__":
    main()
