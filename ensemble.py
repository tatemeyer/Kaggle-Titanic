from __future__ import annotations
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.utils import data_path, out_path
from src.features import add_basic_features, build_preprocessor

SEED = 42

def load_prepare():
    train = pd.read_csv(data_path("train.csv"))
    test = pd.read_csv(data_path("test.csv"))
    tr = add_basic_features(train)
    te = add_basic_features(test)
    X = tr.drop(columns=["Survived"])
    y = tr["Survived"].astype(int)
    return X, y, te

def main():
    X, y, te = load_prepare()
    pre = build_preprocessor()

    lr = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", C=2, penalty="l2"))
    ])
    rf = Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=9, min_samples_split=4, min_samples_leaf=2,
            max_features="log2", bootstrap=True, random_state=SEED, n_jobs=-1
        ))
    ])

    weight_grid = [(1,1), (1,1.2), (1,1.5), (1,2), (0.8,1.5), (0.5,2)]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    best_cv, best_w = -1.0, None
    for w in weight_grid:
        ens = VotingClassifier(
            estimators=[("lr", lr), ("rf", rf)],
            voting="soft",
            weights=list(w),
            n_jobs=-1
        )
        acc = cross_val_score(ens, X, y, scoring="accuracy", cv=skf, n_jobs=-1).mean()
        print(f"weights={w} -> CV={acc:.4f}")
        if acc > best_cv:
            best_cv, best_w = acc, w

    print(f"[CV] Best ensemble: {best_cv:.4f} with weights={best_w}")

    best_ens = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf)],
        voting="soft",
        weights=list(best_w),
        n_jobs=-1
    )
    best_ens.fit(X, y)
    preds = best_ens.predict(te)
    sub = pd.DataFrame({"PassengerId": te["PassengerId"], "Survived": preds})
    name = f"submission_EnsembleSW_{best_cv:.4f}.csv"
    sub.to_csv(out_path(name), index=False)
    print(f"Wrote {name}")

if __name__ == "__main__":
    main()
