from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer
from sklearn.impute import SimpleImputer

# Column sets used by the preprocessors
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "FareLog", "TicketGroupSize"]
CAT_COLS = [
    "Pclass", "Sex", "Embarked", "CabinDeck", "Title", "TicketPrefix", "IsAlone",
    "Sex_Pclass", "IsAlone_Sex",
    "FamilyGroup",
]

# Map for title normalization + rare grouping
TITLE_MAP = {
    # normalize variants
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    # group uncommon/rare titles
    "Sir": "Other", "Lady": "Other", "Countess": "Other",
    "Capt": "Other", "Col": "Other", "Don": "Other", "Dr": "Other",
    "Jonkheer": "Other", "Major": "Other", "Rev": "Other", "Dona": "Other"
}
MAIN_TITLES = {"Mr", "Mrs", "Miss", "Master"}

# Feature engineering
def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Title from Name
    out["Title"] = (
        out["Name"]
        .str.extract(r",\s*([^\.]+)\.", expand=False)
        .str.strip()
        .replace(TITLE_MAP)
    )
    # Collapse anything not in the main set to "Other"
    out.loc[~out["Title"].isin(MAIN_TITLES), "Title"] = "Other"

    # Family features
    out["FamilySize"] = out["SibSp"].fillna(0) + out["Parch"].fillna(0) + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)

    # FamilyGroup buckets (categorical)
    def family_bucket(n: float) -> str:
        if pd.isna(n): return "Unknown"
        n = int(n)
        if n == 1:      return "Solo"
        elif n <= 3:    return "Small"
        elif n <= 5:    return "Medium"
        else:           return "Large"
    out["FamilyGroup"] = out["FamilySize"].apply(family_bucket)

    # Ticket group size: count passengers sharing the same ticket
    ticket_counts = out["Ticket"].map(out["Ticket"].value_counts())
    out["TicketGroupSize"] = ticket_counts.fillna(1).astype(int)

    # Ticket prefix (remove digits/punct, uppercase; empty -> "NONE")
    out["TicketPrefix"] = (
        out["Ticket"]
        .str.replace(r"\d", "", regex=True)
        .str.replace(r"[./\s]+", "", regex=True)
        .str.upper()
        .replace("", "NONE")
    )

    # Cabin deck (first letter). Keep NaN; imputer will handle.
    out["CabinDeck"] = out["Cabin"].astype(str).str[0].replace("n", np.nan).str.upper()

    # Skewed Fare -> log1p
    out["FareLog"] = np.log1p(out["Fare"])

    # Simple interactions (categorical)
    out["Sex_Pclass"] = out["Sex"].astype(str) + "_" + out["Pclass"].astype(str)
    out["IsAlone_Sex"] = out["IsAlone"].astype(str) + "_" + out["Sex"].astype(str)

    return out

# Preprocessing pipeline
def build_preprocessor() -> ColumnTransformer:
    # Numeric: impute median, then scale (with_mean=False for sparse compatibility)
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler(with_mean=False)),
    ])

    # Categorical: impute mode, then one-hot
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    # Separate quantile binning for Age and Fare to avoid duplicate/zero-width bins
    bin_age = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("kb",  KBinsDiscretizer(
            n_bins=4,
            encode="onehot",
            strategy="quantile",
            quantile_method="averaged_inverted_cdf",  # future-proof
        )),
    ])
    bin_fare = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("kb",  KBinsDiscretizer(
            n_bins=4,
            encode="onehot",
            strategy="quantile",
            quantile_method="averaged_inverted_cdf",
        )),
    ])

    # Combine all branches
    pre = ColumnTransformer(
        transformers=[
            ("num",     num_pipe, NUM_COLS),
            ("cat",     cat_pipe, CAT_COLS),
            ("bin_age", bin_age,  ["Age"]),
            ("bin_fare",bin_fare, ["Fare"]),
        ],
        sparse_threshold=0.3,
        remainder="drop",
    )

    return pre
