"""Data loading and validation. One place defines the schema so training and serving agree."""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUM = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
BIN = ["HasCrCard", "IsActiveMember"]
CAT = ["Geography", "Gender"]
FEATS = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", "NumOfProducts",
         "HasCrCard", "IsActiveMember", "EstimatedSalary"]
TARGET = "Exited"
RANGES = {"CreditScore": (300, 900), "Age": (18, 100), "Tenure": (0, 50), "Balance": (0, 1e7),
          "NumOfProducts": (1, 10), "EstimatedSalary": (0, 1e7), "HasCrCard": (0, 1), "IsActiveMember": (0, 1)}

def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean feature frame or raise ValueError with a clear message."""
    miss = [c for c in FEATS if c not in df]
    if miss:
        raise ValueError(f"missing columns: {miss}")
    x = df[FEATS].copy()
    for c in NUM + BIN:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    bad = [c for c in FEATS if x[c].isna().any()]
    if bad:
        raise ValueError(f"null or non-numeric values in: {bad}")
    for c, (lo, hi) in RANGES.items():
        n = int(((x[c] < lo) | (x[c] > hi)).sum())
        if n:
            raise ValueError(f"{c}: {n} value(s) outside [{lo}, {hi}]")
    for c in CAT:
        x[c] = x[c].astype(str).str.strip()
    return x

def load_data(path: str | None = None) -> pd.DataFrame:
    """Load the training CSV. Identifier columns (RowNumber, CustomerId, Surname) are never used as features."""
    df = pd.read_csv(path or os.path.join(ROOT, "data", "churn.csv"))
    if "Exited" not in df:
        raise ValueError("training data needs an 'Exited' column (0/1)")
    x = validate(df)
    x[TARGET] = df[TARGET].astype(int).values
    return x
