import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
C = json.load(open(HERE / "contract.json"))
COLS = list(C["inputs"])
ACT = [c for c in COLS if c.endswith(".Actions")]
TARGET = C["source_target"]["column"]


class ContractError(Exception):
    pass


def validate(df, with_target=True):
    """Reject anything that breaks the contract."""
    allowed = COLS + ([TARGET] if with_target else [])
    extra = set(df.columns) - set(allowed)
    missing = set(allowed) - set(df.columns)
    if extra:
        raise ContractError(f"undeclared fields: {sorted(extra)}")
    if missing:
        raise ContractError(f"missing fields: {sorted(missing)}")
    if df[allowed].isnull().any().any():
        raise ContractError("missing values")
    for c, r in C["inputs"].items():
        x = pd.to_numeric(df[c], errors="coerce")
        if x.isnull().any() or (x % 1 != 0).any():
            raise ContractError(f"{c}: must be a whole number")
        if ((x < r["min"]) | (x > r["max"])).any():
            raise ContractError(f"{c}: must be {r['min']} to {r['max']}")
    if with_target:
        bad = set(df[TARGET]) - set(C["source_target"]["values"])
        if bad:
            raise ContractError(f"bad labels: {bad}")
    return df


def in_range_mask(df):
    """True for rows inside every contract range. Missing values are an error, not a drop."""
    if df[COLS].isnull().any().any():
        raise ContractError("missing values")
    ok = pd.Series(True, index=df.index)
    for c, r in C["inputs"].items():
        ok &= df[c].between(r["min"], r["max"])
    return ok


def features(df):
    """Same 6 features for training and the app."""
    d = df[COLS].astype(int).copy()
    d["Total"] = d[ACT].sum(axis=1)
    d["Per_day"] = d["Total"] / (d["Days.Logged.In"] + 1)
    d["Read_per_day"] = d["Read.Actions"] / (d["Days.Logged.In"] + 1)
    d["Sub_ratio"] = d["Submission.Actions"] / (d["Total"] + 1)
    d["Quiz_ratio"] = d["Quiz.Actions"] / (d["Total"] + 1)
    d["Read_ratio"] = d["Read.Actions"] / (d["Total"] + 1)
    return d


def target_gate():
    """Stop if Pass/Fail mapping has no source."""
    t = C["app_target"]
    if not t["mapping"] or not t["mapping_source"]:
        raise ContractError("BLOCKED: no sourced High/Low -> Pass/Fail mapping")
    return t["mapping"]


# ---------- weekly demo logs (v1.1) ----------
W = C["weekly_log"]
WEEK, SID = W["week_col"], W["id"]
WEEKLY_COLS = [SID] + W["info_fields"] + [WEEK] + COLS


def validate_weekly(df):
    """Rules for the demo weekly log."""
    extra = set(df.columns) - set(WEEKLY_COLS)
    missing = set(WEEKLY_COLS) - set(df.columns)
    if extra:
        raise ContractError(f"undeclared fields: {sorted(extra)}")
    if missing:
        raise ContractError(f"missing fields: {sorted(missing)}")
    if df[WEEKLY_COLS].isnull().any().any():
        raise ContractError("missing values")
    lo, hi = W["weeks"]
    wk = pd.to_numeric(df[WEEK], errors="coerce")
    if wk.isnull().any() or (wk % 1 != 0).any() or ((wk < lo) | (wk > hi)).any():
        raise ContractError(f"{WEEK}: must be {lo} to {hi}")
    if df.duplicated([SID, WEEK]).any():
        raise ContractError("duplicate Student.ID + Week")
    for c in COLS:
        x = pd.to_numeric(df[c], errors="coerce")
        if x.isnull().any() or (x % 1 != 0).any() or (x < 0).any():
            raise ContractError(f"{c}: must be a whole number >= 0")
    for c, m in W["per_week_max"].items():
        if (df[c] > m).any():
            raise ContractError(f"{c}: max {m} per week")
    return df


def midterm_totals(df):
    """Sum weeks 1-8 per student. Only complete students get a row."""
    df = validate_weekly(df)
    n_weeks = df.groupby(SID)[WEEK].nunique()
    full = n_weeks[n_weeks == W["midterm_week"]].index
    d = df[df[SID].isin(full)]
    info = d.groupby(SID)[W["info_fields"]].first()
    tot = d.groupby(SID)[COLS].sum().astype(int)
    return info.join(tot).reset_index()


if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    ok = in_range_mask(df)
    print("rows:", len(df), "| out of range (will drop):", (~ok).sum())
    validate(df[ok])
    print("schema / range / labels: OK")
    print("exact duplicate rows:", df.duplicated().sum())
    print("mapping:", target_gate())