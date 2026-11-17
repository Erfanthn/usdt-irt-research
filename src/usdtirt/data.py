"""Loading and cleaning the 1-minute USDT/IRT price panel.

Expected CSV columns (neutral labels, so the analysis is about the *method*
rather than about any one venue):

    timestamp   UTC, one row per minute
    target      the exchange whose USDT/IRT price we try to predict
    ex_b, ex_c  two other Iranian crypto exchanges (USDT/IRT)
    usd_spot    Tehran open-market US dollar, same-day settlement
    usd_next    Tehran open-market US dollar, next-day settlement ("fardaii")
    herat_spot  Herat (Afghanistan) USD market, same-day settlement
    herat_next  Herat USD market, next-day settlement
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "target"
REFERENCES = ["ex_b", "ex_c", "usd_spot", "usd_next", "herat_spot", "herat_next"]

MAX_FFILL_MIN = 5      # fill at most 5 missing minutes
GAP_MIN = 30           # a hole longer than this starts a new "segment"
STALE_MIN = 120        # a price frozen this long is treated as a dead feed


def load_prices(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV export and return a clean 1-minute DataFrame.

    Steps:
      1. keep only the columns we use
      2. put rows on a regular 1-minute grid (UTC)
      3. forward-fill short holes only; long holes stay NaN
      4. drop obviously broken quotes (price <= 0, a >5% jump in 1 minute,
         or a price frozen for STALE_MIN minutes)
      5. add a `segment` id that increases after every long data gap, so
         rolling statistics and trades never bridge an outage
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df[["timestamp", TARGET, *REFERENCES]]
    df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.floor("min")
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()

    cols = [TARGET, *REFERENCES]
    df[cols] = df[cols].astype(float)
    df[cols] = df[cols].where(df[cols] > 0)

    # Remove single-minute spikes: a >5% move that is not sustained.
    for c in cols:
        jump = df[c].pct_change().abs()
        df.loc[jump > 0.05, c] = float("nan")

    # A crypto exchange trades 24/7, so a price frozen for hours means a dead
    # feed. (The Tehran/Herat dollar markets really do close at night, so
    # they are excluded from this check.)
    for c in [TARGET, "ex_b", "ex_c"]:
        run_id = (df[c] != df[c].shift()).cumsum()
        run_len = df.groupby(run_id)[c].transform("size")
        df.loc[run_len >= STALE_MIN, c] = float("nan")
    df = df.dropna(subset=[TARGET])

    observed = df.index.to_series()
    full_index = pd.date_range(df.index[0], df.index[-1], freq="1min", tz="UTC")
    df = df.reindex(full_index)
    df[cols] = df[cols].ffill(limit=MAX_FFILL_MIN)

    # A new segment starts after any gap longer than GAP_MIN minutes.
    gap = observed.diff().dt.total_seconds().div(60).reindex(full_index)
    df["segment"] = (gap > GAP_MIN).cumsum().ffill().astype(int)
    df = df.dropna(subset=[TARGET])
    df.index.name = "timestamp"
    return df


def describe_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Share of non-missing minutes per column (used in the README)."""
    return df.notna().mean().rename("coverage").to_frame()
