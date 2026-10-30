"""Feature engineering.

Core idea: the target exchange's USDT/IRT price should stay close to the
other Iranian dollar markets. When it drifts unusually far below (or above)
them, we ask whether it tends to move back. "Unusually far" is measured with
a rolling z-score of the percentage spread.

Every feature at minute t uses only data up to and including minute t,
so there is no look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import REFERENCES, TARGET

Z_WINDOWS = [60, 240, 1440]          # 1 hour, 4 hours, 1 day (in minutes)
RETURN_LAGS = [5, 15, 60]


def _mask_segment_start(s: pd.Series, segment: pd.Series, window: int) -> pd.Series:
    """Blank out the first `window` minutes of each data segment.

    A rolling window that straddles a multi-day outage would mix two very
    different market states, so those values are set to NaN.
    """
    minutes_into_segment = segment.groupby(segment).cumcount()
    return s.where(minutes_into_segment >= window)


def rolling_z(s: pd.Series, window: int) -> pd.Series:
    """(x - rolling mean) / rolling std, over the last `window` minutes."""
    mean = s.rolling(window, min_periods=window // 2).mean()
    std = s.rolling(window, min_periods=window // 2).std()
    return (s - mean) / std.replace(0, np.nan)


def rsi(price: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index with Wilder smoothing (0..100)."""
    delta = price.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature table indexed like `df`."""
    out = pd.DataFrame(index=df.index)
    seg = df["segment"]

    # Composite reference = median of the available reference prices.
    # The median is robust to one market being stale or mis-quoted.
    composite = df[REFERENCES].median(axis=1)
    spread = (df[TARGET] / composite - 1) * 100          # in percent
    out["spread_composite"] = spread

    for w in Z_WINDOWS:
        out[f"z_composite_{w}"] = _mask_segment_start(rolling_z(spread, w), seg, w)

    # Per-reference spread z-scores (1-hour window) show *which* market disagrees.
    for ref in REFERENCES:
        s = (df[TARGET] / df[ref] - 1) * 100
        out[f"z_{ref}_60"] = _mask_segment_start(rolling_z(s, 60), seg, 60)

    # Recent momentum of the target and of the composite (possible lead-lag).
    for lag in RETURN_LAGS:
        out[f"ret_target_{lag}"] = _mask_segment_start(df[TARGET].pct_change(lag) * 100, seg, lag)
        out[f"ret_composite_{lag}"] = _mask_segment_start(composite.pct_change(lag) * 100, seg, lag)

    out["rsi_14"] = _mask_segment_start(rsi(df[TARGET]), seg, 60)

    # Time of day in Tehran (UTC+3:30), encoded on a circle so 23:59 is next to 00:00.
    tehran_minute = (df.index.hour * 60 + df.index.minute + 210) % 1440
    out["tod_sin"] = np.sin(2 * np.pi * tehran_minute / 1440)
    out["tod_cos"] = np.cos(2 * np.pi * tehran_minute / 1440)
    return out


def forward_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Percent change of the target price over the NEXT `horizon` minutes.

    This is the label and is never used as an input feature. Values that
    would cross a data gap are set to NaN.
    """
    price = df[TARGET]
    fwd = (price.shift(-horizon) / price - 1) * 100
    same_segment = df["segment"].shift(-horizon) == df["segment"]
    return fwd.where(same_segment)
