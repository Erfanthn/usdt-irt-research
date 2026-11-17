"""Generate a synthetic price panel with the same shape as the real data.

The real dataset is not published (see README, "Data"). This script lets
anyone run the full pipeline end to end. The synthetic market has a known,
built-in mean-reverting spread, so the methods should find *some* signal;
the numbers it produces say nothing about the real market.

Usage:
    python scripts/make_synthetic_data.py --out data/sample/prices_synthetic.csv.gz --days 120
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def simulate(days: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = days * 1440
    idx = pd.date_range("2025-11-21", periods=n, freq="1min", tz="UTC")

    # Common "fair value" of the dollar: random walk with a slight upward drift
    fair = 100_000 * np.exp(np.cumsum(rng.normal(2e-6, 4e-4, n)))

    def ou(theta: float, sigma: float) -> np.ndarray:
        """Ornstein-Uhlenbeck process: noise that is pulled back towards 0."""
        x = np.zeros(n)
        eps = rng.normal(0, sigma, n)
        for i in range(1, n):
            x[i] = x[i - 1] * (1 - theta) + eps[i]
        return x

    target_dev = ou(theta=0.02, sigma=0.0006)       # the target wanders more
    cols = {"target": fair * (1 + target_dev)}
    for name, noise in [("ex_b", 3e-4), ("ex_c", 4e-4), ("usd_spot", 6e-4),
                        ("usd_next", 5e-4), ("herat_spot", 8e-4), ("herat_next", 7e-4)]:
        cols[name] = fair * (1 + ou(0.05, noise))

    df = pd.DataFrame(cols, index=idx).round(0)
    # Physical dollar markets are closed 22:00-08:00 Tehran time: freeze their price.
    tehran_hour = (idx + pd.Timedelta(minutes=210)).hour
    closed = (tehran_hour >= 22) | (tehran_hour < 8)
    for c in ["usd_spot", "usd_next", "herat_spot", "herat_next"]:
        df.loc[closed, c] = np.nan
        df[c] = df[c].ffill()
    return df.rename_axis("timestamp").reset_index().dropna()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sample/prices_synthetic.csv.gz")
    ap.add_argument("--days", type=int, default=310)
    args = ap.parse_args()
    simulate(args.days).to_csv(args.out, index=False)
    print("wrote", args.out)
