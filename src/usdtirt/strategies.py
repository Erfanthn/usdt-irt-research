"""Two ways of turning features into a 0/1 position.

1. `zscore_rule`   - a transparent, hand-written mean-reversion rule.
2. `walk_forward_model` - a classifier retrained every month on past data only.

Both are research baselines with generic, textbook parameters. They are not
the configuration of any live trading system.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def zscore_rule(z: pd.Series, entry: float, exit: float, max_hold: int) -> pd.Series:
    """Buy when the target is `entry` std-devs cheap vs the references.

    Stay in until the spread has normalised (z >= exit) or `max_hold`
    minutes have passed, whichever comes first. A plain loop is used because
    the rule has state (are we in a position, and for how long).
    """
    zv = z.to_numpy()
    pos = np.zeros(len(zv), dtype=np.int8)
    holding, held_for = False, 0
    for i, value in enumerate(zv):
        if np.isnan(value):
            holding, held_for = False, 0          # data gap: step aside
        elif not holding and value <= -entry:
            holding, held_for = True, 0
        elif holding:
            held_for += 1
            if value >= exit or held_for >= max_hold:
                holding = False
        pos[i] = holding
    return pd.Series(pos, index=z.index)


def make_model(kind: str):
    """'logit' is the interpretable baseline; 'gbm' allows non-linear effects."""
    if kind == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500))
    if kind == "gbm":
        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=500,
            random_state=0,
        )
    raise ValueError(kind)


def walk_forward_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_months: list[pd.Period],
    kind: str,
    embargo_min: int,
    sample_every: int = 5,
) -> pd.Series:
    """Out-of-sample probabilities, one month at a time.

    For each test month the model is trained only on data that ended at
    least `embargo_min` minutes before the month starts. The embargo matters:
    the label looks `horizon` minutes ahead, so without it the last training
    labels would overlap the test period.

    Training uses every `sample_every`-th minute; neighbouring minutes are
    almost identical, so this loses little information and runs much faster.
    """
    proba = pd.Series(np.nan, index=X.index)
    valid = X.notna().all(axis=1) & y.notna()
    months = X.index.tz_localize(None).to_period("M")

    for month in test_months:
        start = month.start_time.tz_localize("UTC")
        train_end = start - pd.Timedelta(minutes=embargo_min)
        train = valid & (X.index < train_end)
        train_idx = X.index[train][::sample_every]
        test = (months == month) & X.notna().all(axis=1).to_numpy()
        if len(train_idx) < 10_000 or not test.any():
            continue
        model = make_model(kind).fit(X.loc[train_idx], y.loc[train_idx].astype(int))
        proba.loc[test] = model.predict_proba(X.loc[test])[:, 1]
    return proba


def hold_after_trigger(trigger: pd.Series, hold: int) -> pd.Series:
    """Turn a one-minute trigger into a position held for `hold` minutes."""
    return trigger.astype(int).rolling(hold, min_periods=1).max().fillna(0)
