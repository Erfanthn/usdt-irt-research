"""Tests for the properties that, if broken, would make every result wrong."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from make_synthetic_data import simulate  # noqa: E402
from usdtirt.backtest import max_drawdown, run_backtest  # noqa: E402
from usdtirt.data import load_prices  # noqa: E402
from usdtirt.features import build_features, forward_return  # noqa: E402
from usdtirt.strategies import zscore_rule  # noqa: E402


def _panel(tmp_path, days=5):
    path = tmp_path / "p.csv"
    simulate(days).to_csv(path, index=False)
    return load_prices(path)


def test_features_do_not_use_the_future(tmp_path):
    """Changing prices AFTER minute t must not change features AT minute t."""
    df = _panel(tmp_path)
    cut = df.index[len(df) // 2]
    before = build_features(df).loc[:cut]

    shocked = df.copy()
    shocked.loc[shocked.index > cut, "target"] *= 1.5
    after = build_features(shocked).loc[:cut]

    pd.testing.assert_frame_equal(before, after)


def test_label_looks_forward_exactly_h_minutes(tmp_path):
    df = _panel(tmp_path)
    fwd = forward_return(df, 10)
    t = df.index[100]
    expected = (df["target"].iloc[110] / df["target"].iloc[100] - 1) * 100
    assert np.isclose(fwd.loc[t], expected)


def test_backtest_applies_delay():
    idx = pd.date_range("2026-01-01", periods=6, freq="1min", tz="UTC")
    df = pd.DataFrame({"target": [100, 100, 110, 110, 110, 110], "segment": 0}, index=idx)
    signal = pd.Series([1, 0, 0, 0, 0, 0], index=idx)   # buy at minute 0 only
    # With delay 0 we'd hold during minute 1 (flat); the +10% jump at minute 2 is missed.
    assert run_backtest(df, signal, 0, delay_min=0)["strategy"].sum() == 0
    # A signal at minute 1 with delay 0 is in position for minute 2 and catches the jump.
    signal2 = pd.Series([0, 1, 0, 0, 0, 0], index=idx)
    assert np.isclose(run_backtest(df, signal2, 0, delay_min=0)["strategy"].sum(), 0.10)
    # With the default 1-minute delay, the same signal is too late.
    assert run_backtest(df, signal2, 0)["strategy"].sum() == 0


def test_costs_are_charged_on_entry_and_exit():
    idx = pd.date_range("2026-01-01", periods=5, freq="1min", tz="UTC")
    df = pd.DataFrame({"target": [100.0] * 5, "segment": 0}, index=idx)
    signal = pd.Series([1, 1, 0, 0, 0], index=idx)
    bt = run_backtest(df, signal, cost_pct=0.1, delay_min=0)
    assert bt["trade"].sum() == 2
    assert np.isclose(bt["strategy"].sum(), -0.002)


def test_zscore_rule_enters_and_exits():
    z = pd.Series([0, -2.5, -1.0, 0.2, 0, -2.5, np.nan, 0])
    pos = zscore_rule(z, entry=2.0, exit=0.0, max_hold=100)
    assert pos.tolist() == [0, 1, 1, 0, 0, 1, 0, 0]


def test_max_drawdown():
    r = pd.Series([0.1, -0.5, 0.2])
    assert np.isclose(max_drawdown(r), -50.0)
