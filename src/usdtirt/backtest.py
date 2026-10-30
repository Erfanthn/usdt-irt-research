"""A deliberately simple, vectorised long/flat backtest.

Conventions (these matter and are easy to get wrong):
  * A signal computed with data up to minute t can at best earn the move
    from t to t+1. By default we add one more minute of latency (see
    `delay_min`), because trading at the exact signal price is unrealistic.
  * Position is 1 (holding USDT) or 0 (holding rial). No leverage, no
    shorting: the USDT/IRT spot market does not offer it to retail users.
  * Each change of position pays `cost_pct` percent (fee + half the
    bid/ask spread). Results are reported for several cost levels.
  * Results are reported as risk-adjusted metrics (Sharpe, drawdown,
    win rate, profit factor), not as cumulative profit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TARGET


def run_backtest(
    df: pd.DataFrame, signal: pd.Series, cost_pct: float, delay_min: int = 1
) -> pd.DataFrame:
    """Return per-minute strategy and buy-and-hold returns.

    `signal` holds the desired position (0 or 1) decided at each minute.
    `delay_min` is extra execution latency: with delay 0 we would trade at
    the very price that produced the signal, which is not achievable in
    practice. The default of 1 minute is the realistic case.
    """
    price = df[TARGET]
    bar_ret = price.pct_change().fillna(0.0)
    # Never earn a return across a data gap.
    bar_ret = bar_ret.where(df["segment"] == df["segment"].shift(1), 0.0)

    position = signal.shift(1 + delay_min).fillna(0).clip(0, 1)
    trades = position.diff().abs().fillna(position.abs())
    strat_ret = position * bar_ret - trades * cost_pct / 100

    return pd.DataFrame(
        {"position": position, "trade": trades, "strategy": strat_ret, "buy_hold": bar_ret}
    )


def daily(returns: pd.Series) -> pd.Series:
    """Compound minute returns into calendar-day returns."""
    return (1 + returns).groupby(returns.index.floor("D")).prod() - 1


def sharpe(daily_ret: pd.Series) -> float:
    """Annualised Sharpe ratio from daily returns, risk-free rate = 0.

    Iranian crypto markets trade every day, hence sqrt(365).
    """
    sd = daily_ret.std()
    return float(np.sqrt(365) * daily_ret.mean() / sd) if sd > 0 else float("nan")


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough fall of the equity curve, in percent."""
    equity = (1 + returns).cumprod()
    return float((equity / equity.cummax() - 1).min() * 100)


def trade_list(bt: pd.DataFrame) -> pd.DataFrame:
    """Group consecutive in-position minutes into round-trip trades."""
    in_pos = bt["position"] > 0
    trade_id = (in_pos & ~in_pos.shift(1, fill_value=False)).cumsum().where(in_pos)
    grouped = bt.loc[in_pos].groupby(trade_id[in_pos])
    return pd.DataFrame(
        {
            "entry": grouped.apply(lambda g: g.index[0]),
            "minutes": grouped.size(),
            "return_pct": grouped["strategy"].apply(lambda r: ((1 + r).prod() - 1) * 100),
        }
    )


def summarize(bt: pd.DataFrame) -> dict:
    """Risk and accuracy metrics for one backtest."""
    trades = trade_list(bt)
    wins = trades["return_pct"][trades["return_pct"] > 0]
    losses = trades["return_pct"][trades["return_pct"] <= 0]
    d_strat, d_bh = daily(bt["strategy"]), daily(bt["buy_hold"])
    return {
        "sharpe": sharpe(d_strat),
        "sharpe_buy_hold": sharpe(d_bh),
        "max_drawdown_pct": max_drawdown(bt["strategy"]),
        "max_drawdown_buy_hold_pct": max_drawdown(bt["buy_hold"]),
        "n_trades": int(len(trades)),
        "win_rate": float(len(wins) / len(trades)) if len(trades) else float("nan"),
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("nan"),
        "avg_trade_pct": float(trades["return_pct"].mean()) if len(trades) else float("nan"),
        "time_in_market": float(bt["position"].mean()),
    }
