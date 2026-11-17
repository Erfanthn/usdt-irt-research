"""Run the full study and write tables + figures to results/.

Usage:
    python scripts/run_research.py --prices path/to/prices.csv.gz [--signals path/to/signal_log.csv] [--out results]

Time split (decided before looking at any results):
    2025-11-21 .. 2026-01-31  training only (warm-up for the first model)
    2026-02-01 .. 2026-07-31  VALIDATION - walk-forward; all choices are made here
    2026-08-01 .. end         TEST - touched once, with the settings chosen above
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from usdtirt.backtest import daily, run_backtest, summarize  # noqa: E402
from usdtirt.data import TARGET, describe_coverage, load_prices  # noqa: E402
from usdtirt.features import Z_WINDOWS, build_features, forward_return  # noqa: E402
from usdtirt.strategies import hold_after_trigger, walk_forward_model, zscore_rule  # noqa: E402

VALID = ("2026-02-01", "2026-07-31 23:59")
TEST = ("2026-08-01", None)
HORIZON = 60                     # predict the next 60 minutes
BASE_COST = 0.10                 # % per side, used for all selection decisions
COSTS = [0.0, 0.05, 0.10, 0.20]
DELAYS = [0, 1, 2, 5, 15]            # minutes of execution latency
OUT = ROOT / "results"          # overridden by --out


def period(obj, bounds):
    start, end = bounds
    return obj.loc[start:end] if end else obj.loc[start:]


def classification_metrics(y_true: pd.Series, position: pd.Series) -> dict:
    """Precision/recall of 'in position' as a prediction of 'price goes up'."""
    m = y_true.notna()
    y, p = y_true[m].astype(int), position[m].astype(int)
    return {
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "base_rate": float(y.mean()),
    }


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True)
    ap.add_argument("--signals", help="optional: log of a live signal system to audit")
    ap.add_argument("--out", default=str(OUT), help="folder for tables and figures")
    args = ap.parse_args()
    OUT = Path(args.out)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)

    df = load_prices(args.prices)
    X = build_features(df)
    fwd = forward_return(df, HORIZON)
    # Label: did the price rise by more than a round-trip cost?
    y = (fwd > 2 * BASE_COST).where(fwd.notna())

    report: dict = {
        "data": {
            "start": str(df.index[0]), "end": str(df.index[-1]),
            "minutes": int(len(df)), "segments": int(df["segment"].nunique()),
            "coverage": describe_coverage(df.drop(columns="segment"))["coverage"].round(3).to_dict(),
        }
    }

    # ---------------- 1. Rule-based baseline: grid search on VALIDATION only
    grid = []
    for w in Z_WINDOWS:
        for entry in [1.5, 2.0, 2.5, 3.0]:
            for max_hold in [60, 240]:
                sig = zscore_rule(X[f"z_composite_{w}"], entry, exit=0.0, max_hold=max_hold)
                bt = period(run_backtest(df, sig, BASE_COST), VALID)
                s = summarize(bt)
                grid.append({"window": w, "entry": entry, "max_hold": max_hold, **s})
    grid = pd.DataFrame(grid)
    grid.to_csv(OUT / "rule_grid_validation.csv", index=False)
    best = grid[grid["n_trades"] >= 30].sort_values("sharpe", ascending=False).iloc[0]
    rule_cfg = {"window": int(best["window"]), "entry": float(best["entry"]),
                "max_hold": int(best["max_hold"])}
    rule_sig = zscore_rule(X[f"z_composite_{rule_cfg['window']}"], rule_cfg["entry"], 0.0,
                           rule_cfg["max_hold"])

    # ---------------- 2. Walk-forward classifiers
    months = pd.period_range(VALID[0], df.index[-1].tz_localize(None), freq="M")
    probas, auc = {}, {}
    for kind in ["logit", "gbm"]:
        p = walk_forward_model(X, y, list(months), kind, embargo_min=HORIZON)
        probas[kind] = p
        for name, b in [("validation", VALID), ("test", TEST)]:
            m = period(p, b).notna() & period(y, b).notna()
            auc[f"{kind}_{name}"] = float(roc_auc_score(period(y, b)[m].astype(int), period(p, b)[m]))

    # Probability threshold chosen on validation (by Sharpe), then frozen.
    model_cfg = {}
    for kind, p in probas.items():
        rows = []
        for thr in [0.5, 0.55, 0.6, 0.65, 0.7]:
            sig = hold_after_trigger(p > thr, HORIZON)
            s = summarize(period(run_backtest(df, sig, BASE_COST), VALID))
            rows.append({"threshold": thr, **s})
        rows = pd.DataFrame(rows)
        rows.to_csv(OUT / f"{kind}_threshold_validation.csv", index=False)
        ok = rows[rows["n_trades"] >= 30]
        model_cfg[kind] = float(ok.sort_values("sharpe").iloc[-1]["threshold"]) if len(ok) else 0.5

    signals = {
        "zscore_rule": rule_sig,
        "logit": hold_after_trigger(probas["logit"] > model_cfg["logit"], HORIZON),
        "gbm": hold_after_trigger(probas["gbm"] > model_cfg["gbm"], HORIZON),
    }

    # ---------------- 3. Evaluate every strategy on validation and test, at several costs
    rows = []
    for name, sig in signals.items():
        for cost in COSTS:
            bt = run_backtest(df, sig, cost)
            for pname, b in [("validation", VALID), ("test", TEST)]:
                part = period(bt, b)
                cls = classification_metrics(period(y, b), part["position"])
                rows.append({"strategy": name, "period": pname, "cost_pct": cost,
                             **summarize(part), **cls})
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "results_all.csv", index=False)

    # ---------------- 3b. Sensitivity to execution delay (the key robustness check)
    delay_rows = []
    for name, sig in signals.items():
        for d in DELAYS:
            bt = run_backtest(df, sig, BASE_COST, delay_min=d)
            for pname, b in [("validation", VALID), ("test", TEST)]:
                delay_rows.append({"strategy": name, "period": pname, "delay_min": d,
                                   **summarize(period(bt, b))})
    delays = pd.DataFrame(delay_rows)
    delays.to_csv(OUT / "delay_sensitivity.csv", index=False)

    report["chosen_settings"] = {"zscore_rule": rule_cfg, "thresholds": model_cfg,
                                 "horizon_min": HORIZON, "base_cost_pct": BASE_COST}
    report["auc"] = auc
    report["n_rule_configs_tried"] = int(len(grid))

    # ---------------- 4. Feature importance (logistic coefficients, last fit on all pre-test data)
    from usdtirt.strategies import make_model
    pre = X.index < pd.Timestamp(TEST[0], tz="UTC") - pd.Timedelta(minutes=HORIZON)
    ok = pre & X.notna().all(axis=1).to_numpy() & y.notna().to_numpy()
    logit = make_model("logit").fit(X[ok][::5], y[ok][::5].astype(int))
    coefs = pd.Series(logit[-1].coef_[0], index=X.columns).sort_values()
    coefs.to_csv(OUT / "logit_coefficients.csv", header=["coef"])

    # ---------------- 5. Optional: audit an existing live signal log
    if args.signals:
        report["live_signal_audit"] = audit_live_signals(df, args.signals)

    (OUT / "summary.json").write_text(json.dumps(report, indent=2, default=str))
    make_figures(df, X, signals, coefs, results, delays)
    print(json.dumps(report, indent=2, default=str))
    base = results[results.cost_pct == BASE_COST]
    print(base.drop(columns=["cost_pct"]).round(3).to_string())
    print(delays[delays.period == "test"][["strategy", "delay_min", "sharpe", "win_rate", "profit_factor", "max_drawdown_pct"]].round(3).to_string())


def audit_live_signals(df: pd.DataFrame, path: str) -> dict:
    """Did the logged BUY signals precede rises, and SELLs precede falls?

    Only direction is checked; no position sizes or amounts are used.
    """
    log = pd.read_csv(path, parse_dates=["timestamp"])
    # The log is stored in Tehran local time without a zone.
    ts = log["timestamp"].dt.tz_localize("Asia/Tehran").dt.tz_convert("UTC").dt.floor("min")
    out = {"n_signals": int(len(log)), "first": str(ts.min()), "last": str(ts.max())}
    for h in [30, 60, 240]:
        fwd = forward_return(df, h).reindex(ts.to_numpy())
        fwd.index = log.index
        buy, sell = log["signal_type"] == "buy", log["signal_type"] == "sell"
        base_up = float((forward_return(df, h).loc[ts.min():ts.max()] > 0).mean())
        out[f"h{h}"] = {
            "buy_hit_rate": float((fwd[buy] > 0).mean()),
            "sell_hit_rate": float((fwd[sell] < 0).mean()),
            "base_rate_up": base_up,
            "median_move_after_buy_pct": float(fwd[buy].median()),
            "median_move_after_sell_pct": float(fwd[sell].median()),
        }
    return out


def make_figures(df, X, signals, coefs, results, delays) -> None:
    fig_dir = OUT / "figures"

    # Figure 1: the spread and its z-score over time
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    hourly = X["spread_composite"].resample("1h").mean()
    ax[0].plot(hourly.index, hourly, lw=0.7, color="#2b6cb0")
    ax[0].axhline(0, color="grey", lw=0.5)
    ax[0].set_ylabel("spread vs composite (%)")
    ax[0].set_title("Target exchange premium/discount to other dollar markets (hourly mean)")
    zh = X["z_composite_240"].resample("1h").last()
    ax[1].plot(zh.index, zh, lw=0.5, color="#4a5568")
    for v in (-2, 2):
        ax[1].axhline(v, color="#c53030", lw=0.6, ls="--")
    ax[1].set_ylabel("z-score (4h window)")
    for a in ax:
        a.axvspan(pd.Timestamp(VALID[0], tz="UTC"), pd.Timestamp(VALID[1], tz="UTC"),
                  color="#ebf8ff", zorder=0)
        a.axvspan(pd.Timestamp(TEST[0], tz="UTC"), df.index[-1], color="#fffaf0", zorder=0)
    fig.tight_layout()
    fig.savefig(fig_dir / "spread_zscore.png", dpi=130)
    plt.close(fig)

    # Figure 2: drawdown curves on the test period (risk, not profit)
    fig, ax = plt.subplots(figsize=(11, 4))
    for name, sig in signals.items():
        bt = period(run_backtest(df, sig, BASE_COST), TEST)
        eq = (1 + bt["strategy"]).cumprod()
        ax.plot(eq.index, (eq / eq.cummax() - 1) * 100, lw=0.9, label=name)
    bh = period(run_backtest(df, signals["zscore_rule"], BASE_COST), TEST)["buy_hold"]
    eq = (1 + bh).cumprod()
    ax.plot(eq.index, (eq / eq.cummax() - 1) * 100, lw=0.9, color="black", ls=":", label="buy & hold")
    ax.set_ylabel("drawdown (%)")
    ax.set_title(f"Drawdown on held-out test period (cost {BASE_COST}% per side)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "test_drawdown.png", dpi=130)
    plt.close(fig)

    # Figure 3: Sharpe vs transaction cost
    fig, ax = plt.subplots(figsize=(7, 4))
    t = results[results.period == "test"]
    for name, g in t.groupby("strategy"):
        ax.plot(g["cost_pct"], g["sharpe"], marker="o", label=name)
    ax.axhline(t["sharpe_buy_hold"].iloc[0], color="black", ls=":", label="buy & hold")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("cost per side (%)")
    ax.set_ylabel("Sharpe (test)")
    ax.set_title("How fast does the edge disappear with costs?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "sharpe_vs_cost.png", dpi=130)
    plt.close(fig)

    # Figure 5: Sharpe vs execution delay
    fig, ax = plt.subplots(figsize=(7, 4))
    t = delays[delays.period == "test"]
    for name, g in t.groupby("strategy"):
        ax.plot(g["delay_min"], g["sharpe"], marker="o", label=name)
    ax.axhline(t["sharpe_buy_hold"].iloc[0], color="black", ls=":", label="buy & hold")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xscale("symlog")
    ax.set_xticks(DELAYS)
    ax.set_xticklabels(DELAYS)
    ax.set_xlabel("execution delay (minutes)")
    ax.set_ylabel("Sharpe (test)")
    ax.set_title("Robustness: how much edge survives execution delay?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "sharpe_vs_delay.png", dpi=130)
    plt.close(fig)

    # Figure 4: logistic regression coefficients
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(coefs.index, coefs.values, color=np.where(coefs > 0, "#2f855a", "#c53030"))
    ax.set_title("Logistic regression coefficients (standardised features)")
    fig.tight_layout()
    fig.savefig(fig_dir / "logit_coefficients.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
