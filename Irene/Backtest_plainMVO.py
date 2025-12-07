# -*- coding: utf-8 -*-
"""
Regime Allocation Backtesting — One-File Framework (U.S.-centric, Amelia constraints)

Inputs (under ./Data):
  - df_1M_ret.csv  (required)
  - Russel2000monthlyprice.csv (monthly prices for Russell 2000)

Optional weights file locations (first hit wins):
  - ./rebalance_weights_full.csv
  - ./MVO_result/rebalance_weights_full.csv
  - ./Data/weights.csv
  - ./Data/rebalance_weights_full.csv

Outputs (under ./RollingMVO_Backtest_result/KMeans):
  - backtest_summary.csv
  - portfolio_returns.csv
  - benchmark_returns.csv
  - portfolio_vs_benchmark_returns.csv
  - weights_used.csv
  - weights_last24_snapshot.csv
  - plot_cumulative_full.png
  - plot_drawdown_full.png        # test-only
  - plot_rolling_excess_return.png  # test-only
  - plot_rolling_ir.png             # test-only
  - plot_alloc_stacked.png
  - plot_cumulative_test_rebased.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from difflib import get_close_matches
from math import sqrt
from typing import Optional

# ------------------------------- PATHS -------------------------------
try:
    CODE_DIR = Path(__file__).resolve().parent
except NameError:  # e.g., running in notebook
    CODE_DIR = Path.cwd()

Methods = "BGMM"
DATA_DIR   = CODE_DIR / "Data"
OUT_DIR    = CODE_DIR / "RollingMVO_Backtest_result" / Methods
OUT_DIR.mkdir(parents=True, exist_ok=True)

RETURNS_FILE       = DATA_DIR / "df_1M_ret.csv"     # required
R2000_PRICE_FILE   = DATA_DIR / "Russel2000monthlyprice.csv"  # new: monthly prices for Russell 2000

WEIGHTS_FILE = CODE_DIR / "RollingMVO_result" / Methods / "1M_rebalance_weights_full.csv"
PLAIN_WEIGHTS_FILE = CODE_DIR / "RollingMVO_result" / Methods / "1M_plain_weights_full.csv"

# NEW: regime path 
regime = Methods + "_regime.csv"
REGIME_FILE = CODE_DIR / "regime_path" / regime

TRAIN_END = pd.Timestamp("2019-12-31")
RNG_SEED  = 42
SIM_NOISE_STD = 0.015  # simulation noise around benchmark

ROLL_WINDOW_MONTHS = 6

# ------------------------------- HELPERS -----------------------------
def canon(s: str) -> str:
    return "".join(str(s).strip().lower().split())

def load_r2000_returns(path: Path) -> pd.Series:
    """
    Load Russell 2000 monthly prices and compute simple returns,
    aggregated to month-end dates.
    """
    df = pd.read_csv(path)

    # Detect date column
    date_col = None
    for c in df.columns:
        if "date" in str(c).lower():
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col)

    # Detect price column as last non-date column
    non_date_cols = [c for c in df.columns if c != date_col]
    if not non_date_cols:
        raise ValueError("Russell 2000 price file must contain a price column.")
    price_col = non_date_cols[-1]

    # Simple monthly returns
    df["Ret"] = df[price_col].pct_change()

    # Aggregate to month-end dates
    df["MonthEnd"] = df[date_col].dt.to_period("M").dt.to_timestamp("M")
    s = df.groupby("MonthEnd")["Ret"].last().dropna()
    s.name = "Russell 2000"
    return s

def load_returns(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # accept either 'Date' or unnamed index-like first column
    date_col = None
    for c in df.columns:
        if str(c).strip().lower() == "date":
            date_col = c
            break
    if date_col is None:
        # fall back: try first column if parseable
        c0 = df.columns[0]
        try:
            pd.to_datetime(df[c0], errors="raise")
            date_col = c0
        except Exception:
            pass
    if date_col is None:
        raise ValueError("df_1M_ret.csv must include a date-like column (e.g., 'Date').")

    df["Date"] = pd.to_datetime(df[date_col]) + pd.offsets.MonthEnd(0)
    df = df.set_index("Date").sort_index()

    # keep only numeric columns from original file
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        raise ValueError("No numeric asset columns found in df_1M_ret.csv.")
    df_ret = df[cols].copy()

    # merge Russell 2000 returns
    if not R2000_PRICE_FILE.exists():
        raise FileNotFoundError(f"Russell 2000 price file not found: {R2000_PRICE_FILE}")
    s_r2000 = load_r2000_returns(R2000_PRICE_FILE)
    df_ret = df_ret.join(s_r2000, how="inner")  # intersection of dates

    return df_ret

def pick_us_assets(returns_df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "Russell 1000",                 # fallback broad equity
        "Russell 1000 Value",
        "Russell 1000 Growth",
        "Russell 2000",
        "US Agg Bond",
        "US Short-term Treasury",
        "US Long-term Treasury",
        "US IG Corporate Bond",
        "US HY Corporate Bond",
    ]
    keep = [c for c in keep if c in returns_df.columns]
    return returns_df[keep].copy() if keep else returns_df.copy()

def build_benchmark_weights(returns_df: pd.DataFrame) -> pd.Series:
    """
    Prefer the fixed 7-asset benchmark if all are available, otherwise fall back
    to the original heuristic.
    """
    target_assets = {
        "Russell 1000 Value": 0.25,
        "Russell 1000 Growth": 0.25,
        "Russell 2000": 0.10,
        "US Short-term Treasury": 0.10,
        "US Long-term Treasury": 0.10,
        "US IG Corporate Bond": 0.15,
        "US HY Corporate Bond": 0.05,
    }
    if set(target_assets.keys()).issubset(set(returns_df.columns)):
        w = pd.Series(target_assets, dtype=float)
        w = w / w.sum()
        return w

    # Fallback to heuristic if some assets are missing
    w = {}
    # Equity 60%
    if {"Russell 1000 Value","Russell 1000 Growth"}.issubset(returns_df.columns):
        w["Russell 1000 Value"] = 0.30
        w["Russell 1000 Growth"] = 0.30
    elif "Russell 1000" in returns_df.columns:
        w["Russell 1000"] = 0.60
    else:
        eq = [c for c in returns_df.columns if "Russell 1000" in c]
        if eq:
            for a in eq:
                w[a] = 0.60 / len(eq)

    # Bonds 40%
    if {"US Long-term Treasury","US Short-term Treasury","US IG Corporate Bond","US HY Corporate Bond"}.issubset(returns_df.columns):
        w["US Long-term Treasury"]  = 0.10
        w["US Short-term Treasury"] = 0.10
        w["US IG Corporate Bond"]   = 0.15
        w["US HY Corporate Bond"]   = 0.05
    elif "US Agg Bond" in returns_df.columns:
        w["US Agg Bond"] = 0.40
    else:
        bonds = [c for c in returns_df.columns if ("Treasury" in c or "Bond" in c)]
        if bonds:
            for a in bonds:
                w[a] = w.get(a, 0.0) + 0.40 / len(bonds)

    # if some classes missing (e.g., only equities in returns_df), normalize what we have
    w = pd.Series(w, dtype=float)
    if w.sum() == 0:
        # fallback: equal weight across available assets
        w = pd.Series(1.0, index=returns_df.columns, dtype=float)
    w = w / w.sum()
    return w

def classify_assets(columns) -> dict:
    m = {}
    for a in columns:
        if "Russell 1000" in a or "Russell 2000" in a:
            m[a] = "Equity"
        elif ("Treasury" in a) or ("Bond" in a):
            m[a] = "Bond"
        else:
            m[a] = "Other"
    return m

SUB_PAIRS = [
    ("Russell 1000 Value", "Russell 1000 Growth"),
    ("US Short-term Treasury", "US Long-term Treasury"),
    ("US IG Corporate Bond", "US HY Corporate Bond"),
]

def enforce_constraints(raw_w: pd.Series, benchmark_w: pd.Series, class_map: dict) -> pd.Series:
    # No shorting + renorm
    w = raw_w.clip(lower=0.0).copy()
    w = benchmark_w.copy() if w.sum() == 0 else (w / w.sum())

    # Asset-class deviation ±10% (Equity / Bond)
    for cls in ["Equity","Bond"]:
        bench_cls = benchmark_w[benchmark_w.index.map(lambda x: class_map.get(x,"Other")==cls)].sum()
        curr_cls  = w[w.index.map(lambda x: class_map.get(x,"Other")==cls)].sum()
        lo, hi = max(0.0, bench_cls - 0.10), min(1.0, bench_cls + 0.10)
        if (curr_cls < lo) or (curr_cls > hi):
            group = w.index[w.index.map(lambda x: class_map.get(x,"Other")==cls)]
            other = w.index.difference(group)
            target = lo if curr_cls < lo else hi
            if len(group) and w[group].sum() > 0:
                w[group] = w[group] / w[group].sum() * target
            elif len(group) and benchmark_w[group].sum() > 0:
                w[group] = benchmark_w[group] / benchmark_w[group].sum() * target

            rem = 1.0 - target
            if len(other) and w[other].sum() > 0:
                w[other] = w[other] / w[other].sum() * rem
            elif len(other) and benchmark_w[other].sum() > 0:
                w[other] = benchmark_w[other] / benchmark_w[other].sum() * rem

    # Sub-asset deviation ±5% (Value/Growth; Short/Long Tsy; IG/HY)
    for a, b in SUB_PAIRS:
        if a in w.index and b in w.index and a in benchmark_w.index and b in benchmark_w.index:
            for name in (a, b):
                lo, hi = max(0.0, benchmark_w[name] - 0.05), min(1.0, benchmark_w[name] + 0.05)
                if w[name] < lo:
                    delta = lo - w[name]
                    w[name] = lo
                    cp = b if name == a else a
                    if cp in w.index and w[cp] >= delta:
                        w[cp] -= delta
                elif w[name] > hi:
                    delta = w[name] - hi
                    w[name] = hi
                    cp = b if name == a else a
                    if cp in w.index:
                        w[cp] += delta
            # Keep total sum = 1 by scaling others
            pair_sum = w[[a, b]].sum()
            others = w.index.difference([a, b])
            if len(others) and w[others].sum() > 0:
                w[others] *= (1.0 - pair_sum) / w[others].sum()

    w = w.clip(lower=0.0)
    return benchmark_w.copy() if w.sum() == 0 else (w / w.sum())

# -------- robust weights reader (wide or long; auto-detect date column) --------
def _guess_date_col(df: pd.DataFrame):
    # prefer column literally named Date (case-insensitive)
    for c in df.columns:
        if str(c).strip().lower() == "date":
            return c
    # unnamed/index-like candidates
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl.startswith("unnamed") or cl in {"index", ""}:
            try:
                pd.to_datetime(df[c], errors="raise")
                return c
            except Exception:
                pass
    # any parseable datetime column
    for c in df.columns:
        try:
            pd.to_datetime(df[c], errors="raise")
            return c
        except Exception:
            continue
    return None

def weights_from_file(path: Path, returns_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path)

    # support long format (Date, Asset, Weight) or wide format (Date + asset columns)
    cols_lower = [str(c).lower() for c in df.columns]
    is_long = {"date", "asset", "weight"}.issubset(set(cols_lower))

    if is_long:
        dcol = next(c for c in df.columns if str(c).lower() == "date")
        acol = next(c for c in df.columns if str(c).lower() == "asset")
        wcol = next(c for c in df.columns if str(c).lower() == "weight")
        df[dcol] = pd.to_datetime(df[dcol]) + pd.offsets.MonthEnd(0)
        w_wide = df.pivot_table(index=dcol, columns=acol, values=wcol, aggfunc="last")
    else:
        dcol = _guess_date_col(df)
        if dcol is None:
            raise ValueError("Weights file: cannot find a date column (tried 'Date' and any parseable datetime column).")
        df[dcol] = pd.to_datetime(df[dcol]) + pd.offsets.MonthEnd(0)
        w_wide = df.set_index(dcol).sort_index()

    # harmonize names to returns_df (fuzzy allowed)
    ret_canon = {canon(c): c for c in returns_df.columns}
    mapped = {}
    for c in w_wide.columns:
        key = canon(c)
        if key in ret_canon:
            mapped[c] = ret_canon[key]
        else:
            hit = get_close_matches(key, list(ret_canon.keys()), n=1, cutoff=0.7)
            if hit:
                mapped[c] = ret_canon[hit[0]]

    if not mapped:
        raise ValueError("No asset columns in weights file match the returns universe after name harmonization.")

    w_wide = w_wide[[c for c in w_wide.columns if c in mapped]].copy()
    w_wide.columns = [mapped[c] for c in w_wide.columns]

    # keep only intersection with returns universe and order like returns_df
    common = [c for c in returns_df.columns if c in w_wide.columns]
    if not common:
        raise ValueError("After alignment, no overlapping assets between weights and returns.")
    w_wide = w_wide[common]
    w_wide.index.name = "Date"
    return w_wide

def simulate_weights(index: pd.Index, bench_w: pd.Series, noise_std: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    class_map = classify_assets(bench_w.index)
    rows = []
    for dt in index:
        base = bench_w.copy()
        noise = pd.Series(rng.normal(0, noise_std, size=len(base)), index=base.index)
        raw = (base + noise).clip(lower=0.0)
        raw = base.copy() if raw.sum() == 0 else (raw / raw.sum())
        w = enforce_constraints(raw, base, class_map)
        w.name = dt
        rows.append(w)
    wdf = pd.DataFrame(rows, index=index)
    wdf.index.name = "Date"
    return wdf

def portfolio_returns(weights_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.Series:
    cols = [c for c in weights_df.columns if c in returns_df.columns]
    if not cols:
        raise ValueError("No overlapping assets between weights_df and returns_df when computing portfolio returns.")
    w = weights_df[cols].copy()
    r = returns_df[cols].copy()
    idx = w.index.intersection(r.index)
    w = w.loc[idx].sort_index()
    r = r.loc[idx].sort_index()
    port = (w * r).sum(axis=1)
    port.name = "portfolio_ret"
    return port

# ---------------------- REGIME LOADING & SHADING ----------------------
def load_regime_series(path: Path) -> Optional[pd.Series]:
    """
    Load regime path from CSV.
    Expect a date column + one regime column (int/str).
    """
    if not path.exists():
        print(f"[WARN] Regime file not found: {path}. Proceeding without regime overlay.")
        return None

    df = pd.read_csv(path)
    # detect date column
    date_col = None
    for c in df.columns:
        if "date" in str(c).lower():
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col]) + pd.offsets.MonthEnd(0)
    df = df.sort_values(date_col)

    # regime column = first non-date column
    regime_cols = [c for c in df.columns if c != date_col]
    if not regime_cols:
        print(f"[WARN] No regime column found in regime file: {path}")
        return None
    reg_col = regime_cols[0]

    s = df.set_index(date_col)[reg_col].copy()
    s.name = "Regime"
    return s

def _shade_train_test(ax, index, train_end):
    if len(index) == 0: return
    xmin = index.min(); xmax = index.max()
    if xmin <= train_end:
        ax.axvspan(xmin, min(train_end, xmax), alpha=0.10)
    if train_end < xmax:
        ax.axvline(train_end, linestyle="--", linewidth=1)
        ax.axvspan(max(train_end, xmin), xmax, alpha=0.07)

def _fmt_pct(ax, decimals=0):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.{decimals}%}"))

def _rolling_annualized_excess(excess: pd.Series, window=ROLL_WINDOW_MONTHS):
    def _ann(x):
        x = np.asarray(x, dtype=float)
        if np.any(~np.isfinite(x)) or len(x) == 0:
            return np.nan
        return (np.prod(1.0 + x) ** (12.0 / len(x))) - 1.0
    return excess.rolling(window).apply(_ann, raw=False)

def _rolling_ir(port: pd.Series, bench: pd.Series, window=ROLL_WINDOW_MONTHS):
    ex = port - bench
    roll_ex = _rolling_annualized_excess(ex, window)
    roll_te = ex.rolling(window).std(ddof=0) * np.sqrt(12.0)
    ir = roll_ex / roll_te.replace(0.0, np.nan)
    return roll_ex, ir

def _plot_regime_bands(ax, regimes: Optional[pd.Series]):
    """
    Draw semi-transparent pastel vertical bands for each regime on ax.
    Return mapping {regime_label: color} for legend.
    """
    color_map = {}
    if regimes is None:
        return color_map
    reg = regimes.dropna()
    if reg.empty:
        return color_map
    reg = reg.sort_index()
    unique = list(pd.unique(reg.values))

    # pastel palette with good separation
    pastel_colors = [
        "#FFE0B2",  # soft orange
        "#FFCDD2",  # soft pink
        "#BBDEFB",  # light blue
        "#C8E6C9",  # light green
        "#D1C4E9",  # light purple
        "#FFF9C4",  # soft yellow
        "#B2DFDB",  # teal
        "#FFECB3",  # pastel amber
    ]

    last_reg = None
    start = None
    for dt, r in reg.items():
        if last_reg is None:
            last_reg, start = r, dt
        elif r != last_reg:
            idx = unique.index(last_reg)
            color = pastel_colors[idx % len(pastel_colors)]
            color_map[last_reg] = color
            ax.axvspan(start, dt, alpha=0.28, color=color, zorder=0)
            last_reg, start = r, dt
    # final span
    if last_reg is not None:
        idx = unique.index(last_reg)
        color = pastel_colors[idx % len(pastel_colors)]
        color_map[last_reg] = color
        ax.axvspan(start, reg.index[-1], alpha=0.28, color=color, zorder=0)
        color_map[last_reg] = color
    return color_map

# ----------------------------- METRICS -------------------------------
def annual_return(r: pd.Series) -> float:
    if len(r)==0: return np.nan
    return (1 + r).prod() ** (12 / len(r)) - 1

def annual_volatility(r: pd.Series) -> float:
    if len(r)==0: return np.nan
    return r.std(ddof=0) * sqrt(12)

def sharpe_ratio(r: pd.Series, rf_monthly: float = 0.0) -> float:
    if len(r)==0: return np.nan
    er = r - rf_monthly
    vol = annual_volatility(er)
    return np.nan if (vol == 0 or np.isnan(vol)) else annual_return(er) / vol

def max_drawdown(r: pd.Series) -> float:
    if len(r)==0: return np.nan
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1.0
    return dd.min()

def tracking_error(r: pd.Series, b: pd.Series) -> float:
    if len(r)==0: return np.nan
    ex = (r - b)
    return ex.std(ddof=0) * sqrt(12)

def information_ratio(r: pd.Series, b: pd.Series) -> float:
    te = tracking_error(r, b)
    if te == 0 or np.isnan(te): return np.nan
    ex_ann = annual_return(r - b)
    return ex_ann / te

def summarize(r: pd.Series, b: pd.Series) -> pd.Series:
    return pd.Series({
        "Annual Return": annual_return(r),
        "Annual Volatility": annual_volatility(r),
        "Sharpe (rf~0)": sharpe_ratio(r, 0.0),
        "Max Drawdown": max_drawdown(r),
        "Tracking Error": tracking_error(r, b),
        "Information Ratio": information_ratio(r, b),
        "Excess Return (annualized)": annual_return(r - b),
    })

# --------------------------- VISUALS ---------------------------------
def _add_regime_legend(ax, color_map: dict):
    """
    Add regime legend OUTSIDE the axes (right side).
    Portfolio / Benchmark legend stays inside the plot.
    """
    if not color_map:
        return
    handles = [
        Patch(facecolor=color_map[r], alpha=0.8, label=f"Regime {r}")
        for r in color_map
    ]
    reg_legend = ax.legend(
        handles=handles,
        title="Regimes",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        fontsize=8,
        framealpha=0.9,
    )
    ax.add_artist(reg_legend)

def plot_cumulative_ts(port_regime, port_plain, bench, train_end, out_dir,
                       filename="plot_cumulative_full.png",
                       as_percent=False,
                       regimes: Optional[pd.Series] = None):

    cum_pr = (1.0 + port_regime).cumprod()
    cum_pp = (1.0 + port_plain).cumprod()
    cum_b = (1.0 + bench).cumprod()

    reg = regimes.reindex(cum_pr.index).ffill() if regimes is not None else None

    fig, ax = plt.subplots(figsize=(10, 6))
    color_map = _plot_regime_bands(ax, reg)

    ax.plot(cum_pr.index, cum_pr.values, label="Regime MVO", linewidth=1.6)
    ax.plot(cum_pp.index, cum_pp.values, label="Plain MVO", linewidth=1.6)
    ax.plot(cum_b.index, cum_b.values, label="60/40 Benchmark", linewidth=1.6, linestyle="--")

    _shade_train_test(ax, cum_pr.index, train_end)
    ax.set_title("Cumulative Return — Three Strategies (Full Sample)")
    ax.set_xlabel("Date")
    if as_percent:
        _fmt_pct(ax, decimals=0)

    _add_regime_legend(ax, color_map)
    ax.legend(loc="upper left")

    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(out_dir / filename, dpi=160)
    plt.show()



def plot_drawdown_ts(port_regime, port_plain, bench, train_end, out_dir,
                     filename="plot_drawdown_full.png",
                     regimes: Optional[pd.Series] = None):

    def _dd(r: pd.Series) -> pd.Series:
        cum = (1.0 + r).cumprod()
        peak = cum.cummax()
        return (cum / peak) - 1.0

    dd_pr = _dd(port_regime)
    dd_pp = _dd(port_plain)
    dd_b = _dd(bench)

    mask = dd_pr.index > train_end
    dd_pr = dd_pr.loc[mask]
    dd_pp = dd_pp.loc[mask]
    dd_b = dd_b.loc[mask]
    if dd_pr.empty or dd_b.empty:
        return

    reg = regimes.reindex(dd_pr.index).ffill() if regimes is not None else None

    fig, ax = plt.subplots(figsize=(10, 6))
    color_map = _plot_regime_bands(ax, reg)

    ax.plot(dd_pr.index, dd_pr.values, label="Regime MVO DD", linewidth=1.6)
    ax.plot(dd_pp.index, dd_pp.values, label="Plain MVO DD", linewidth=1.6)
    ax.plot(dd_b.index, dd_b.values, label="Benchmark DD", linewidth=1.6, linestyle="--")

    ax.set_title("Drawdown — Three Strategies (Test Period)")
    ax.set_xlabel("Date")
    _fmt_pct(ax, decimals=0)

    _add_regime_legend(ax, color_map)
    ax.legend(loc="upper left")

    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(out_dir / filename, dpi=160)
    plt.show()



def plot_rolling_excess_ts(port_regime, port_plain, bench, train_end, out_dir,
                           filename="plot_rolling_excess_return.png",
                           window=ROLL_WINDOW_MONTHS,
                           regimes: Optional[pd.Series] = None):
    """
    Rolling annualized excess return — ONLY from test period.
    """
    ex_regime = port_regime - bench
    ex_plain = port_plain - bench
    roll_ex_regime = _rolling_annualized_excess(ex_regime, window)
    roll_ex_plain = _rolling_annualized_excess(ex_plain, window)

    mask = roll_ex_regime.index > train_end
    roll_ex_regime = roll_ex_regime.loc[mask]
    roll_ex_plain = roll_ex_plain.loc[mask]
    if roll_ex_regime.empty:
        return

    reg = regimes.reindex(roll_ex_regime.index).ffill() if regimes is not None else None

    fig, ax = plt.subplots(figsize=(10, 6))
    color_map = _plot_regime_bands(ax, reg)

    ax.plot(roll_ex_regime.index, roll_ex_regime.values, label="Regime MVO Excess")
    ax.plot(roll_ex_plain.index, roll_ex_plain.values, label="Plain MVO Excess")
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax.set_title(f"Rolling {window}M Annualized Excess Return (Test Period)")
    ax.set_xlabel("Date")
    _fmt_pct(ax, decimals=1)
    ax.grid(True, alpha=0.3)

    _add_regime_legend(ax, color_map)
    ax.legend(loc="upper left")

    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(out_dir / filename, dpi=160)
    plt.show()

def plot_rolling_ir_ts(port_regime, port_plain, bench, train_end, out_dir,
                       filename="plot_rolling_ir.png",
                       window=ROLL_WINDOW_MONTHS,
                       regimes: Optional[pd.Series] = None):
    """
    Rolling Information Ratio — ONLY from test period.
    """
    _, roll_ir_regime = _rolling_ir(port_regime, bench, window)
    _, roll_ir_plain = _rolling_ir(port_plain, bench, window)

    mask = roll_ir_regime.index > train_end
    roll_ir_regime = roll_ir_regime.loc[mask]
    roll_ir_plain = roll_ir_plain.loc[mask]
    if roll_ir_regime.empty:
        return

    reg = regimes.reindex(roll_ir_regime.index).ffill() if regimes is not None else None

    fig, ax = plt.subplots(figsize=(10, 6))
    color_map = _plot_regime_bands(ax, reg)

    ax.plot(roll_ir_regime.index, roll_ir_regime.values, label="Regime MVO IR")
    ax.plot(roll_ir_plain.index, roll_ir_plain.values, label="Plain MVO IR")
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax.set_title(f"Rolling Information Ratio ({window}M, Test Period)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)

    _add_regime_legend(ax, color_map)
    ax.legend(loc="upper left")

    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(out_dir / filename, dpi=160)
    plt.show()

def plot_alloc_stacked(weights_df: pd.DataFrame, out_path: Path, strategy_name: str = "Portfolio"):
    plt.figure(figsize=(10, 6))
    cols = list(weights_df.columns)
    x = weights_df.index
    y = [weights_df[c].values for c in cols]
    plt.stackplot(x, *y, labels=cols)
    plt.title(f"{strategy_name} Allocation — Test Period Only")
    plt.legend(loc="upper left", ncols=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.show()

def plot_cumulative_rebased_test(port_regime: pd.Series,
                                 port_plain: pd.Series,
                                 bench: pd.Series,
                                 train_end: pd.Timestamp,
                                 out_dir: Path,
                                 regimes: Optional[pd.Series] = None,
                                 filename: str = "plot_cumulative_test_rebased.png"):

    mask = port_regime.index > train_end
    port_regime_t = port_regime.loc[mask]
    port_plain_t = port_plain.loc[mask]
    bench_t = bench.loc[mask]
    if len(port_regime_t) == 0 or len(bench_t) == 0:
        return

    cum_pr = (1.0 + port_regime_t).cumprod()
    cum_pp = (1.0 + port_plain_t).cumprod()
    cum_b = (1.0 + bench_t).cumprod()

    cum_pr /= cum_pr.iloc[0]
    cum_pp /= cum_pp.iloc[0]
    cum_b /= cum_b.iloc[0]

    reg = regimes.reindex(cum_pr.index).ffill() if regimes is not None else None

    fig, ax = plt.subplots(figsize=(10, 6))
    color_map = _plot_regime_bands(ax, reg)

    ax.plot(cum_pr.index, cum_pr.values, label="Regime MVO", linewidth=1.6)
    ax.plot(cum_pp.index, cum_pp.values, label="Plain MVO", linewidth=1.6)
    ax.plot(cum_b.index, cum_b.values, label="60/40 Benchmark", linewidth=1.6, linestyle="--")

    ax.set_title("Cumulative Return — Test Period (Rebased)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)

    _add_regime_legend(ax, color_map)
    ax.legend(loc="upper left")

    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(out_dir / filename, dpi=160)
    plt.show()


def plot_all_visuals(port_ret_regime: pd.Series,
                     port_ret_plain: pd.Series,
                     bench_ret: pd.Series,
                     train_end: pd.Timestamp,
                     regimes: Optional[pd.Series],
                     out_dir: Path):
    idx = port_ret_regime.index.intersection(port_ret_plain.index).intersection(bench_ret.index)
    port_regime = port_ret_regime.loc[idx].sort_index()
    port_plain = port_ret_plain.loc[idx].sort_index()
    bench = bench_ret.loc[idx].sort_index()

    plot_cumulative_ts(port_regime, port_plain, bench, train_end, out_dir,
                       filename="plot_cumulative_full.png",
                       as_percent=False,
                       regimes=regimes)

    plot_drawdown_ts(port_regime, port_plain, bench, train_end, out_dir,
                     filename="plot_drawdown_full.png",
                     regimes=regimes)
    plot_rolling_excess_ts(port_regime, port_plain, bench, train_end, out_dir,
                           filename="plot_rolling_excess_return.png",
                           window=ROLL_WINDOW_MONTHS,
                           regimes=regimes)
    plot_rolling_ir_ts(port_regime, port_plain, bench, train_end, out_dir,
                       filename="plot_rolling_ir.png",
                       window=ROLL_WINDOW_MONTHS,
                       regimes=regimes)

    plot_cumulative_rebased_test(port_regime, port_plain, bench, train_end, out_dir,
                                 regimes=regimes,
                                 filename="plot_cumulative_test_rebased.png")

# ------------------------------ MAIN ---------------------------------
def main():
    # 1) Load returns & keep U.S.-centric universe (now includes Russell 2000)
    returns_all = load_returns(RETURNS_FILE)
    returns_df = pick_us_assets(returns_all)
    print(f"[INFO] Returns universe columns: {list(returns_df.columns)}")
    print(f"[INFO] Sample period: {returns_df.index.min().date()} -> {returns_df.index.max().date()}")

    # 2) Benchmark weights & returns
    bench_w = build_benchmark_weights(returns_df)
    bench_w_df = pd.DataFrame(index=returns_df.index, columns=bench_w.index, data=0.0)
    for c in bench_w.index:
        bench_w_df[c] = bench_w[c]
    bench_ret = portfolio_returns(bench_w_df, returns_df)
    bench_ret.name = "benchmark_ret"
    (OUT_DIR / "benchmark_returns.csv").write_text("")
    bench_ret.to_csv(OUT_DIR / "benchmark_returns.csv", header=True)

    # 3) Portfolio weights - Regime MVO (external or simulated)
    if WEIGHTS_FILE.exists():
        try:
            rel = WEIGHTS_FILE.relative_to(CODE_DIR)
            msg_path = rel
        except Exception:
            msg_path = WEIGHTS_FILE
        print(f"[INFO] Using Regime MVO weights from: {msg_path}")
        w_ext = weights_from_file(WEIGHTS_FILE, returns_df).reindex(returns_df.index)
        class_map = classify_assets(bench_w.index)
        aligned_rows = []
        for dt, row in w_ext.iterrows():
            if (row.isna()).all():
                row = bench_w.copy()
            else:
                row = row.fillna(0.0)
                row = bench_w.copy() if row.sum()==0 else (row / row.sum())
                row = enforce_constraints(row, bench_w, class_map)
            row.name = dt
            aligned_rows.append(row)
        weights_regime = pd.DataFrame(aligned_rows, index=returns_df.index)
    else:
        print("[INFO] No Regime MVO weights found. Simulating weights for framework verification.")
        weights_regime = simulate_weights(returns_df.index, bench_w, SIM_NOISE_STD, RNG_SEED)

    weights_regime.to_csv(OUT_DIR / "weights_used_regime.csv")

    # 3.5) Plain MVO weights
    if PLAIN_WEIGHTS_FILE.exists():
        print(f"[INFO] Using Plain MVO weights from: {PLAIN_WEIGHTS_FILE.relative_to(CODE_DIR)}")
        w_plain = weights_from_file(PLAIN_WEIGHTS_FILE, returns_df).reindex(returns_df.index)
        aligned_rows_plain = []
        for dt, row in w_plain.iterrows():
            if (row.isna()).all():
                row = bench_w.copy()
            else:
                row = row.fillna(0.0)
                row = bench_w.copy() if row.sum()==0 else (row / row.sum())
                row = enforce_constraints(row, bench_w, class_map)
            row.name = dt
            aligned_rows_plain.append(row)
        weights_plain = pd.DataFrame(aligned_rows_plain, index=returns_df.index)
    else:
        print("[WARN] No Plain MVO weights found. Using benchmark as placeholder.")
        weights_plain = bench_w_df.copy()

    weights_plain.to_csv(OUT_DIR / "weights_used_plain.csv")

    # 4) Portfolio returns
    port_ret_regime = portfolio_returns(weights_regime, returns_df)
    port_ret_regime.name = "regime_mvo_ret"
    port_ret_regime.to_csv(OUT_DIR / "portfolio_returns_regime.csv", header=True)
    
    port_ret_plain = portfolio_returns(weights_plain, returns_df)
    port_ret_plain.name = "plain_mvo_ret"
    port_ret_plain.to_csv(OUT_DIR / "portfolio_returns_plain.csv", header=True)

    # 5) Align & Train/Test masks
    idx = bench_ret.index.intersection(port_ret_regime.index).intersection(port_ret_plain.index)
    port_ret_regime = port_ret_regime.loc[idx]
    port_ret_plain = port_ret_plain.loc[idx]
    bench_ret = bench_ret.loc[idx]
    train_mask = bench_ret.index <= TRAIN_END
    test_mask  = bench_ret.index >  TRAIN_END

    # 5.5) Load regime path and align
    regime_series = load_regime_series(REGIME_FILE)
    if regime_series is not None:
        regime_series = regime_series.reindex(idx).ffill()

    # 6) Evaluation matrices - THREE STRATEGIES
    summary_regime_full  = summarize(port_ret_regime, bench_ret)
    summary_regime_train = summarize(port_ret_regime[train_mask], bench_ret[train_mask])
    summary_regime_test  = summarize(port_ret_regime[test_mask],  bench_ret[test_mask])
    
    summary_plain_full  = summarize(port_ret_plain, bench_ret)
    summary_plain_train = summarize(port_ret_plain[train_mask], bench_ret[train_mask])
    summary_plain_test  = summarize(port_ret_plain[test_mask],  bench_ret[test_mask])
    
    summary_bench_full  = summarize(bench_ret, bench_ret)
    summary_bench_train = summarize(bench_ret[train_mask], bench_ret[train_mask])
    summary_bench_test  = summarize(bench_ret[test_mask],  bench_ret[test_mask])
    
    summary_df = pd.DataFrame({
        "Regime MVO (Full)": summary_regime_full,
        "Regime MVO (Train)": summary_regime_train,
        "Regime MVO (Test)": summary_regime_test,
        "Plain MVO (Full)": summary_plain_full,
        "Plain MVO (Train)": summary_plain_train,
        "Plain MVO (Test)": summary_plain_test,
        "Benchmark (Full)": summary_bench_full,
        "Benchmark (Train)": summary_bench_train,
        "Benchmark (Test)": summary_bench_test,
    })
    print("\n===== PERFORMANCE SUMMARY (Three Strategies) =====")
    print(summary_df.round(6))
    summary_df.to_csv(OUT_DIR / "backtest_summary.csv")

    # 7) Visualizations — SAVE + SHOW
    plot_all_visuals(port_ret_regime, port_ret_plain, bench_ret, TRAIN_END, regime_series, OUT_DIR)
    
    # Plot weights for TEST PERIOD ONLY
    test_weights_regime = weights_regime.loc[weights_regime.index > TRAIN_END]
    test_weights_plain = weights_plain.loc[weights_plain.index > TRAIN_END]
    plot_alloc_stacked(test_weights_regime, OUT_DIR / "plot_alloc_stacked_regime.png", "Regime MVO")
    plot_alloc_stacked(test_weights_plain, OUT_DIR / "plot_alloc_stacked_plain.png", "Plain MVO")

    # 8) Tables for downstream
    ret_cmp = pd.concat([port_ret_regime, port_ret_plain, bench_ret], axis=1)
    ret_cmp.columns = ["Regime_MVO", "Plain_MVO", "Benchmark"]
    ret_cmp.to_csv(OUT_DIR / "portfolio_vs_benchmark_returns.csv")
    
    last24_regime = weights_regime.tail(24)
    last24_regime.to_csv(OUT_DIR / "weights_last24_regime.csv")
    last24_plain = weights_plain.tail(24)
    last24_plain.to_csv(OUT_DIR / "weights_last24_plain.csv")

    print(f"\n[DONE] All outputs saved to: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()