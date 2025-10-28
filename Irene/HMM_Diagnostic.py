# -*- coding: utf-8 -*-
"""
HMM Diagnostics — regimes + monthly returns (with risk-free for bonds)

Inputs (adjust paths if needed)
  - NowcastResult/nowcast/regimes_for_diagnostics.csv   (Date index, monthly, Month-Start)
  - Data/df_1M_ret.csv                                  (monthly simple returns, EOM; decimals)

What this script produces
  - 12M forward returns:
      * Absolute totals: Russell 1000, US Agg Bond, Gold, Dollar
      * Bond EXCESS: Agg, IG, HY, Long Tsy (monthly subtract rf=US Short-term Treasury, then compound)
      * Relative spreads: Value-Growth, IG-HY, Short-Long Tsy, DM ex-US - EM
  - Train/Test split: Train ≤ 2019-12, Test ≥ 2020-01
  - Per-regime stats: mean, IR (=mean/std), WinRate, N  (Train & Test)
  - Plots: Train/Test side-by-side bars (mean/IR/WinRate) showing ALL regimes present in the slice
  - Significance:
      * Kruskal–Wallis across regimes (Train/Test)
      * Pairwise Mann–Whitney U with Benjamini–Hochberg correction (adaptive p-value heatmaps)
  - KS separability across regimes (adaptive heatmaps for KS D and KS p)
  - Slide-friendly tables: mean ± std per regime with a last row of ANOVA p-values (Train/Test)
  - Console audit: regime coverage (counts & spans) after 12M-forward trimming

All figures are shown and saved; all tables are saved.
"""

import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 150

# -------------------------- Paths & constants --------------------------
REGIME_PATH  = "NowcastResult/nowcast/regimes_for_diagnostics.csv"
RETURNS_PATH = "Data/df_1M_ret.csv"

OUT_ROOT = "DiagnosticsResult"
FIG_DIR  = os.path.join(OUT_ROOT, "figs")
TAB_DIR  = os.path.join(OUT_ROOT, "tables")
for d in [OUT_ROOT, FIG_DIR, TAB_DIR]:
    os.makedirs(d, exist_ok=True)

TRAIN_END  = pd.Timestamp("2019-12-01")
TEST_START = pd.Timestamp("2020-01-01")
FORWARD_M  = 12

# significance & stats controls
MIN_REG_SAMPLES      = 5   # used for Train significance
MIN_REG_SAMPLES_TEST = 3   # relaxed threshold for Test significance (keeps plots intact)

# canonical RF column for bonds
RF_COL_CANON = "US Short-term Treasury"

# Asset aliases to be robust to naming variations
ALIASES = {
    "R1000":      ["Russell 1000", "RUSSELL 1000"],
    "R1000_VAL":  ["Russell 1000 Value"],
    "R1000_GRO":  ["Russell 1000 Growth"],
    "US_AGG":     ["US Agg Bond", "US Aggregate Bond"],
    "GOLD":       ["Gold", "XAU", "Gold Spot"],
    "DOLLAR":     ["Dollar", "US Dollar", "USD"],
    "IG":         ["US IG Corporate Bond", "US Investment Grade", "US Corp IG"],
    "HY":         ["US HY Corporate Bond", "US High Yield", "US Corp HY"],
    "TSY_SHORT":  ["US Short-term Treasury", "UST Short-term", "Short-term Treasury", "1-3M T-Bill"],
    "TSY_LONG":   ["US Long-term Treasury", "UST Long-term", "Long-term Treasury"],
    "DM_EX_US":   ["MSCI World ex USA Index (DM ex-US Equities)", "MSCI World ex USA", "DM ex-US"],
    "EM":         ["MSCI EM Index (EM Equities)", "MSCI EM", "EM Equities"],
}

ABSOLUTE_TOTAL_NAMES = ["Russell 1000", "US Agg Bond", "Gold", "Dollar"]
RELATIVE_PAIRS = [
    ("R1000_VAL", "R1000_GRO", "R1000 Value - Growth"),
    ("IG",        "HY",        "IG Corp - HY Corp"),
    ("TSY_SHORT", "TSY_LONG",  "Short Tsy - Long Tsy"),
    ("DM_EX_US",  "EM",        "DM ex-US - EM"),
]

# -------------------------- Date utilities --------------------------
def to_month_start_index(idx: pd.Index) -> pd.DatetimeIndex:
    dt = pd.to_datetime(idx, errors="coerce")
    return dt.to_period("M").to_timestamp(how="start")

def parse_date_index_eom_to_ms(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    if df.index.name != date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True)
        df = df.set_index(date_col)
    if not np.issubdtype(df.index.dtype, np.datetime64):
        df.index = pd.to_datetime(df.index, errors="coerce", infer_datetime_format=True)
    df.index = to_month_start_index(df.index)
    return df.sort_index()

# -------------------------- IO --------------------------
def read_regimes(path: str) -> pd.Series:
    df = pd.read_csv(path)
    df = parse_date_index_eom_to_ms(df, "Date")
    candidates = [c for c in df.columns if "Regime" in c] or \
                 [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    if not candidates:
        raise ValueError("No numeric regime column found in regimes_for_diagnostics.csv")
    s = pd.to_numeric(df[candidates[0]], errors="coerce").round().astype("Int64").dropna().astype(int)
    s.name = candidates[0]
    return s

def read_returns(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = parse_date_index_eom_to_ms(df, "Date")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all", axis=1).ffill().bfill()
    return df

def _ci_match(colnames: List[str], target: str) -> str:
    hits = [c for c in colnames if c.lower() == target.lower()]
    return hits[0] if hits else ""

def resolve_aliases(ret_df: pd.DataFrame) -> Dict[str, str]:
    colmap = {}
    lower_cols = {c.lower(): c for c in ret_df.columns}
    for key, alias_list in ALIASES.items():
        found = ""
        for a in alias_list:
            if a in ret_df.columns:
                found = a; break
            ci = _ci_match(list(lower_cols.keys()), a)
            if ci:
                found = lower_cols[ci]; break
        if found:
            colmap[key] = found
    return colmap

# -------------------------- Returns engineering --------------------------
def build_total_return_indices(ret_df: pd.DataFrame) -> pd.DataFrame:
    ret_df = ret_df.copy().dropna(how="all").sort_index()
    tri = (1.0 + ret_df).cumprod() * 100.0
    return tri

def forward_return(tri: pd.Series, m: int = 12) -> pd.Series:
    tri = tri.dropna()
    fwd = tri.shift(-m) / tri - 1.0
    return fwd.iloc[:-m] if m > 0 else fwd

def build_forward_returns_with_rf(ret_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    colmap = resolve_aliases(ret_df)

    # risk-free
    rf_col = colmap.get("TSY_SHORT", RF_COL_CANON)
    if rf_col not in ret_df.columns:
        raise KeyError(f"Risk-free column not found. Expecting '{RF_COL_CANON}' or TSY_SHORT alias.\n"
                       f"Available: {list(ret_df.columns)}")
    rf = ret_df[rf_col]

    # absolute totals
    abs_needed = {
        "Russell 1000": colmap.get("R1000", ""),
        "US Agg Bond":  colmap.get("US_AGG", ""),
        "Gold":         colmap.get("GOLD", ""),
        "Dollar":       colmap.get("DOLLAR", "")
    }
    abs_needed = {k: v for k, v in abs_needed.items() if v}
    tri_abs = build_total_return_indices(ret_df[list(abs_needed.values())]) if abs_needed else pd.DataFrame(index=ret_df.index)
    fwd_abs = { disp: forward_return(tri_abs[src], m=FORWARD_M) for disp, src in abs_needed.items() }

    # bonds: monthly excess then TRI_ex then 12M forward excess
    bond_map = {
        "US Agg Bond (excess)":          colmap.get("US_AGG", ""),
        "US IG Corporate Bond (excess)": colmap.get("IG", ""),
        "US HY Corporate Bond (excess)": colmap.get("HY", ""),
        "US Long-term Treasury (excess)":colmap.get("TSY_LONG", ""),
    }
    bond_map = {k: v for k, v in bond_map.items() if v}
    excess_bond = ret_df[list(bond_map.values())].sub(rf, axis=0) if bond_map else pd.DataFrame(index=ret_df.index)
    tri_excess_bond = build_total_return_indices(excess_bond) if not excess_bond.empty else pd.DataFrame(index=ret_df.index)
    fwd_bond_ex = { disp: forward_return(tri_excess_bond[src], m=FORWARD_M) for disp, src in bond_map.items() }

    # relatives (TRI total; rf cancels if both are risky; Short-Long is a term premium)
    rel_cols = {}
    for a, b, label in RELATIVE_PAIRS:
        if a in colmap and b in colmap:
            rel_cols[label] = (colmap[a], colmap[b])

    used_rel_cols = list(set([x for pair in rel_cols.values() for x in pair]))
    tri_rel = build_total_return_indices(ret_df[used_rel_cols]) if used_rel_cols else pd.DataFrame(index=ret_df.index)
    fwd_rel = {}
    for label, (ca, cb) in rel_cols.items():
        xa = forward_return(tri_rel[ca], m=FORWARD_M)
        xb = forward_return(tri_rel[cb], m=FORWARD_M)
        fwd_rel[label] = xa.sub(xb, fill_value=np.nan)

    out = pd.concat([pd.DataFrame(fwd_abs), pd.DataFrame(fwd_bond_ex), pd.DataFrame(fwd_rel)], axis=1)
    return out.sort_index(), {"rf_col": rf_col}

# -------------------------- Stats & tests --------------------------
def regime_stats(x: pd.Series, regime: pd.Series) -> pd.DataFrame:
    df = pd.concat([x, regime.rename("reg")], axis=1).dropna()
    if df.empty:
        return pd.DataFrame()
    out = []
    for r, grp in df.groupby("reg"):
        vals = grp.iloc[:, 0]
        mean = vals.mean()
        std  = vals.std(ddof=1)
        ir   = (mean / std) if std > 0 else np.nan
        win  = (vals > 0).mean()
        out.append((r, mean, std, ir, win, len(vals)))
    res = pd.DataFrame(out, columns=["reg","mean","std","IR","WinRate","N"]).set_index("reg").sort_index()
    return res

def kruskal_across_regimes(x: pd.Series, regime: pd.Series, min_n: int) -> Tuple[float, float]:
    df = pd.concat([x, regime.rename("reg")], axis=1).dropna()
    groups = [g.iloc[:,0].values for _, g in df.groupby("reg") if len(g) >= min_n]
    if len(groups) < 2:
        return np.nan, np.nan
    stat, p = stats.kruskal(*groups)
    return stat, p

def pairwise_mwu_pvals(x: pd.Series, regime: pd.Series, min_n: int) -> pd.DataFrame:
    df = pd.concat([x, regime.rename("reg")], axis=1).dropna()
    regs = sorted([r for r, g in df.groupby("reg") if len(g) >= min_n])
    if len(regs) < 2:
        return pd.DataFrame()
    pmat = pd.DataFrame(np.nan, index=regs, columns=regs, dtype=float)
    raw_ps, pairs = [], []
    for i, ri in enumerate(regs):
        xi = df.loc[df["reg"]==ri].iloc[:,0].values
        for j, rj in enumerate(regs):
            if j <= i:
                continue
            xj = df.loc[df["reg"]==rj].iloc[:,0].values
            _, p = stats.mannwhitneyu(xi, xj, alternative="two-sided")
            pmat.loc[ri, rj] = p
            raw_ps.append(p); pairs.append((ri, rj))
    # Benjamini–Hochberg
    if raw_ps:
        p_arr = np.array(raw_ps, dtype=float)
        order = np.argsort(p_arr)
        m = len(p_arr)
        bh = np.empty_like(p_arr, dtype=float)
        prev = 1.0
        for rank, idx in enumerate(order, start=1):
            bh_val = min(prev, p_arr[idx] * m / rank)
            bh[idx] = bh_val
            prev = bh_val
        for (ri, rj), adjp in zip(pairs, bh):
            pmat.loc[ri, rj] = adjp
            pmat.loc[rj, ri] = adjp
        np.fill_diagonal(pmat.values, 0.0)
    return pmat

def pairwise_ks(x: pd.Series, regime: pd.Series, return_stat: bool, min_n: int) -> pd.DataFrame:
    df = pd.concat([x, regime.rename("reg")], axis=1).dropna()
    regs = sorted([r for r, g in df.groupby("reg") if len(g) >= min_n])
    if len(regs) < 2:
        return pd.DataFrame()
    mat = pd.DataFrame(np.nan, index=regs, columns=regs, dtype=float)
    for i, ri in enumerate(regs):
        xi = df.loc[df["reg"]==ri].iloc[:,0].values
        for j, rj in enumerate(regs):
            if j <= i:
                continue
            xj = df.loc[df["reg"]==rj].iloc[:,0].values
            stat, p = stats.ks_2samp(xi, xj, alternative="two-sided", mode="auto")
            val = stat if return_stat else p
            mat.loc[ri, rj] = val
            mat.loc[rj, ri] = val
    np.fill_diagonal(mat.values, 0.0)
    return mat

# -------------------------- Plotting helpers --------------------------
def save_show(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    #plt.show()
    plt.close(fig)

def _auto_limits(mat: pd.DataFrame, kind: str) -> Tuple[float, float, str]:
    """
    kind: 'p' for p-values (MWU-BH, KS p), 'ksd' for KS D statistic.
    """
    vals = mat.replace([np.inf, -np.inf], np.nan).values.astype(float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return (0.0, 1.0, ".3f")

    if kind == "p":
        hi = np.nanpercentile(vals, 95) if len(vals) > 4 else np.nanmax(vals)
        vmax = float(max(0.10, min(0.30, hi)))
        vmin = 0.0
        fmt = ".3f"
    elif kind == "ksd":
        lo = np.nanpercentile(vals, 5) if len(vals) > 4 else np.nanmin(vals)
        vmin = float(min(0.70, max(0.30, lo)))
        vmax = 1.0
        fmt = ".2f"
    else:
        vmin, vmax, fmt = float(np.nanmin(vals)), float(np.nanmax(vals)), ".3f"
        if vmin == vmax:
            vmin, vmax = 0.0, 1.0
    if abs(vmax - vmin) < 1e-6:
        if kind == "p":
            vmax = min(0.30, vmin + 1e-3); vmin = 0.0
        else:
            vmin = max(0.30, vmin - 1e-3); vmax = 1.0
    return vmin, vmax, fmt

def heatmap_matrix(mat: pd.DataFrame, title: str, fname: str, kind: str = "p"):
    """
    kind='p'  -> MWU-BH p, KS p  (lower ⇒ more significant)
    kind='ksd'-> KS D            (higher ⇒ more separable)
    """
    fig = plt.figure(figsize=(4.8, 4.1))
    ax = plt.gca()
    if mat.empty:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        ax.axis("off")
    else:
        vmin, vmax, fmt = _auto_limits(mat, kind)
        sns.heatmap(mat, annot=True, fmt=fmt, cmap="viridis",
                    vmin=vmin, vmax=vmax, ax=ax, cbar_kws={"label": "value"})
        if kind == "p":
            ax.set_title(f"{title}\n(lower ⇒ more significant)")
        else:
            ax.set_title(f"{title}\n(higher ⇒ more separable)")
        ax.set_xlabel("Regime"); ax.set_ylabel("Regime")
    save_show(fig, os.path.join(FIG_DIR, f"{fname}.png"))

def bargrid_train_test(stats_tr: Dict[str, pd.DataFrame], stats_te: Dict[str, pd.DataFrame],
                       title_prefix: str, fname_prefix: str, metric: str,
                       regime_tr: pd.Series = None, regime_te: pd.Series = None):
    """
    Two subplots: Train vs Test for the chosen metric.
    Shows all regimes present in each slice and annotates sample counts (N).
    """
    names = list(stats_tr.keys())
    n = len(names)
    fig, axes = plt.subplots(1, 2, figsize=(min(16, 4 + 2.4*n), 4.2), sharey=True)

    for ax, label, src, reg_series in zip(
        axes, ["Train", "Test"], [stats_tr, stats_te], [regime_tr, regime_te]
    ):
        if reg_series is not None and not reg_series.empty:
            regs = sorted(reg_series.unique().tolist())
            counts = reg_series.value_counts().reindex(regs).fillna(0).astype(int)
        else:
            regs = sorted(set().union(*[df.index.tolist() for df in src.values()]))
            counts = pd.Series({r: np.nan for r in regs})

        mat = pd.DataFrame(index=regs, columns=names, dtype=float)
        for nm, df in src.items():
            if df.empty or metric not in df.columns:
                continue
            mat[nm] = df.reindex(regs)[metric]

        if mat.dropna(how="all").empty:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            ax.axis("off")
            continue

        mat.plot(kind="bar", ax=ax)
        xticks = [f"{r}\n(N={counts.get(r, np.nan)})" for r in regs]
        ax.set_xticklabels(xticks, rotation=0)

        ax.set_title(f"{label} — {metric}")
        ax.set_xlabel("Regime"); ax.set_ylabel(metric)
        ax.legend(loc="best", fontsize=8, frameon=False)

    fig.suptitle(f"{title_prefix} — {metric}", y=1.03, fontsize=12)
    save_show(fig, os.path.join(FIG_DIR, f"{fname_prefix}_{metric}.png"))

# -------------------------- Slide-friendly summary table --------------------------
def _fmt_mean_std(s: pd.Series) -> str:
    if s.dropna().empty:
        return ""
    return f"{s.mean():.3f} ± {s.std(ddof=1):.3f}"

def build_slide_table(
    fwd_df: pd.DataFrame,
    regime: pd.Series,
    series_list: List[str],
    use_kruskal: bool = False,
    add_n: bool = True,
    regime_labels: Dict[int, str] | None = None,
    min_n_train: int = MIN_REG_SAMPLES,
) -> pd.DataFrame:
    """
    Returns a table:
      Regime | Regime_Label | <Series1> | <Series2> | ... | (optional N)
      Last row: 'ANOVA p-value' or 'Kruskal p-value'
    """
    df = pd.concat([fwd_df[series_list], regime.rename("reg")], axis=1).dropna()
    if df.empty:
        return pd.DataFrame()

    rows = []
    counts = df["reg"].value_counts()
    for r, grp in df.groupby("reg"):
        row = {"Regime": int(r)}
        if regime_labels and int(r) in regime_labels:
            row["Regime_Label"] = regime_labels[int(r)]
        for c in series_list:
            row[c] = _fmt_mean_std(grp[c])
        if add_n:
            row["N"] = int(counts.get(r, 0))
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("Regime").reset_index(drop=True)

    # Global p-values per series (ANOVA or Kruskal), with min-N filter
    pvals = {}
    for c in series_list:
        groups = [g[c].values for _, g in df.groupby("reg") if len(g) >= min_n_train]
        if len(groups) < 2 or any(len(g)==0 for g in groups):
            pvals[c] = np.nan
        else:
            if use_kruskal:
                _, p = stats.kruskal(*groups)
            else:
                _, p = stats.f_oneway(*groups)
            pvals[c] = p

    label = "Kruskal p-value" if use_kruskal else "ANOVA p-value"
    last = {"Regime": label}
    if "Regime_Label" in out.columns:
        last["Regime_Label"] = ""
    for c in series_list:
        last[c] = f"{pvals[c]:.4f}" if pd.notna(pvals[c]) else ""
    if add_n:
        last["N"] = ""
    out = pd.concat([out, pd.DataFrame([last])], ignore_index=True)
    return out

def save_table_png(df: pd.DataFrame, title: str, path_png: str):
    ncols = df.shape[1]
    fig_w = min(16, max(6, 1.6 * ncols))
    fig_h = 0.6 * (len(df) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.2)
    ax.set_title(title, fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(path_png, dpi=220, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# -------------------------- Audit helpers --------------------------
def audit_regime_coverage(name: str, idx: pd.DatetimeIndex, regime_slice: pd.Series):
    if regime_slice.empty:
        print(f"[AUDIT] {name}: EMPTY")
        return
    print(f"\n[AUDIT] {name} coverage")
    print(f"  Dates: {idx.min().date()} -> {idx.max().date()}  (N={len(idx)})")
    counts = regime_slice.value_counts().sort_index()
    for r, n in counts.items():
        d_sub = idx[regime_slice==r]
        print(f"  Regime {int(r)}: N={int(n)}  span= {d_sub.min().date()} -> {d_sub.max().date()}")

# -------------------------- Main workflow --------------------------
if __name__ == "__main__":
    # 1) Load regimes (MS index)
    regime = read_regimes(REGIME_PATH)

    # 2) Load returns (EOM -> MS) and build forward returns (with rf for bonds)
    ret = read_returns(RETURNS_PATH)
    fwd_ret, meta = build_forward_returns_with_rf(ret)
    rf_used = meta["rf_col"]
    print(f"[INFO] Using '{rf_used}' as risk-free for bond excess returns.")

    # 3) Align and split
    common_idx = fwd_ret.index.intersection(regime.index)
    fwd_ret = fwd_ret.loc[common_idx]
    regime   = regime.loc[common_idx]

    train_idx = fwd_ret.index[fwd_ret.index <= TRAIN_END]
    test_idx  = fwd_ret.index[fwd_ret.index >= TEST_START]

    regime_tr = regime.loc[train_idx]
    regime_te = regime.loc[test_idx]

    # ---- Audit coverage after 12M-forward trimming ----
    audit_regime_coverage("Train (after 12M fwd trim)", train_idx, regime_tr)
    audit_regime_coverage("Test  (after 12M fwd trim)",  test_idx,  regime_te)

    # Series buckets
    abs_total_cols = [nm for nm in ABSOLUTE_TOTAL_NAMES if nm in fwd_ret.columns]
    bond_ex_cols   = [c for c in fwd_ret.columns if c.endswith("(excess)")]
    rel_cols       = [lab for _, _, lab in RELATIVE_PAIRS if lab in fwd_ret.columns]

    # 4) Per-regime stats tables (saved per series)
    def compute_stats_for_list(cols: List[str], prefix: str):
        stats_tr, stats_te = {}, {}
        for c in cols:
            stats_tr[c] = regime_stats(fwd_ret[c].loc[train_idx], regime_tr)
            stats_te[c] = regime_stats(fwd_ret[c].loc[test_idx],  regime_te)
            if not stats_tr[c].empty:
                stats_tr[c].to_csv(os.path.join(TAB_DIR, f"{prefix}_{c}_train_stats.csv"))
            if not stats_te[c].empty:
                stats_te[c].to_csv(os.path.join(TAB_DIR, f"{prefix}_{c}_test_stats.csv"))
        return stats_tr, stats_te

    stats_tr_abs, stats_te_abs = compute_stats_for_list(abs_total_cols, "abs_total")
    stats_tr_bex, stats_te_bex = compute_stats_for_list(bond_ex_cols, "bond_ex")
    stats_tr_rel, stats_te_rel = compute_stats_for_list(rel_cols, "rel")

    # 5) Barplots Train/Test side-by-side for mean/IR/WinRate
    for metric in ["mean", "IR", "WinRate"]:
        if abs_total_cols:
            bargrid_train_test(stats_tr_abs, stats_te_abs, "Absolute performance across regimes",
                               "abs_total_perf", metric, regime_tr=regime_tr, regime_te=regime_te)
        if bond_ex_cols:
            bargrid_train_test(stats_tr_bex, stats_te_bex, "Bond EXCESS performance across regimes",
                               "bond_excess_perf", metric, regime_tr=regime_tr, regime_te=regime_te)
        if rel_cols:
            bargrid_train_test(stats_tr_rel, stats_te_rel, "Relative performance across regimes",
                               "rel_perf", metric, regime_tr=regime_tr, regime_te=regime_te)

    # 6) Significance tests & separability
    def do_tests_for_list(cols: List[str], label_prefix: str):
        rows = []
        for c in cols:
            stat_tr, p_tr = kruskal_across_regimes(fwd_ret[c].loc[train_idx], regime_tr, MIN_REG_SAMPLES)
            stat_te, p_te = kruskal_across_regimes(fwd_ret[c].loc[test_idx],  regime_te, MIN_REG_SAMPLES_TEST)
            rows.append([c, stat_tr, p_tr, stat_te, p_te])

            # pairwise MWU (BH)
            pmat_tr = pairwise_mwu_pvals(fwd_ret[c].loc[train_idx], regime_tr, MIN_REG_SAMPLES)
            pmat_te = pairwise_mwu_pvals(fwd_ret[c].loc[test_idx],  regime_te, MIN_REG_SAMPLES_TEST)
            if not pmat_tr.empty:
                heatmap_matrix(pmat_tr, f"{label_prefix} {c} — Train pairwise p (BH)",
                               f"{label_prefix}_mwu_{c}_train", kind="p")
            if not pmat_te.empty:
                heatmap_matrix(pmat_te, f"{label_prefix} {c} — Test pairwise p (BH)",
                               f"{label_prefix}_mwu_{c}_test", kind="p")

            # KS separability
            ksD_tr = pairwise_ks(fwd_ret[c].loc[train_idx], regime_tr, return_stat=True,  min_n=MIN_REG_SAMPLES)
            ksP_tr = pairwise_ks(fwd_ret[c].loc[train_idx], regime_tr, return_stat=False, min_n=MIN_REG_SAMPLES)
            ksD_te = pairwise_ks(fwd_ret[c].loc[test_idx],  regime_te, return_stat=True,  min_n=MIN_REG_SAMPLES_TEST)
            ksP_te = pairwise_ks(fwd_ret[c].loc[test_idx],  regime_te, return_stat=False, min_n=MIN_REG_SAMPLES_TEST)
            if not ksD_tr.empty:
                heatmap_matrix(ksD_tr, f"{label_prefix} {c} — Train KS D",
                               f"{label_prefix}_ksD_{c}_train", kind="ksd")
            if not ksP_tr.empty:
                heatmap_matrix(ksP_tr, f"{label_prefix} {c} — Train KS p",
                               f"{label_prefix}_ksP_{c}_train", kind="p")
            if not ksD_te.empty:
                heatmap_matrix(ksD_te, f"{label_prefix} {c} — Test KS D",
                               f"{label_prefix}_ksD_{c}_test", kind="ksd")
            if not ksP_te.empty:
                heatmap_matrix(ksP_te, f"{label_prefix} {c} — Test KS p",
                               f"{label_prefix}_ksP_{c}_test", kind="p")

        if rows:
            cols_name = ["Series","KW_stat_train","KW_p_train","KW_stat_test","KW_p_test"]
            df_out = pd.DataFrame(rows, columns=cols_name)
            df_out.to_csv(os.path.join(TAB_DIR, f"{label_prefix}_kw_summary.csv"), index=False)

    if abs_total_cols:
        do_tests_for_list(abs_total_cols, "abs_total")
    if bond_ex_cols:
        do_tests_for_list(bond_ex_cols, "bond_ex")
    if rel_cols:
        do_tests_for_list(rel_cols, "rel")

    # 7) Slide-style tables (absolute + NEW relative tables)
    regime_labels = None  # e.g., {1:"Stable Expansion", 2:"Disinflationary Boom", ...}
    abs_for_table = [nm for nm in ABSOLUTE_TOTAL_NAMES if nm in fwd_ret.columns]
    rel_for_table = [lab for _, _, lab in RELATIVE_PAIRS if lab in fwd_ret.columns]

    # Absolute — Train
    tbl_train_abs = build_slide_table(
        fwd_df=fwd_ret.loc[train_idx],
        regime=regime_tr,
        series_list=abs_for_table,
        use_kruskal=False,
        add_n=True,
        regime_labels=regime_labels,
        min_n_train=MIN_REG_SAMPLES,
    )
    if not tbl_train_abs.empty:
        csv_path = os.path.join(TAB_DIR, "slide_abs_train_table.csv")
        png_path = os.path.join(FIG_DIR, "slide_abs_train_table.png")
        tbl_train_abs.to_csv(csv_path, index=False)
        save_table_png(tbl_train_abs, "Absolute (12M fwd) — Train", png_path)

    # Absolute — Test
    tbl_test_abs = build_slide_table(
        fwd_df=fwd_ret.loc[test_idx],
        regime=regime_te,
        series_list=abs_for_table,
        use_kruskal=False,
        add_n=True,
        regime_labels=regime_labels,
        min_n_train=MIN_REG_SAMPLES_TEST,
    )
    if not tbl_test_abs.empty:
        csv_path = os.path.join(TAB_DIR, "slide_abs_test_table.csv")
        png_path = os.path.join(FIG_DIR, "slide_abs_test_table.png")
        tbl_test_abs.to_csv(csv_path, index=False)
        save_table_png(tbl_test_abs, "Absolute (12M fwd) — Test", png_path)

    # -------- NEW: Relative — Train/Test slide tables (ANOVA) --------
    if rel_for_table:
        # Relative — Train
        tbl_train_rel = build_slide_table(
            fwd_df=fwd_ret.loc[train_idx],
            regime=regime_tr,
            series_list=rel_for_table,
            use_kruskal=False,
            add_n=True,
            regime_labels=regime_labels,
            min_n_train=MIN_REG_SAMPLES,
        )
        if not tbl_train_rel.empty:
            csv_path = os.path.join(TAB_DIR, "slide_rel_train_table.csv")
            png_path = os.path.join(FIG_DIR, "slide_rel_train_table.png")
            tbl_train_rel.to_csv(csv_path, index=False)
            save_table_png(tbl_train_rel, "Relative (12M fwd) — Train", png_path)

        # Relative — Test
        tbl_test_rel = build_slide_table(
            fwd_df=fwd_ret.loc[test_idx],
            regime=regime_te,
            series_list=rel_for_table,
            use_kruskal=False,
            add_n=True,
            regime_labels=regime_labels,
            min_n_train=MIN_REG_SAMPLES_TEST,
        )
        if not tbl_test_rel.empty:
            csv_path = os.path.join(TAB_DIR, "slide_rel_test_table.csv")
            png_path = os.path.join(FIG_DIR, "slide_rel_test_table.png")
            tbl_test_rel.to_csv(csv_path, index=False)
            save_table_png(tbl_test_rel, "Relative (12M fwd) — Test", png_path)

    print("\nDiagnostics complete.")
    print(f"Figures  -> {FIG_DIR}")
    print(f"Tables   -> {TAB_DIR}")
    print(f"RF used  -> {rf_used}")

