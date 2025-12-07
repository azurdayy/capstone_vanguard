# -*- coding: utf-8 -*-
"""
Regime-Conditional MVO & Plain MVO — Monthly Rebalance

Computes TWO strategies:
  1. Regime-Conditional MVO: Uses regime information for asset allocation
  2. Plain MVO: Does NOT use regime information (baseline)

Logic:
  - Train phase (<= TRAIN_END):
      * Regime MVO: Use full train sample to estimate regime-specific MVO weights
      * Plain MVO: Use full train sample to estimate ONE set of MVO weights
  - Test phase (> TRAIN_END):
      * Rebalance EVERY MONTH
      * Regime MVO: For each month, estimate all regimes' MVO weights using expanding window,
                    then apply the weight corresponding to that month's realized regime
      * Plain MVO: For each month, estimate ONE set of MVO weights using expanding window

Outputs (under ./RollingMVO_result/KMeans):
  - regime_weights.csv (long table of regime-specific weights)
  - 1M_rebalance_weights_full.csv (Regime MVO monthly weights)
  - 1M_plain_weights_full.csv (Plain MVO monthly weights)
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import cvxpy as cp

# =========================
# ---- Config & Paths  ----
# =========================
try:
    CODE_DIR = Path(__file__).resolve().parent
except NameError:
    CODE_DIR = Path.cwd()

Methods = "BGMM"
DATA_DIR   = CODE_DIR / "Data"
OUT_DIR    = CODE_DIR / "RollingMVO_result" / Methods
OUT_DIR.mkdir(parents=True, exist_ok=True)

RETURNS_FILE       = DATA_DIR / "df_1M_ret.csv"
R2000_PRICE_FILE   = DATA_DIR / "Russel2000monthlyprice.csv"
regime = Methods + "_regime.csv"
REGIME_FILE        = CODE_DIR / "regime_path" / regime

# Output files
REGIME_WEIGHTS_HISTORY = OUT_DIR / "regime_weights.csv"
REGIME_WEIGHTS_FULL    = OUT_DIR / "1M_rebalance_weights_full.csv"
PLAIN_WEIGHTS_FULL     = OUT_DIR / "1M_plain_weights_full.csv"

TRAIN_END = pd.Timestamp("2019-12-31")

RNG_SEED            = 42
ALPHA_SHRINK        = 0.20
GAMMA               = 1
MIN_OBS_PER_REGIME  = 12
MIN_OBS_PLAIN       = 12
APPLY_NEXT_MONTH    = False

# =========================
# ---- Benchmark & Sets ----
# =========================
BENCHMARK = {
    "Russell 1000 Value": 0.25,
    "Russell 1000 Growth": 0.25,
    "Russell 2000": 0.10,
    "US Short-term Treasury": 0.10,
    "US Long-term Treasury": 0.10,
    "US IG Corporate Bond": 0.15,
    "US HY Corporate Bond": 0.05,
}
ASSET_ORDER = list(BENCHMARK.keys())

ASSET_TO_CLASS = {
    "Russell 1000 Value": "Equity",
    "Russell 1000 Growth": "Equity",
    "Russell 2000": "Equity",
    "US Short-term Treasury": "Treasuries",
    "US Long-term Treasury": "Treasuries",
    "US IG Corporate Bond": "Credit",
    "US HY Corporate Bond": "Credit",
}

CLASS_BENCH = {
    "Equity": (
        BENCHMARK["Russell 1000 Value"]
        + BENCHMARK["Russell 1000 Growth"]
        + BENCHMARK["Russell 2000"]
    ),
    "Treasuries": (
        BENCHMARK["US Short-term Treasury"]
        + BENCHMARK["US Long-term Treasury"]
    ),
    "Credit": (
        BENCHMARK["US IG Corporate Bond"]
        + BENCHMARK["US HY Corporate Bond"]
    ),
}

ASSET_LO = {k: max(0.0, v - 0.05) for k, v in BENCHMARK.items()}
ASSET_HI = {k: min(1.0, v + 0.05) for k, v in BENCHMARK.items()}
CLASS_LO = {k: max(0.0, v - 0.10) for k, v in CLASS_BENCH.items()}
CLASS_HI = {k: min(1.0, v + 0.10) for k, v in CLASS_BENCH.items()}

# =========================
# ---- Data Loading    ----
# =========================
def load_r2000_returns(path: Path) -> pd.Series:
    df = pd.read_csv(path)

    date_candidates = [c for c in df.columns if "date" in str(c).lower()]
    date_col = date_candidates[0] if date_candidates else df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col)

    non_date_cols = [c for c in df.columns if c != date_col]
    if not non_date_cols:
        raise ValueError("Russell 2000 price file missing price column.")
    price_col = non_date_cols[-1]

    df["Ret"] = df[price_col].pct_change()
    df["Period"] = df[date_col].dt.to_period("M")

    s = df.groupby("Period")["Ret"].last().dropna()
    s.name = "Russell 2000"
    return s


def load_returns(ret_path: Path, r2000_path: Path) -> pd.DataFrame:
    df = pd.read_csv(ret_path)
    if "Date" not in df.columns:
        raise ValueError("df_1M_ret.csv must have a 'Date' column.")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Period"] = df["Date"].dt.to_period("M")
    df = df.set_index("Period").sort_index()

    # check old 6 assets
    old_assets = [a for a in ASSET_ORDER if a != "Russell 2000"]
    missing_old = [a for a in old_assets if a not in df.columns]
    if missing_old:
        raise ValueError(f"Missing assets in df_1M_ret.csv: {missing_old}")

    s_r2000 = load_r2000_returns(r2000_path)
    df = df.join(s_r2000, how="inner")

    missing = [a for a in ASSET_ORDER if a not in df.columns]
    if missing:
        raise ValueError(f"After merging Russell 2000, missing assets: {missing}")

    return df[ASSET_ORDER]


def load_regime(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if "Unnamed: 0" not in df.columns:
        raise ValueError("Regime file missing 'Unnamed: 0' column.")

    regime_col = [c for c in df.columns if c != "Unnamed: 0"][-1]

    df["Date"] = pd.to_datetime(df["Unnamed: 0"])
    df["Period"] = df["Date"].dt.to_period("M")
    s = df.set_index("Period")[regime_col].astype(str)
    s.name = "Regime"
    return s

# =========================
# ---- Estimation Utils --- 
# =========================
def diag_shrinkage_cov(X: np.ndarray, alpha: float) -> np.ndarray:
    if X.shape[0] <= 1:
        return np.zeros((X.shape[1], X.shape[1]))
    S = np.cov(X, rowvar=False, ddof=1)
    D = np.diag(np.diag(S))
    return (1 - alpha) * S + alpha * D


def solve_mvo(mu, Sigma, asset_names, asset_to_class,
              asset_lo, asset_hi, class_lo, class_hi, gamma=0.5) -> pd.Series:
    n = len(asset_names)
    w = cp.Variable(n)
    obj = cp.Maximize(mu @ w - gamma * cp.quad_form(w, Sigma))

    cons = [cp.sum(w) == 1]

    for i, a in enumerate(asset_names):
        cons += [w[i] >= asset_lo[a], w[i] <= asset_hi[a]]

    classes = sorted(set(asset_to_class[a] for a in asset_names))
    for c in classes:
        idx = [i for i, a in enumerate(asset_names) if asset_to_class[a] == c]
        cons += [cp.sum(w[idx]) >= class_lo[c], cp.sum(w[idx]) <= class_hi[c]]

    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
        if w.value is None:
            prob.solve(solver=cp.SCS, verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)

    if w.value is None:
        raise RuntimeError("MVO failed to converge.")

    w_hat = np.array(w.value).ravel()
    w_hat[w_hat < 0] = 0.0
    if w_hat.sum() <= 0:
        w_hat = np.array([BENCHMARK[a] for a in asset_names])
    else:
        w_hat /= w_hat.sum()

    return pd.Series(w_hat, index=asset_names)

# =========================
# ---- Train helper    ----
# =========================
def compute_train_regime_weights(df_train: pd.DataFrame) -> dict:
    """
    Use the full TRAIN sample to compute one set of regime-specific MVO weights.
    """
    regimes = sorted(df_train["Regime"].unique())
    weight_map = {}
    for r in regimes:
        block = df_train[df_train["Regime"] == r][ASSET_ORDER]
        T = block.shape[0]
        if T < MIN_OBS_PER_REGIME:
            w_r = pd.Series(BENCHMARK)[ASSET_ORDER]
        else:
            mu = block.mean(axis=0).values
            Sigma = diag_shrinkage_cov(block.values, ALPHA_SHRINK) + 1e-8 * np.eye(len(ASSET_ORDER))
            w_r = solve_mvo(
                mu, Sigma,
                ASSET_ORDER, ASSET_TO_CLASS,
                ASSET_LO, ASSET_HI, CLASS_LO, CLASS_HI,
                gamma=GAMMA,
            )
        weight_map[r] = w_r
    return weight_map


def compute_train_plain_weights(df_train: pd.DataFrame) -> pd.Series:
    """
    Use the full TRAIN sample to compute ONE set of MVO weights (no regime).
    """
    T = df_train.shape[0]
    if T < MIN_OBS_PLAIN:
        return pd.Series(BENCHMARK)[ASSET_ORDER]
    
    mu = df_train[ASSET_ORDER].mean(axis=0).values
    Sigma = diag_shrinkage_cov(df_train[ASSET_ORDER].values, ALPHA_SHRINK) + 1e-8 * np.eye(len(ASSET_ORDER))
    w = solve_mvo(
        mu, Sigma,
        ASSET_ORDER, ASSET_TO_CLASS,
        ASSET_LO, ASSET_HI, CLASS_LO, CLASS_HI,
        gamma=GAMMA,
    )
    return w

# =========================
# ---- Monthly Rolling MVO ----
# =========================
def rolling_monthly_mvo(df: pd.DataFrame):
    """
    Compute BOTH Regime MVO and Plain MVO with monthly rebalance.
    
    Returns:
      - regime_weights_period: Regime MVO monthly weights
      - plain_weights_period: Plain MVO monthly weights
      - regime_weights_history: Long table of regime-specific weights
    """
    periods = list(df.index)
    n = len(periods)
    if n == 0:
        raise ValueError("Empty DataFrame for rolling MVO.")

    date_index = df.index.to_timestamp(how="end")
    train_mask_arr = (date_index <= TRAIN_END)
    test_mask_arr  = ~train_mask_arr

    regime_weights_period = pd.DataFrame(index=df.index, columns=ASSET_ORDER, dtype=float)
    plain_weights_period = pd.DataFrame(index=df.index, columns=ASSET_ORDER, dtype=float)
    regime_weight_history_rows = []

    # ----------------- TRAIN PHASE -----------------
    if train_mask_arr.any():
        df_train = df.iloc[train_mask_arr]
        
        # Regime MVO
        train_regime_map = compute_train_regime_weights(df_train)
        periods_train = list(df_train.index)
        last_train_period = periods_train[-1]

        for p in periods_train:
            r = df.loc[p, "Regime"]
            w_r = train_regime_map.get(r, pd.Series(BENCHMARK)[ASSET_ORDER])
            regime_weights_period.loc[p] = w_r.values

        for r, w_r in train_regime_map.items():
            row = {"RebalPeriod": last_train_period, "Regime": r}
            row.update(w_r.to_dict())
            regime_weight_history_rows.append(row)
        
        # Plain MVO
        train_plain_weights = compute_train_plain_weights(df_train)
        for p in periods_train:
            plain_weights_period.loc[p] = train_plain_weights.values

    # ----------------- TEST PHASE - MONTHLY -----------------
    if test_mask_arr.any():
        regimes_all = sorted(df["Regime"].unique())
        test_indices = np.where(test_mask_arr)[0]
        first_test_idx = int(test_indices[0])

        # Rebalance every 1 month
        rebal_indices = list(range(first_test_idx, n, 1))

        for k, i_rebal in enumerate(rebal_indices):
            p_rebal = periods[i_rebal]
            df_hist = df.iloc[:i_rebal]

            # ===== Regime MVO =====
            regime_weight_map = {}
            for r in regimes_all:
                hist_block = df_hist[df_hist["Regime"] == r][ASSET_ORDER]
                T = hist_block.shape[0]

                if T == 0 or T < MIN_OBS_PER_REGIME:
                    w_r = pd.Series(BENCHMARK)[ASSET_ORDER]
                else:
                    mu = hist_block.mean(axis=0).values
                    Sigma = diag_shrinkage_cov(hist_block.values, ALPHA_SHRINK) + 1e-8 * np.eye(len(ASSET_ORDER))
                    w_r = solve_mvo(
                        mu, Sigma,
                        ASSET_ORDER, ASSET_TO_CLASS,
                        ASSET_LO, ASSET_HI, CLASS_LO, CLASS_HI,
                        gamma=GAMMA,
                    )

                regime_weight_map[r] = w_r

                row = {"RebalPeriod": p_rebal, "Regime": r}
                row.update(w_r.to_dict())
                regime_weight_history_rows.append(row)

            # ===== Plain MVO =====
            hist_all = df_hist[ASSET_ORDER]
            T_all = hist_all.shape[0]
            
            if T_all < MIN_OBS_PLAIN:
                w_plain = pd.Series(BENCHMARK)[ASSET_ORDER]
            else:
                mu_plain = hist_all.mean(axis=0).values
                Sigma_plain = diag_shrinkage_cov(hist_all.values, ALPHA_SHRINK) + 1e-8 * np.eye(len(ASSET_ORDER))
                w_plain = solve_mvo(
                    mu_plain, Sigma_plain,
                    ASSET_ORDER, ASSET_TO_CLASS,
                    ASSET_LO, ASSET_HI, CLASS_LO, CLASS_HI,
                    gamma=GAMMA,
                )

            # Apply weights for this rebalance period
            i_next = rebal_indices[k + 1] if (k + 1) < len(rebal_indices) else n
            for j in range(i_rebal, i_next):
                if not test_mask_arr[j]:
                    continue
                p = periods[j]
                
                # Regime MVO: use regime-specific weight
                r_p = df.loc[p, "Regime"]
                w_regime = regime_weight_map.get(r_p, pd.Series(BENCHMARK)[ASSET_ORDER])
                regime_weights_period.loc[p] = w_regime.values
                
                # Plain MVO: use same weight for all
                plain_weights_period.loc[p] = w_plain.values

    regime_weights_history = pd.DataFrame(regime_weight_history_rows)
    return regime_weights_period, plain_weights_period, regime_weights_history

# =========================
# -------- Main ----------
# =========================
def main():
    np.random.seed(RNG_SEED)

    df_ret = load_returns(RETURNS_FILE, R2000_PRICE_FILE)
    s_reg  = load_regime(REGIME_FILE)

    df = df_ret.join(s_reg, how="inner")
    if df.empty:
        raise ValueError("No overlapping periods between returns and regime file.")

    print("[INFO] Computing Regime MVO and Plain MVO...")
    regime_weights, plain_weights, regime_history = rolling_monthly_mvo(df)

    # Save regime weights history
    regime_history["RebalDate"] = (
        pd.PeriodIndex(regime_history["RebalPeriod"], freq="M").to_timestamp(how="end")
    )
    col_order = ["RebalDate", "Regime"] + ASSET_ORDER
    regime_history[col_order].to_csv(REGIME_WEIGHTS_HISTORY, index=False, float_format="%.6f")

    # Save regime MVO monthly weights
    regime_reb = regime_weights.copy()
    if APPLY_NEXT_MONTH:
        regime_reb = regime_reb.shift(1)
    regime_reb.index = regime_reb.index.to_timestamp(how="end")
    regime_reb.to_csv(REGIME_WEIGHTS_FULL, float_format="%.6f", date_format="%Y-%m-%d")

    # Save plain MVO monthly weights
    plain_reb = plain_weights.copy()
    if APPLY_NEXT_MONTH:
        plain_reb = plain_reb.shift(1)
    plain_reb.index = plain_reb.index.to_timestamp(how="end")
    plain_reb.to_csv(PLAIN_WEIGHTS_FULL, float_format="%.6f", date_format="%Y-%m-%d")

    train_mask = (df.index.to_timestamp(how="end") <= TRAIN_END)
    print(f"[OK] Total months: {df.shape[0]}, Train months: {int(train_mask.sum())}")
    print(f"[OK] Regimes detected: {sorted(df['Regime'].unique())}")
    print(f"[OK] Saved regime weights history -> {REGIME_WEIGHTS_HISTORY}")
    print(f"[OK] Saved Regime MVO monthly weights -> {REGIME_WEIGHTS_FULL}")
    print(f"[OK] Saved Plain MVO monthly weights -> {PLAIN_WEIGHTS_FULL}")

if __name__ == "__main__":
    main()