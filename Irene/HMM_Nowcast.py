# -*- coding: utf-8 -*-
"""
HMM Regime Nowcast (Clean Module) — unified features + FSHMM soft weights (no feature removal)

What it does:
  - Read unified features CSV
  - Preprocess (winsorize + train-only z-score)
  - Learn FSHMM saliency on TRAIN -> build soft-weighted features (keep all columns)
  - Select K via BIC (on TRAIN, weighted)
  - Fit robust sticky HMM (several attempt configs to ensure convergence)
  - Nowcast regimes for TRAIN, OOS, FULL
  - Plot: BIC vs K; TRAIN/OOS/FULL regime step line with colored background bands
  - Save regimes for downstream diagnostics as a single CSV

Inputs:
  - Data/top_features_df.csv

Outputs (under OUT_ROOT/nowcast):
  - bic_FSW_train.csv + bic_FSW_train.png
  - regime_path_train_K*_FSW.csv/.png
  - regime_path_oos_K*_FSW.csv/.png
  - regime_path_full_K*_FSW.csv/.png
  - regimes_for_diagnostics.csv  (columns: Regime_FSW [, Regime_RAW])

Note:
  - This file intentionally contains NO diagnostics beyond BIC selection and plotting.
"""

import os, io, contextlib, warnings
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from hmmlearn.hmm import GaussianHMM

# ============================== Config ==============================
TOP_FEATURES_PATH = "Data/top_features_df.csv"

OUT_ROOT = "NowcastResult"
NOW_DIR  = os.path.join(OUT_ROOT, "nowcast")
SEL_DIR  = os.path.join(OUT_ROOT, "selection")
for d in [OUT_ROOT, NOW_DIR, SEL_DIR]:
    os.makedirs(d, exist_ok=True)

GLOBAL_MIN_DATE = "1990-01-01"
TRAIN_START = "1990-04-01"
TRAIN_END   = "2019-12-01"   # inclusive in TRAIN
OOS_START   = "2020-01-01"   # OOS begins here

# Preprocess
WINSOR_P = 0.01
USE_ZSCORE = True
CONST_STD_EPS = 1e-12

# FSHMM selection
FSHMM_K_FOR_SELECTION = 4
FSHMM_MAX_ITERS = 600
FSHMM_TOL = 1e-4
FSHMM_VAR_FLOOR = 1e-6
FSHMM_WEIGHT_EPS = 0.05
TOPN_FOR_DISPLAY = 30
RHO_MIN = 1e-8
RHO_MAX = 1.0 - 1e-8

# K candidates (BIC)
BIC_K_RANGE = list(range(3, 6))

# HMM base priors
DEF_COVARIANCE_TYPE = "diag"
DEF_N_INIT = 5
DEF_N_ITER = 1000
DEF_TOL = 1e-3
DEF_MIN_COVAR = 1e-3
DEF_SELF_LOOP = 0.90
DEF_STARTPROB_PRIOR = 10.0

# Branch switches
RUN_FSW = True          # FSHMM soft-weighted branch (primary)
RUN_RAW = False         # RAW branch (same scaler, no weights)

# Repro/plot
SEED = 19961211
np.random.seed(SEED)
plt.rcParams["figure.dpi"] = 150
warnings.filterwarnings("ignore")

# ============================== Utilities ==============================
def parse_dates_index(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    if df.index.name != date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True)
        df = df.set_index(date_col)
    if not np.issubdtype(df.index.dtype, np.datetime64):
        df.index = pd.to_datetime(df.index, errors="coerce", infer_datetime_format=True)
    return df.sort_index()

def read_top_csv(path: str, min_date: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = parse_dates_index(df, "Date")
    if min_date is not None:
        df = df[df.index >= pd.to_datetime(min_date)]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(axis=1, how="all").ffill().bfill()
    std = df.std(axis=0, skipna=True)
    keep = std[std > CONST_STD_EPS].index.tolist()
    return df[keep].copy()

def winsorize_df(df: pd.DataFrame, p: float) -> pd.DataFrame:
    if p is None or p <= 0: return df
    ql = df.quantile(p); qh = df.quantile(1 - p)
    return df.clip(lower=ql, upper=qh, axis=1)

def zfit(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    mu = df.mean(axis=0)
    sd = df.std(axis=0).replace(0.0, 1.0)
    return mu, sd

def zapply(df: pd.DataFrame, mu: pd.Series, sd: pd.Series) -> pd.DataFrame:
    return (df - mu) / sd

def save_and_show(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.show()
    plt.close(fig)

# ============================== FSHMM ==============================
def safe_logsumexp(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True) + 1e-300)
    return np.squeeze(out, axis=axis)

class FSHMM:
    def __init__(self, n_states: int, max_iter: int = 300, tol: float = 1e-4,
                 k: Optional[float] = None, var_floor: float = 1e-6,
                 random_state: int = 0):
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol
        self.k = k
        self.var_floor = var_floor
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    @staticmethod
    def _init_params(X: np.ndarray, S: int, rng):
        T, L = X.shape
        pi = np.full(S, 1.0 / S)
        A = np.full((S, S), 1.0 / S); np.fill_diagonal(A, 0.6)
        A = A / A.sum(axis=1, keepdims=True)
        labels = rng.integers(0, S, size=T)
        mu = np.zeros((S, L)); var = np.zeros((S, L))
        for s in range(S):
            xs = X[labels == s] if (labels == s).any() else X
            mu[s] = xs.mean(axis=0)
            var[s] = xs.var(axis=0) + 1e-3
        eps = X.mean(axis=0)
        tau2 = X.var(axis=0) + 1e-2
        rho = np.clip(0.8 + 0.2 * rng.random(L), 0.5, 0.99)
        return pi, A, mu, var, eps, tau2, rho

    def _emission_logs(self, X: np.ndarray):
        T, L = X.shape; S = self.n_states
        log_r = np.empty((T, S, L)); log_q = np.empty((T, L))
        for l in range(L):
            for s in range(S):
                v = 1e-12 + max(self.var_[s, l], self.var_floor)
                log_r[:, s, l] = -0.5 * (np.log(2*np.pi*v) + ((X[:, l]-self.mu_[s, l])**2)/v)
            v = 1e-12 + max(self.tau2_[l], self.var_floor)
            log_q[:, l] = -0.5 * (np.log(2*np.pi*v) + ((X[:, l]-self.eps_[l])**2)/v)
        return log_r, log_q

    def _forward_backward(self, log_B: np.ndarray):
        T, S = log_B.shape
        log_pi = np.log(self.pi_ + 1e-300)
        log_A  = np.log(self.A_  + 1e-300)
        log_alpha = np.empty((T, S))
        log_alpha[0] = log_pi + log_B[0]
        for t in range(1, T):
            log_alpha[t] = safe_logsumexp(log_alpha[t-1][:, None] + log_A, axis=0) + log_B[t]
        loglik = safe_logsumexp(log_alpha[-1], axis=0)
        log_beta = np.zeros((T, S))
        for t in range(T-2, -1, -1):
            log_beta[t] = safe_logsumexp(log_A + log_B[t+1] + log_beta[t+1], axis=1)
        gamma = np.exp(log_alpha + log_beta - safe_logsumexp(log_alpha + log_beta, axis=1)[:, None])
        xi = np.empty((T-1, S, S))
        for t in range(T-1):
            log_xi_t = log_alpha[t][:, None] + log_A + log_B[t+1][None, :] + log_beta[t+1][None, :]
            xi[t] = np.exp(log_xi_t - safe_logsumexp(log_xi_t))
        return gamma, xi, float(loglik)

    def fit(self, X: np.ndarray):
        T, L = X.shape
        if self.k is None:
            self.k = max(1.0, T/4.0)
        (self.pi_, self.A_, self.mu_, self.var_,
         self.eps_, self.tau2_, self.rho_) = self._init_params(X, self.n_states, self.rng)
        prev_ll = -np.inf
        for _ in range(self.max_iter):
            log_r, log_q = self._emission_logs(X)
            rho = np.clip(self.rho_, RHO_MIN, RHO_MAX)
            log_mix = np.empty((T, self.n_states, L))
            for l in range(L):
                a = np.log(rho[l])   + log_r[:, :, l]
                b = np.log(1-rho[l]) + np.broadcast_to(log_q[:, l][:, None], (T, self.n_states))
                log_mix[:, :, l] = safe_logsumexp(np.stack([a, b]), axis=0)
            log_B = np.sum(log_mix, axis=2)
            gamma, xi, ll = self._forward_backward(log_B)
            # M-step
            u = np.empty((T, self.n_states, L)); v = np.empty((T, self.n_states, L))
            for l in range(L):
                num = np.log(rho[l]) + log_r[:, :, l]; den = log_mix[:, :, l]
                u[:, :, l] = np.exp(num - den) * gamma
                v[:, :, l] = gamma - u[:, :, l]
            self.pi_ = np.maximum(gamma[0], 1e-12); self.pi_ /= self.pi_.sum()
            self.A_  = np.maximum(xi.sum(axis=0), 1e-12); self.A_ /= self.A_.sum(axis=1, keepdims=True)
            u_sum = u.sum(axis=0); v_sum = v.sum(axis=0)
            for s in range(self.n_states):
                w = u_sum[s] + 1e-12
                self.mu_[s]  = (u[:, s, :]*X).sum(axis=0)/w
                diff2 = (X - self.mu_[s])**2
                self.var_[s] = np.maximum((u[:, s, :]*diff2).sum(axis=0)/w, FSHMM_VAR_FLOOR)
            v_all = v_sum.sum(axis=0)
            self.eps_  = (v*X[:, None, :]).sum(axis=(0, 1))/np.maximum(v_all, 1e-12)
            diff2_q = (X[:, None, :] - self.eps_)**2
            self.tau2_ = np.maximum((v*diff2_q).sum(axis=(0, 1))/np.maximum(v_all, 1e-12), FSHMM_VAR_FLOOR)
            u_all = u_sum.sum(axis=0)
            k_eff = (self.k if self.k is not None else 1.0)
            T_hat = (T + 1 + k_eff)
            disc = np.maximum(T_hat*T_hat - 4.0*k_eff*u_all, 0.0)
            self.rho_ = np.clip((T_hat - np.sqrt(disc)) / (2.0*k_eff + 1e-12), RHO_MIN, RHO_MAX)
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        return self

# ============================== HMM train / BIC / Convergence ==============================
def _sticky_transmat_prior(K: int, sticky_strength: float = 0.90, concentration: float = 100.0) -> np.ndarray:
    base = np.full((K, K), (1.0 - sticky_strength) / (K - 1))
    np.fill_diagonal(base, sticky_strength)
    return np.maximum(base * concentration, 1e-6)

def _ergodic_transmat(K: int, self_loop: float = 0.90) -> np.ndarray:
    A = np.full((K, K), (1.0 - self_loop) / max(1, K - 1))
    np.fill_diagonal(A, self_loop)
    return A / A.sum(axis=1, keepdims=True)

def _repair_stochastic_rows(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    row_sums = M.sum(axis=1, keepdims=True)
    dead = (row_sums <= eps).ravel()
    if np.any(dead):
        K = M.shape[1]
        M = M.copy()
        M[dead, :] = 1.0 / K
    M = np.maximum(M, eps)
    M /= M.sum(axis=1, keepdims=True)
    return M

def _fit_hmm_once(X: np.ndarray, K: int,
                  covariance_type: str, n_iter: int, tol: float,
                  min_covar: float, self_loop: float, startprob_prior: float, n_init: int) -> GaussianHMM:
    rng = np.random.default_rng(SEED)
    best_model, best_score = None, -np.inf
    transmat_prior = _sticky_transmat_prior(K, sticky_strength=self_loop, concentration=100.0)
    for _ in range(n_init):
        model = GaussianHMM(
            n_components=K, covariance_type=covariance_type,
            n_iter=n_iter, tol=tol, min_covar=min_covar,
            random_state=int(rng.integers(0, 1_000_000)),
            init_params="mc", params="stmc", verbose=False
        )
        model.startprob_prior = np.full(K, float(startprob_prior))
        model.transmat_prior  = transmat_prior.copy()
        model.startprob_ = np.full(K, 1.0 / K)
        model.transmat_  = _ergodic_transmat(K, self_loop=self_loop)
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(X)
        model.startprob_ = np.maximum(model.startprob_, 1e-12); model.startprob_ /= model.startprob_.sum()
        model.transmat_  = _repair_stochastic_rows(model.transmat_, eps=1e-12)
        score = model.score(X)
        if score > best_score:
            best_model, best_score = model, score
    return best_model

def train_hmm_robust_until_converged(X: np.ndarray, K: int, verbose: bool = True) -> GaussianHMM:
    attempts = [
        {"cov":"diag","n_iter":1000,"tol":1e-3,"min_covar":1e-3,"self_loop":0.90,"n_init":5},
        {"cov":"diag","n_iter":1500,"tol":5e-4,"min_covar":1e-3,"self_loop":0.92,"n_init":8},
        {"cov":"diag","n_iter":2000,"tol":1e-4,"min_covar":3e-3,"self_loop":0.94,"n_init":10},
        {"cov":"diag","n_iter":2500,"tol":1e-4,"min_covar":1e-2,"self_loop":0.95,"n_init":12},
        {"cov":"full","n_iter":2000,"tol":5e-4,"min_covar":3e-3,"self_loop":0.92,"n_init":8},
        {"cov":"full","n_iter":3000,"tol":1e-4,"min_covar":1e-2,"self_loop":0.95,"n_init":12},
    ]
    for i, cfg in enumerate(attempts, 1):
        if verbose: print(f"[HMM converge try {i}/{len(attempts)}] cfg={cfg}")
        m = _fit_hmm_once(
            X, K,
            covariance_type=cfg["cov"], n_iter=cfg["n_iter"], tol=cfg["tol"],
            min_covar=cfg["min_covar"], self_loop=cfg["self_loop"],
            startprob_prior=DEF_STARTPROB_PRIOR, n_init=cfg["n_init"]
        )
        if getattr(m, "monitor_", None) is not None and m.monitor_.converged:
            if verbose: print(f"Converged with cfg={cfg}")
            return m
        if verbose:
            last_delta = (m.monitor_.history[-1] if getattr(m, "monitor_", None) and m.monitor_.history else None)
            print(f"Not converged; last delta ~ {last_delta}")
    print("WARNING: returning last attempt model (not strictly converged).")
    return m

def hmm_bic(model: GaussianHMM, X: np.ndarray, covariance_type: str = "diag") -> float:
    K = model.n_components
    T, d = X.shape
    loglik = model.score(X)
    if covariance_type == "full":
        p = (K - 1) + K*(K - 1) + K*d + K*(d*(d+1)//2)
    else:
        p = (K - 1) + K*(K - 1) + K*d + K*d
    return -2.0 * loglik + p * np.log(max(T, 1))

# ============================== K selection & plotting ==============================
def plot_bic_curve(ks: List[int], bics: List[float], tag: str):
    fig = plt.figure(figsize=(6, 4))
    plt.plot(ks, bics, marker="o")
    plt.xlabel("K"); plt.ylabel("BIC (lower is better)"); plt.title(f"BIC vs K — {tag}")
    save_and_show(fig, os.path.join(NOW_DIR, f"bic_{tag}.png"))

def select_k_via_bic_and_plot(X_train: np.ndarray, tag: str) -> int:
    ks, bics = [], []
    for K in BIC_K_RANGE:
        m = _fit_hmm_once(
            X_train, K,
            covariance_type=DEF_COVARIANCE_TYPE, n_iter=DEF_N_ITER, tol=DEF_TOL,
            min_covar=DEF_MIN_COVAR, self_loop=DEF_SELF_LOOP,
            startprob_prior=DEF_STARTPROB_PRIOR, n_init=DEF_N_INIT
        )
        ks.append(K); bics.append(hmm_bic(m, X_train, DEF_COVARIANCE_TYPE))
    bic_df = pd.DataFrame({"K": ks, "BIC": bics}).sort_values("K")
    bic_df.to_csv(os.path.join(NOW_DIR, f"bic_{tag}.csv"), index=False)
    plot_bic_curve(ks, bics, tag)
    k_best = int(bic_df.loc[bic_df["BIC"].idxmin(), "K"])
    print(f"[{tag}] BIC-selected K: {k_best}")
    return k_best

# ============================== Plot: step line with colored bands ==============================
def _state_colors(K: int) -> List[tuple]:
    base = plt.get_cmap("tab10").colors
    if K <= 10:
        return [base[i] for i in range(K)]
    extra = plt.get_cmap("tab20").colors
    colors = list(base) + list(extra)
    return [colors[i % len(colors)] for i in range(K)]

import matplotlib.patches as mpatches

def plot_regime(series: pd.Series, title: str, save_path: str,
                                        split_date: Optional[str] = None,
                                        bg_alpha: float = 0.18, line_color: str = "k"):
    s = series.dropna()
    if s.empty:
        print(f"[WARN] Empty series for plot: {title}")
        return
    K = int(np.nanmax(s.values))
    colors = _state_colors(K)

    # infer step size for the last segment
    idx = s.index
    try:
        # works if PeriodIndex-like monthly freq
        freq = pd.infer_freq(idx)
        if freq is not None:
            step = pd.tseries.frequencies.to_offset(freq)
        else:
            raise ValueError
    except Exception:
        diffs = pd.Series(idx[1:]).reset_index(drop=True) - pd.Series(idx[:-1]).reset_index(drop=True)
        step = diffs.median() if len(diffs) else pd.Timedelta(days=30)

    # build right edges: next timestamp for all but last; last extends by one step
    left_edges = list(idx)
    right_edges = list(idx[1:]) + [idx[-1] + step]

    fig = plt.figure(figsize=(12, 3))
    ax = plt.gca()

    # draw background bands for each contiguous regime segment using [left, right_next)
    vals = s.values.astype(int)
    seg_start = 0
    for t in range(1, len(vals)+1):
        if (t == len(vals)) or (vals[t] != vals[t-1]):
            left  = left_edges[seg_start]
            right = right_edges[t-1]
            ax.axvspan(left, right,
                       facecolor=colors[vals[seg_start]-1],
                       alpha=bg_alpha, edgecolor="none", zorder=0)
            seg_start = t

    # step line on top
    ax.step(s.index, s.values, where="post", color=line_color, lw=1.8, zorder=1)

    # dashed vertical split
    if split_date is not None:
        ax.axvline(pd.to_datetime(split_date), linestyle="--", color="k", lw=1.0, zorder=2)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Regime")
    ax.set_yticks(range(1, K+1))

    patches = [mpatches.Patch(color=colors[k-1], label=f"Regime {k}", alpha=bg_alpha) for k in range(1, K+1)]
    ax.legend(handles=patches, ncol=min(K, 5), fontsize=8, frameon=False, loc="upper left")

    save_and_show(fig, save_path)


# ============================== Pipeline helpers ==============================
def add_soft_weights(Z: pd.DataFrame, rho: pd.Series, eps: float = 0.05) -> pd.DataFrame:
    rho = rho.reindex(Z.columns).fillna(0.0).clip(lower=0.0, upper=1.0)
    w = np.sqrt(np.maximum(rho.values, float(eps)))
    Zw = Z.values * w[None, :]
    return pd.DataFrame(Zw, index=Z.index, columns=Z.columns)

# ============================== Main ==============================
if __name__ == "__main__":
    # 0) Load features
    top_df_raw = read_top_csv(TOP_FEATURES_PATH, min_date=GLOBAL_MIN_DATE)

    # 1) Preprocess
    top_df = winsorize_df(top_df_raw, WINSOR_P) if WINSOR_P and WINSOR_P > 0 else top_df_raw.copy()

    # 2) Split
    tr_mask = (top_df.index >= pd.to_datetime(TRAIN_START)) & (top_df.index <= pd.to_datetime(TRAIN_END))
    df_train = top_df.loc[tr_mask].copy()

    # 3) Standardize by TRAIN only
    if USE_ZSCORE:
        mu_tr, sd_tr = zfit(df_train)
        Z_train = zapply(df_train, mu_tr, sd_tr)
        Z_full  = zapply(top_df,  mu_tr, sd_tr)
    else:
        Z_train, Z_full = df_train.copy(), top_df.copy()

    outputs = {}

    # -------- FSW branch (primary) --------
    if RUN_FSW:
        # 4) FSHMM saliency on TRAIN
        fsel = FSHMM(
            n_states=FSHMM_K_FOR_SELECTION, max_iter=FSHMM_MAX_ITERS, tol=FSHMM_TOL,
            k=max(1.0, len(Z_train)/4.0), var_floor=FSHMM_VAR_FLOOR, random_state=SEED
        )
        fsel.fit(Z_train.values.astype(float))
        rho = pd.Series(fsel.rho_, index=Z_train.columns, name="rho").sort_values(ascending=False)
        rho.to_csv(os.path.join(SEL_DIR, "rho_full_train.csv"))

        # optional saliency bar
        fig = plt.figure(figsize=(10, max(5, len(rho.head(TOPN_FOR_DISPLAY))*0.28)))
        rho.head(TOPN_FOR_DISPLAY).sort_values(ascending=True).plot(kind="barh")
        plt.xlabel("rho (saliency)"); plt.ylabel("feature")
        plt.title(f"Top-{TOPN_FOR_DISPLAY} Feature Saliency (FSHMM on TRAIN)")
        save_and_show(fig, os.path.join(SEL_DIR, f"rho_top{TOPN_FOR_DISPLAY}_train.png"))

        # 5) Build soft-weighted matrices (keep all columns)
        Z_train_w = add_soft_weights(Z_train, rho, eps=FSHMM_WEIGHT_EPS)
        Z_full_w  = add_soft_weights(Z_full,  rho, eps=FSHMM_WEIGHT_EPS)

        # 6) K selection via BIC (TRAIN, weighted)
        k_best = select_k_via_bic_and_plot(Z_train_w.values.astype(float), tag="FSW_train")

        # 7) Final HMM with robust convergence attempts
        hmm = train_hmm_robust_until_converged(Z_train_w.values.astype(float), K=k_best, verbose=True)

        # 8) TRAIN path
        with contextlib.redirect_stdout(io.StringIO()):
            st_tr = hmm.predict(Z_train_w.values.astype(float)) + 1
        reg_train = pd.Series(st_tr, index=Z_train_w.index, name=f"Regime_FSW_K{k_best}_TRAIN")
        reg_train.to_csv(os.path.join(NOW_DIR, f"regime_path_train_K{k_best}_FSW.csv"))
        plot_regime(reg_train, f"TRAIN Regime Path — FSW (K={k_best})",
                                            os.path.join(NOW_DIR, f"regime_path_train_K{k_best}_FSW.png"))

        # 9) OOS prefix decode (strict no-look-ahead)
        Z_full_w_dates = Z_full_w.index
        oos_start_idx = np.searchsorted(Z_full_w_dates, pd.to_datetime(OOS_START))
        oos_states, oos_dates = [], []
        for t in range(oos_start_idx, len(Z_full_w_dates)):
            X_prefix = Z_full_w.iloc[:t+1].values.astype(float)
            with contextlib.redirect_stdout(io.StringIO()):
                path = hmm.predict(X_prefix)
            oos_states.append(int(path[-1] + 1)); oos_dates.append(Z_full_w_dates[t])
        reg_oos = pd.Series(oos_states, index=pd.DatetimeIndex(oos_dates), name=f"Regime_FSW_K{k_best}_OOS")
        reg_oos.to_csv(os.path.join(NOW_DIR, f"regime_path_oos_K{k_best}_FSW.csv"))
        plot_regime(reg_oos, f"OOS Regime Path — FSW (K={k_best})",
                                            os.path.join(NOW_DIR, f"regime_path_oos_K{k_best}_FSW.png"))

        # 10) FULL path + dashed split
        reg_full = pd.concat([reg_train, reg_oos]).sort_index()
        reg_full.name = f"Regime_FSW_K{k_best}_FULL"
        reg_full.to_csv(os.path.join(NOW_DIR, f"regime_path_full_K{k_best}_FSW.csv"))
        plot_regime(reg_full, f"FULL Regime Path — FSW (K={k_best})",
                                            os.path.join(NOW_DIR, f"regime_path_full_K{k_best}_FSW.png"),
                                            split_date=OOS_START)

        outputs["FSW"] = {"k": k_best, "full": reg_full}

    # -------- RAW branch (optional) --------
    if RUN_RAW:
        k_best_raw = select_k_via_bic_and_plot(Z_train.values.astype(float), tag="RAW_train")
        hmm_raw = train_hmm_robust_until_converged(Z_train.values.astype(float), K=k_best_raw, verbose=True)
        with contextlib.redirect_stdout(io.StringIO()):
            st_tr_raw = hmm_raw.predict(Z_train.values.astype(float)) + 1
        reg_train_raw = pd.Series(st_tr_raw, index=Z_train.index, name=f"Regime_RAW_K{k_best_raw}_TRAIN")
        reg_train_raw.to_csv(os.path.join(NOW_DIR, f"regime_path_train_K{k_best_raw}_RAW.csv"))
        plot_regime(reg_train_raw, f"TRAIN Regime Path — RAW (K={k_best_raw})",
                                            os.path.join(NOW_DIR, f"regime_path_train_K{k_best_raw}_RAW.png"))

        Z_full_dates = Z_full.index
        oos_start_idx2 = np.searchsorted(Z_full_dates, pd.to_datetime(OOS_START))
        oos_states_raw, oos_dates_raw = [], []
        for t in range(oos_start_idx2, len(Z_full_dates)):
            Xp = Z_full.iloc[:t+1].values.astype(float)
            with contextlib.redirect_stdout(io.StringIO()):
                path = hmm_raw.predict(Xp)
            oos_states_raw.append(int(path[-1] + 1)); oos_dates_raw.append(Z_full_dates[t])
        reg_oos_raw = pd.Series(oos_states_raw, index=pd.DatetimeIndex(oos_dates_raw), name=f"Regime_RAW_K{k_best_raw}_OOS")
        reg_oos_raw.to_csv(os.path.join(NOW_DIR, f"regime_path_oos_K{k_best_raw}_RAW.csv"))
        plot_regime(reg_oos_raw, f"OOS Regime Path — RAW (K={k_best_raw})",
                                            os.path.join(NOW_DIR, f"regime_path_oos_K{k_best_raw}_RAW.png"))

        reg_full_raw = pd.concat([reg_train_raw, reg_oos_raw]).sort_index()
        reg_full_raw.name = f"Regime_RAW_K{k_best_raw}_FULL"
        reg_full_raw.to_csv(os.path.join(NOW_DIR, f"regime_path_full_K{k_best_raw}_RAW.csv"))
        plot_regime(reg_full_raw, f"FULL Regime Path — RAW (K={k_best_raw})",
                                            os.path.join(NOW_DIR, f"regime_path_full_K{k_best_raw}_RAW.png"),
                                            split_date=OOS_START)

        outputs["RAW"] = {"k": k_best_raw, "full": reg_full_raw}

    # 11) Save a single file for downstream diagnostics
    diag_df = pd.DataFrame(index=top_df.index)
    if "FSW" in outputs:
        diag_df["Regime_FSW"] = outputs["FSW"]["full"].reindex(diag_df.index)
    if "RAW" in outputs:
        diag_df["Regime_RAW"] = outputs["RAW"]["full"].reindex(diag_df.index)
    diag_df = diag_df.dropna(how="all")
    diag_df.to_csv(os.path.join(NOW_DIR, "regimes_for_diagnostics.csv"))

    print("\nNowcast complete. Files saved under:", NOW_DIR)
