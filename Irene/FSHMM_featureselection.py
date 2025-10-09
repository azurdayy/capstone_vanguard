"""
FSHMM Regime Identification (K=3 & K=4) — show + save figures
- Reads:   Data/combined_features.xlsx   (rows >= 2001-12-01)
- Shows & Saves:
    (1) Top-30 feature saliency ρ  →  Data/figures/saliency_top30_K{K}.png
    (2) FSHMM regime PATH (states 1..K) → Data/figures/regime_path_fshmm_K{K}.png
    (3) (optional) Full-cov HMM regime PATH → Data/figures/regime_path_fullhmm_K{K}.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from typing import Tuple, Optional

# -----------------------------
# User knobs
# -----------------------------
EXCEL_PATH = "Data/combined_features.xlsx"
OUT_DIR = "FSHMM_result"
os.makedirs(OUT_DIR, exist_ok=True)

DATE_COL = None
SHEET_NAME = 0
START_DATE = "2001-12-01"

WINSOR_P = 0.0
Z_SCORE = True

TRY_K_LIST = [2, 3, 4]
MAX_ITERS = 500
TOL = 1e-4
VAR_FLOOR = 1e-6
RHO_EPS = 1e-9
SEED = 42

MAX_TOP_FEATURES_TO_SHOW = 30
RHO_THRESHOLD = 0.5
TRY_REFIT_FULL_HMM = True   # 用于可选的 full-cov HMM

rng = np.random.default_rng(SEED)

# -----------------------------
# Utilities
# -----------------------------
def read_data(path: str) -> pd.DataFrame:
    if DATE_COL is None:
        df = pd.read_excel(path, sheet_name=SHEET_NAME, index_col=0)
    else:
        df = pd.read_excel(path, sheet_name=SHEET_NAME)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        df = df.set_index(DATE_COL)
    if not np.issubdtype(df.index.dtype, np.datetime64):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[df.index >= pd.to_datetime(START_DATE)]
    df = df.dropna(axis=1, how='all').ffill().bfill()
    return df

def winsorize_df(df: pd.DataFrame, p: float) -> pd.DataFrame:
    if p <= 0: return df
    q_low = df.quantile(p); q_high = df.quantile(1 - p)
    return df.clip(lower=q_low, upper=q_high, axis=1)

def zscore_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    mu = df.mean(axis=0)
    sd = df.std(axis=0).replace(0.0, 1.0)
    return (df - mu) / sd, mu, sd

def safe_logsumexp(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True) + 1e-300)
    return np.squeeze(out, axis=axis)

# -----------------------------
# FSHMM (diagonal Gaussian emissions)
# -----------------------------
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
        self.pi_ = None; self.A_ = None
        self.mu_ = None; self.var_ = None
        self.eps_ = None; self.tau2_ = None
        self.rho_ = None
        self.loglik_ = []

    @staticmethod
    def _init_params(X: np.ndarray, S: int, rng):
        T, L = X.shape
        pi = np.full(S, 1.0 / S)
        A = np.full((S, S), 1.0 / S); A[np.arange(S), np.arange(S)] = 0.6
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

    def _log_emission_mix(self, X: np.ndarray):
        T, L = X.shape; S = self.n_states
        log_r = np.empty((T, S, L))
        for l in range(L):
            for s in range(S):
                v = max(self.var_[s, l], self.var_floor)
                log_r[:, s, l] = -0.5 * (np.log(2 * np.pi * v) + ((X[:, l] - self.mu_[s, l]) ** 2) / v)
        log_q = np.empty((T, L))
        for l in range(L):
            v = max(self.tau2_[l], self.var_floor)
            log_q[:, l] = -0.5 * (np.log(2 * np.pi * v) + ((X[:, l] - self.eps_[l]) ** 2) / v)
        return log_r, log_q

    def _forward_backward(self, log_B: np.ndarray):
        T, S = log_B.shape
        log_pi = np.log(self.pi_ + 1e-300)
        log_A = np.log(self.A_ + 1e-300)
        log_alpha = np.empty((T, S)); log_alpha[0] = log_pi + log_B[0]
        for t in range(1, T):
            tmp = log_alpha[t - 1][:, None] + log_A
            log_alpha[t] = safe_logsumexp(tmp, axis=0) + log_B[t]
        loglik = safe_logsumexp(log_alpha[-1], axis=0)
        log_beta = np.empty((T, S)); log_beta[-1] = 0.0
        for t in range(T - 2, -1, -1):
            tmp = log_A + (log_B[t + 1] + log_beta[t + 1])[None, :]
            log_beta[t] = safe_logsumexp(tmp, axis=1)
        log_gamma = log_alpha + log_beta
        log_gamma = (log_gamma.T - safe_logsumexp(log_gamma, axis=1)).T
        gamma = np.exp(log_gamma)
        log_xi = np.empty((T - 1, S, S))
        for t in range(T - 1):
            log_xi[t] = (log_alpha[t][:, None] + log_A + log_B[t + 1][None, :] + log_beta[t + 1][None, :])
            log_xi[t] -= safe_logsumexp(log_xi[t], axis=None)
        xi = np.exp(log_xi)
        return gamma, xi, float(loglik)

    def fit(self, X: np.ndarray):
        T, L = X.shape; S = self.n_states
        if self.k is None: self.k = max(1.0, T / 4.0)
        (self.pi_, self.A_, self.mu_, self.var_,
         self.eps_, self.tau2_, self.rho_) = self._init_params(X, S, self.rng)
        prev_ll = -np.inf; self.loglik_ = []

        for _ in range(self.max_iter):
            log_r, log_q = self._log_emission_mix(X)
            rho = np.clip(self.rho_, RHO_EPS, 1.0 - RHO_EPS)
            log_mix = np.empty((T, S, L))
            for l in range(L):
                a = np.log(rho[l]) + log_r[:, :, l]
                b = np.log(1 - rho[l]) + np.broadcast_to(log_q[:, l][:, None], (T, S))
                tmp = np.stack([a, b], axis=0)
                log_mix[:, :, l] = safe_logsumexp(tmp, axis=0)
            log_B = np.sum(log_mix, axis=2)
            gamma, xi, ll = self._forward_backward(log_B)
            self.loglik_.append(ll)

            u = np.empty((T, S, L)); v = np.empty((T, S, L))
            for l in range(L):
                num = np.log(rho[l]) + log_r[:, :, l]
                den = log_mix[:, :, l]
                u[:, :, l] = np.exp(num - den) * gamma
                v[:, :, l] = gamma - u[:, :, l]

            self.pi_ = np.maximum(gamma[0], 1e-12); self.pi_ /= self.pi_.sum()
            self.A_  = np.maximum(xi.sum(axis=0), 1e-12); self.A_ /= self.A_.sum(axis=1, keepdims=True)

            u_sum = u.sum(axis=0); v_sum = v.sum(axis=0)
            for s in range(S):
                w = u_sum[s] + 1e-12
                num_mu = (u[:, s, :] * X).sum(axis=0)
                self.mu_[s] = num_mu / w
                diff2 = (X - self.mu_[s]) ** 2
                num_var = (u[:, s, :] * diff2).sum(axis=0)
                self.var_[s] = np.maximum(num_var / w, self.var_floor)

            v_all = v_sum.sum(axis=0)
            num_eps = (v * X[:, None, :]).sum(axis=(0, 1))
            self.eps_ = num_eps / np.maximum(v_all, 1e-12)
            diff2_q = (X[:, None, :] - self.eps_) ** 2
            num_tau = (v * diff2_q).sum(axis=(0, 1))
            self.tau2_ = np.maximum(num_tau / np.maximum(v_all, 1e-12), self.var_floor)

            u_all = u_sum.sum(axis=0)
            T_hat = (T + 1 + self.k)
            disc = np.maximum(T_hat * T_hat - 4.0 * self.k * u_all, 0.0)
            rho_new = (T_hat - np.sqrt(disc)) / (2.0 * self.k + 1e-12)
            self.rho_ = np.clip(rho_new, 1e-9, 1 - 1e-9)

            if len(self.loglik_) > 1 and abs(self.loglik_[-1] - prev_ll) < self.tol: break
            prev_ll = self.loglik_[-1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        log_r, log_q = self._log_emission_mix(X)
        rho = np.clip(self.rho_, RHO_EPS, 1.0 - RHO_EPS)
        T = X.shape[0]
        log_mix = np.empty((T, self.n_states, X.shape[1]))
        for l in range(X.shape[1]):
            a = np.log(rho[l]) + log_r[:, :, l]
            b = np.log(1 - rho[l]) + np.broadcast_to(log_q[:, l][:, None], (T, self.n_states))
            tmp = np.stack([a, b], axis=0)
            log_mix[:, :, l] = safe_logsumexp(tmp, axis=0)
        log_B = np.sum(log_mix, axis=2)
        # Viterbi
        T, S = log_B.shape
        log_pi = np.log(self.pi_ + 1e-300)
        log_A = np.log(self.A_ + 1e-300)
        delta = np.empty((T, S)); psi = np.zeros((T, S), dtype=int)
        delta[0] = log_pi + log_B[0]
        for t in range(1, T):
            tmp = delta[t - 1][:, None] + log_A
            psi[t] = np.argmax(tmp, axis=0)
            delta[t] = np.max(tmp, axis=0) + log_B[t]
        states = np.empty(T, dtype=int)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

def try_fit_full_cov_hmm(X: np.ndarray, n_states: int, random_state: int = 0):
    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception:
        return None
    model = GaussianHMM(
        n_components=n_states, covariance_type='full',
        n_iter=500, tol=1e-3, random_state=random_state
    )
    model.fit(X)
    return model

# -----------------------------
# Run
# -----------------------------
df_raw = read_data(EXCEL_PATH)
df_proc = winsorize_df(df_raw, WINSOR_P)
nunique = df_proc.nunique()
const_cols = nunique[nunique <= 1].index.tolist()
if const_cols: df_proc = df_proc.drop(columns=const_cols)
df_z, _, _ = zscore_df(df_proc) if Z_SCORE else (df_proc.copy(), None, None)

X = df_z.values.astype(float)
dates = df_z.index
feat_names = df_z.columns.tolist()
T, L = X.shape
K_prior = max(1.0, T / 4.0)

for K in TRY_K_LIST:
    print(f"\n================  K = {K}  ================\n")

    fshmm = FSHMM(n_states=K, max_iter=MAX_ITERS, tol=TOL, k=K_prior,
                  var_floor=VAR_FLOOR, random_state=SEED)
    fshmm.fit(X)

    # ---- Top-30 saliency ----
    rho = pd.Series(fshmm.rho_, index=feat_names, name=f"rho_K{K}")
    rho.sort_values(ascending=False).to_csv(f"{OUT_DIR}/saliency_full_K{K}.csv")

    rho_top = rho.sort_values(ascending=False).head(MAX_TOP_FEATURES_TO_SHOW)
    print("Top-30 Feature Saliency (ρ):")
    display(rho_top.to_frame())

    plt.figure(figsize=(10, max(6, len(rho_top) * 0.3)))
    rho_top.sort_values(ascending=True).plot(kind="barh")
    plt.xlabel("ρ (saliency)"); plt.ylabel("Feature")
    plt.title(f"Top-30 Feature Saliency ρ (FSHMM, K={K})")
    plt.axvline(0.5, linestyle="--")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/saliency_top30_K{K}.png", dpi=200)   
    plt.show()                                                 

    # ---- Regime PATH (discrete states 1..K) ----
    states = fshmm.predict(X)   # (T,)
    plt.figure(figsize=(12, 3))
    plt.step(dates, states + 1, where="post", linewidth=1.2)
    plt.yticks(range(1, K + 1))
    plt.xlabel("Date"); plt.ylabel("Regime")
    plt.title(f"FSHMM Regime Path (K={K})")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/regime_path_fshmm_K{K}.png", dpi=200) 
    plt.show()                                                    

    # ---- Optional: Full-cov HMM on FS features, PATH only ----
    if TRY_REFIT_FULL_HMM:
        sel_names = rho[rho >= RHO_THRESHOLD].sort_values(ascending=False).index.tolist()
        if len(sel_names) == 0:
            sel_names = rho_top.index.tolist()
        X_sel = df_z[sel_names].values
        full = try_fit_full_cov_hmm(X_sel, n_states=K, random_state=SEED)
        if full is not None:
            path = full.predict(X_sel)   # (T,)
            plt.figure(figsize=(12, 3))
            plt.step(dates, path + 1, where="post", linewidth=1.2)
            plt.yticks(range(1, K + 1))
            plt.xlabel("Date"); plt.ylabel("Regime")
            plt.title(f"Full-cov HMM Regime Path (K={K})")
            plt.tight_layout()
            plt.savefig(f"{OUT_DIR}/regime_path_fullhmm_K{K}.png", dpi=200)  
            plt.show()                                                      
