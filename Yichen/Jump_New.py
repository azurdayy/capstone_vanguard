import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import Enum
from typing import Tuple, Dict, List, Optional, Union, Any

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import optuna
from collections import OrderedDict

PROJECT_DIR = Path(__file__).resolve()
sys.path.insert(0, str(PROJECT_DIR))

from Initialization import Initialization
from Datafactory import normalized
from HelperFunctions import HelperFunction

class PenaltyType(Enum):
    L0 = "l0"
    LASSO = "lasso"
    RIDGE = "ridge"
    GROUP_LASSO = "group_lasso"

    def compute(self, mu: np.ndarray) -> float:
        """
        Compute penalty value for a given cluster centers matrix mu (K x p).
        Each column j corresponds to feature j.
        """
        if self == PenaltyType.L0:
            # P0(mu) = sum_j 1{||mu_{.,j}|| > 0}
            return np.sum(np.linalg.norm(mu, axis=0) > 0)
        elif self == PenaltyType.LASSO:
            # P1(mu) = sum_j ||mu_{.,j}||_1
            return np.sum(np.linalg.norm(mu, ord=1, axis=0))
        elif self == PenaltyType.RIDGE:
            # P2(mu) = sum_j ||mu_{.,j}||^2
            return np.sum(np.linalg.norm(mu, ord=2, axis=0) ** 2)
        elif self == PenaltyType.GROUP_LASSO:
            # P3(mu) = sum_j ||mu_{.,j}||
            return np.sum(np.linalg.norm(mu, ord=2, axis=0))
        else:
            raise ValueError(f"Unsupported penalty type: {self}")

class JumpModel(ABC):
    """Base class for Jump Models."""
    def __init__(self, k: int = 2, lmbd: float = 1.0):
        """k = number of states, lmbd = penalty weight"""
        self.k = k
        self.mu = None
        self.s = None
        self.lmbd = lmbd
        self.regime_series = None
        self.data = None
        self.T = None
        self.p = None
    
    def initialize(self):
        raise NotImplementedError
    
    def calibrate(self):
        raise NotImplementedError

class RegularizedJumpModel(JumpModel):
    def __init__(
            self, 
            k: int = 2, 
            lmbd: float = 1.0, 
            gamma: float = 1.0,
            penalty: str | PenaltyType = "lasso"
    ):
        super().__init__(k, lmbd)
        self.gamma = gamma
        self.feature_score = None
        self.initial_state = None
        self.feature_names: List[str] | None = None
        self.regime_series = None

        if isinstance(penalty, str):
            penalty = penalty.lower()
            try:
                self.penalty = PenaltyType(penalty)
            except ValueError:
                raise ValueError(
                    f"Invalid penalty string '{penalty}'. "
                    f"Must be one of {[p.value for p in PenaltyType]}"
                )
        elif isinstance(penalty, PenaltyType):
            self.penalty = penalty
        else:
            raise TypeError("penalty must be str or PenaltyType")
    
    def input_data(self, data: pd.DataFrame):
        if data.empty:
            raise ValueError("Input data is empty")
        self.raw_data = data.copy()
        self.data = data.copy()
        if not np.issubdtype(self.data.dtypes.values[0], np.number):
            raise TypeError("Data must be numeric")
        if (self.data.isna().sum().sum() > 0):
            raise ValueError("Data contains NaN values")
        if (self.data.std(ddof=0) == 0).any():
            raise ValueError("Data contains constant columns")
        self.data = normalized(self.data)
        self.T, self.p = self.data.shape
        self.feature_names = list(self.data.columns)


    def initialize(self, random_state: int = 42):
        result = Initialization.regularized(self.data, self.k, random_state)
        self.feature_score = result["d"]
        # shift KMeans labels (0..k-1) to 1..k for internal consistency
        self.initial_state = {pct: (lbls.astype(int) + 1) 
                            for pct, lbls in result["state_sequences"].items()}

    def compute_penalty(self, mu: pd.DataFrame | np.ndarray) -> float:
        mu_arr = mu.to_numpy() if isinstance(mu, pd.DataFrame) else np.asarray(mu)
        return self.penalty.compute(mu_arr)

    def _compute_model_score(self, mu_df: pd.DataFrame, s: np.ndarray) -> float:
        """
        Compute model score using the original data:
            score = reconstruction error + λ * jumps + γ * regularization penalty.

        Parameters
        ----------
        mu_df : pd.DataFrame
            (K x p) matrix of state means.
        s : np.ndarray
            (T,) array of integer state labels (1..K).

        Returns
        -------
        float : total score
        """
        if mu_df is None:
            mu = self.mu
        if s is None:
            s = self.s
        Y = self.data.to_numpy()
        mu = mu_df.to_numpy()
        lmbd, gamma = float(self.lmbd), float(self.gamma)

        # Reconstruction error
        recon = np.sum((Y - mu[s - 1]) ** 2)

        # Jump count penalty
        jumps = np.sum(s[1:] != s[:-1])

        # Regularization penalty
        reg = self.compute_penalty(mu_df)

        # Total score
        return float(recon + lmbd * jumps + gamma * reg)

    def calibrate(self, max_iter: int = 10) -> Dict[int, Dict[str, object]]:
        """
        Run EM-like alternating updates until convergence (or max_iter) for each
        initialization percent, score each run, pick the best (min score), set
        model state (self.mu, self.s), and compute feature importance.

        Returns
        -------
        dict with keys:
        - 'calibration': {percent -> {... per-run details ...}}
        - 'best_percentage': int
        - 'mu', 'mu_array', 'state_sequence', 'feature_names', 'iterations', 'score'
        - 'feature_importance': pd.Series (index=feature_names)
        """
        if self.data is None or self.initial_state is None:
            raise RuntimeError("Call input_data() and initialize() before calibrate().")

        # ---------- pure inner helpers ----------
        def _update_mu(
            s: np.ndarray,
            data: pd.DataFrame,
            feature_names: list[str],
            K: int,
            gamma: float,
            penalty: PenaltyType,
            jitter: bool = True,
        ) -> pd.DataFrame:
            """
            Update cluster centers (mu) under fixed assignments `s` using
            the Regularized Jump Model formulas (Proposition B.3, Eq. B.1a–B.1d).

            Parameters
            ----------
            s : np.ndarray
                Cluster assignments of length T (values in 1..K).
            data : pd.DataFrame
                Observations Y of shape (T, p).
            feature_names : list[str]
                Column names of features.
            K : int
                Number of clusters/states.
            gamma : float
                Regularization strength (T * gamma appears in the formula).
            penalty : PenaltyType
                One of L0, LASSO, RIDGE, GROUP_LASSO.
            jitter : bool, default True
                If True, add a small random jitter to empty clusters to prevent “dead clusters”.

            Returns
            -------
            mu_df : pd.DataFrame
                Updated cluster centers, shape (K, p), rows 1..K.
            """

            # ---------- Validate ----------
            if len(s) != len(data):
                raise ValueError("Length of state sequence must match number of samples.")
            if not np.issubdtype(np.asarray(s).dtype, np.integer):
                raise TypeError("Cluster assignments s must be integers (1..K).")

            s = np.asarray(s, dtype=int)
            state = pd.Series(s, index=data.index, name="state")

            # ---------- Step 1: Compute unregularized cluster means mu_star ----------
            mu_star = data.groupby(state).mean(numeric_only=True)
            mu_star = mu_star.reindex(range(1, K + 1)).fillna(0.0)
            mu_star = mu_star.reindex(columns=feature_names).fillna(0.0)

            # Cluster sizes |C_k|
            counts = state.value_counts().reindex(range(1, K + 1)).fillna(0).astype(int)
            counts_arr = counts.to_numpy(dtype=float)
            Tn = float(len(data))

            mu_star_arr = mu_star.to_numpy(copy=True)
            mu_out = mu_star_arr.copy()
            Y = data.to_numpy()
            s_idx = s - 1  # zero-based cluster indices

            # ---------- Step 2: Apply penalty-specific proximal updates ----------
            if penalty == PenaltyType.LASSO:
                # Elementwise soft thresholding (Eq. B.1b)
                abs_mu = np.abs(mu_out)
                denom = 2.0 * counts_arr[:, None] * abs_mu
                scale = np.ones_like(mu_out)
                mask = denom > 0
                scale[mask] = np.maximum(0.0, 1.0 - (Tn * gamma) / denom[mask])
                mu_out = scale * mu_out

            elif penalty == PenaltyType.RIDGE:
                # Row-wise shrinkage (Eq. B.1c)
                with np.errstate(divide='ignore', invalid='ignore'):
                    factors = 1.0 / (1.0 + (Tn * gamma) / counts_arr)
                    factors[~np.isfinite(factors)] = 0.0  # handle |C_k|=0
                mu_out = (factors[:, None]) * mu_out

            elif penalty == PenaltyType.GROUP_LASSO:
                # Column-wise group shrinkage (Eq. B.1d)
                col_norm = np.linalg.norm(mu_out, axis=0)  # (p,)
                for j in range(len(feature_names)):
                    nj = col_norm[j]
                    if nj == 0.0:
                        mu_out[:, j] = 0.0
                        continue
                    denom = 2.0 * counts_arr * nj
                    factors = 1.0 / (1.0 + (Tn * gamma) / denom)
                    factors[~np.isfinite(factors)] = 0.0
                    mu_out[:, j] = factors * mu_out[:, j]

            elif penalty == PenaltyType.L0:
                # Hard column selection (Eq. B.1a)
                for j in range(len(feature_names)):
                    yj = Y[:, j]
                    yhat_j = mu_star_arr[s_idx, j]
                    lhs = np.sum(yj * yj)
                    rhs = np.sum((yj - yhat_j) ** 2) + Tn * gamma
                    if lhs <= rhs:
                        mu_out[:, j] = 0.0

            else:
                raise ValueError(f"Unsupported penalty type: {penalty}")

            # ---------- Step 3: Optional jitter for empty clusters ----------
            if jitter:
                empty_mask = counts_arr == 0
                if empty_mask.any():
                    noise = np.random.normal(scale=1e-3, size=mu_out.shape)
                    mu_out[empty_mask, :] += noise[empty_mask, :]

            # ---------- Step 4: Return as DataFrame ----------
            mu_df = pd.DataFrame(mu_out, index=mu_star.index, columns=mu_star.columns)
            return mu_df

        def _update_s(
            mu: pd.DataFrame | np.ndarray,
            data: pd.DataFrame,
            lam: float,
        ) -> np.ndarray:
            Y = data.to_numpy()
            T, _ = Y.shape
            mu = mu.to_numpy() if isinstance(mu, pd.DataFrame) else np.asarray(mu)
            K = mu.shape[0]
            lam = float(lam)

            # Emission costs: E[t, k] = ||y_t - μ_k||^2
            E = np.empty((T, K))
            for k in range(K):
                diff = Y - mu[k]
                E[:, k] = np.sum(diff * diff, axis=1)

            # Backward DP for value function V
            V = np.full((T, K), np.inf)
            V[T - 1, :] = E[T - 1, :]  # terminal (t = T)

            for t in range(T - 2, -1, -1):
                next_row = V[t + 1, :]
                m_all = np.min(next_row + lam)  # min_j { V(t+1,j) + λ }
                stay = next_row                 # no penalty if staying
                V[t, :] = E[t, :] + np.minimum(stay, m_all)

            # Forward backtrace for optimal states
            s_idx = np.empty(T, dtype=int)
            s_idx[0] = int(np.argmin(V[0, :]))
            for t in range(1, T):
                costs = V[t, :] + lam * (np.arange(K) != s_idx[t - 1])
                s_idx[t] = int(np.argmin(costs))
            return s_idx + 1  # labels 1..K

        # ---------- main loop ----------
        results_per_init: Dict[int, Dict[str, object]] = {}
        K = int(self.k)
        feature_names = list(self.feature_names)
        lam = float(self.lmbd)
        data = self.data  # local alias

        best_pct: int | None = None
        best_score: float = np.inf
        best_mu_df: pd.DataFrame | None = None
        best_s: np.ndarray | None = None
        best_iter: int = 0

        for percent, s0 in self.initial_state.items():
            i = 0
            s = s0.copy()
            s_prev = None

            while i < max_iter and not np.array_equal(s, s_prev):
                s_prev = s.copy()
                mu_df = _update_mu(
                    s=s,
                    data=data,
                    feature_names=feature_names,
                    K=K,
                    gamma=self.gamma,
                    penalty=self.penalty,
                )
                s = _update_s(mu=mu_df, data=data, lam=lam)
                i += 1

            score = self._compute_model_score(mu_df, s)

            results_per_init[percent] = {
                "mu": mu_df,
                "mu_array": mu_df.to_numpy(),
                "feature_names": feature_names,
                "state_sequence": s,
                "iterations": i,
                "score": score,
            }

            # track the best (min score); tie-breaker: smaller percent
            if (score < best_score) or (np.isclose(score, best_score) and (best_pct is None or percent < best_pct)):
                best_score = float(score)
                best_pct = int(percent)
                best_mu_df = mu_df
                best_s = s.copy()
                best_iter = i

        # --- set model state to the best run ---
        assert best_mu_df is not None and best_s is not None and best_pct is not None
        self.mu = best_mu_df
        self.s = best_s
        self.regime_series = pd.Series(best_s, index=self.data.index, name="regime")

        # --- feature importance on the best mu ---
        mu_arr = best_mu_df.to_numpy()
        importance_vec = np.sum(mu_arr ** 2, axis=0)  # (p,)
        importance_by_feature = pd.Series(importance_vec, index=feature_names)
        self.feature_score = importance_by_feature

        # --- return combined structure: per-init details + best summary ---
        return {
            "calibration": results_per_init,
            "best_percentage": best_pct,
            "mu": best_mu_df,
            "mu_array": mu_arr,
            "feature_names": feature_names,
            "state_sequence": best_s,
            "iterations": best_iter,
            "score": best_score,
            "feature_importance": importance_by_feature,
        }
    
    def predict(self, oos_data: pd.DataFrame):
        """
        Assign regimes to out-of-sample (OOS) data using calibrated parameters, considering the jump penalty (lmbd),
        and update object parameters.

        Parameters
        ----------
        oos_data : pd.DataFrame
            Out-of-sample data with the same features as the in-sample data.
        """
        if self.mu is None or self.lmbd is None:
            raise RuntimeError("Model must be calibrated on IS data before predicting OOS data.")
        if not isinstance(oos_data, pd.DataFrame):
            raise TypeError("OOS data must be a pandas DataFrame.")
        if not all(col in oos_data.columns for col in self.feature_names):
            raise ValueError("OOS data must contain the same features as the IS data.")

        # Normalize OOS data using the same normalization method as IS data
        oos_data = oos_data[self.feature_names].copy()  # Ensure column order matches
        self.data = normalized(oos_data)  # Update the object's data attribute

        # Assign regimes based on the closest cluster center, considering jump penalty
        Y = self.data.to_numpy()
        mu = self.mu.to_numpy()
        T, K = Y.shape[0], mu.shape[0]

        # Dynamic programming to minimize cost with jump penalty
        cost = np.full((T, K), np.inf)  # Cost matrix
        backtrack = np.zeros((T, K), dtype=int)  # Backtrack matrix

        # Initialize the first time step
        for k in range(K):
            cost[0, k] = np.sum((Y[0] - mu[k]) ** 2)

        # Fill the cost matrix
        for t in range(1, T):
            for k in range(K):
                for j in range(K):
                    jump_cost = self.lmbd if j != k else 0
                    total_cost = cost[t - 1, j] + np.sum((Y[t] - mu[k]) ** 2) + jump_cost
                    if total_cost < cost[t, k]:
                        cost[t, k] = total_cost
                        backtrack[t, k] = j

        # Backtrack to find the optimal state sequence
        self.s = np.zeros(T, dtype=int)
        self.s[-1] = np.argmin(cost[-1]) + 1  # Regimes are 1-indexed
        for t in range(T - 2, -1, -1):
            self.s[t] = backtrack[t + 1, self.s[t + 1] - 1] + 1

        # Update regime series
        self.regime_series = pd.Series(self.s, index=self.data.index, name="regime")

    def visualize(
        self,
        top_n: int = 20,
        figsize_regime: Tuple[float, float] = (14, 3.2),
        figsize_importance: Tuple[float, float] = (9, 6.5),
        savepath_prefix: Optional[str] = None,
        title_suffix: Optional[str] = None,
    ):
        """
        Plot two figures:
        (1) Regime timeline (step line + vivid colored spans)
        (2) Top-N feature importance (sum of mu^2 across states)
        """
        # ---------- Data checks ----------
        if self.data is None:
            raise RuntimeError("No data. Call input_data() first.")

        # Prefer calibrated regime series
        if getattr(self, "regime_series", None) is not None:
            s_arr = np.asarray(self.regime_series.values, dtype=int)
            dates = self.regime_series.index
        else:
            if self.s is None:
                raise RuntimeError("No state sequence. Run calibrate() first.")
            s_arr = np.asarray(self.s, dtype=int)
            dates = self.data.index

        if self.feature_score is None or not isinstance(self.feature_score, pd.Series):
            raise RuntimeError("feature_score not found. Run calibrate() first.")

        if not isinstance(dates, (pd.DatetimeIndex, pd.PeriodIndex)):
            raise TypeError("Index must be datetime-like for timeline plotting.")

        K = int(self.k)
        s0 = s_arr  # regime labels 1..K (keep consistent)

        # ---------- (1) Regime timeline ----------
        fig1, ax1 = plt.subplots(figsize=figsize_regime)

        # Use vivid tab10 palette
        cmap = plt.cm.get_cmap("tab10", K)
        regime_colors = {k + 1: cmap(k % 10) for k in range(K)}  # regime 1..K

        # Draw colored spans (background)
        start_idx = 0
        for t in range(1, len(s0) + 1):
            if t == len(s0) or s0[t] != s0[t - 1]:
                k_id = int(s0[t - 1])
                ax1.axvspan(
                    dates[start_idx],
                    dates[t - 1],
                    facecolor=regime_colors[k_id],
                    alpha=0.25,
                    edgecolor="none",
                )
                start_idx = t

        # Draw single black step line (regime transitions)
        ax1.step(dates, s0, where="post", linewidth=2.0, color="black")

        # Build legend manually to match regime numbers 1..K
        legend_patches = [
            Patch(facecolor=regime_colors[k_id], alpha=0.6, label=f"Regime {k_id}")
            for k_id in range(1, K + 1)
        ]

        # Configure plot
        ttl = "Identified Market Regimes Over Time"
        if title_suffix:
            ttl += f" {title_suffix}"
        ax1.set_title(ttl, fontsize=13, weight="bold")
        ax1.set_ylabel("Regime ID")
        ax1.set_xlabel("Date")
        ax1.grid(True, which="both", axis="both", alpha=0.25)
        ax1.set_yticks(list(range(1, K + 1)))
        ax1.set_ylim(0.5, K + 0.5)

        # Legend with consistent numbering
        ax1.legend(handles=legend_patches, loc="upper left", frameon=True, title="Regime")

        fig1.tight_layout()

        # ---------- (2) Feature importance ----------
        imp = self.feature_score.sort_values(ascending=False).head(top_n)

        fig2, ax2 = plt.subplots(figsize=figsize_importance)
        colors = [plt.cm.tab10(i % 10) for i in range(len(imp))]
        ax2.barh(imp.index[::-1], imp.values[::-1], color=colors[::-1])
        shown_n = min(top_n, len(self.feature_score))
        ax2.set_title(
            f"Top {shown_n} Feature Importance (sum of μ²)",
            fontsize=13,
            weight="bold",
        )
        ax2.set_xlabel("Importance")
        ax2.set_ylabel("")
        ax2.grid(True, axis="x", alpha=0.25)
        ax2.margins(y=0.01)
        fig2.tight_layout()

        # ---------- optional save ----------
        if savepath_prefix:
            fig1.savefig(f"{savepath_prefix}_regimes.png", dpi=160, bbox_inches="tight")
            fig2.savefig(f"{savepath_prefix}_importance.png", dpi=160, bbox_inches="tight")

        return fig1, fig2

    def optuna_tune_with_bootstrap(
        self,
        lambda_range: tuple = (1e-1, 1e2),    # log scale
        gamma_range: tuple = (1e-1, 1e2),     # log scale
        penalty_candidates: List[str] = ("lasso", "l0", "ridge", "group_lasso"),
        n_boot: int = 10,
        max_iter: int = 20,
        n_trials: int = 60,
        random_state: int = 42,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        storage: Optional[str] = None,
        study_name: Optional[str] = None,
        n_startup_trials: int = 10,
    ) -> Dict:
        """
        Tune (lambda, gamma, penalty) with Optuna by minimizing bootstrap instability
        across ALL initialization percents produced by `initialize`.

        For each trial (i.e., for a given hyperparameter tuple), we:
        1) Evaluate every initialization `percent`:
            - For each bootstrap resample, fit a model starting from that `percent`.
            - Collect the resulting state sequences and compute a stability score.
        2) Pick the `percent` with the lowest stability score for this trial.
            - If there is a tie, break it using the original-data cost
            (reconstruction error + lambda * #jumps + gamma * penalty).

        Returns
        -------
        dict
            {
            "best_params": {"lmbd", "gamma", "penalty"},
            "best_score":  float stability score in [0, 1],
            "best_percent": the chosen initialization key used by the best trial,
            "study": Optuna study object
            }
        """
        data = self.data
        if data is None or data.empty:
            raise ValueError("data must be a non-empty DataFrame.")
        if n_boot < 2:
            raise ValueError("n_boot must be >= 2.")

        K = self.k

        # Build the list of initialization percents from the ORIGINAL data.
        self.input_data(data)
        self.initialize(k=K, random_state=random_state)
        percent_list = list(self.initial_state.keys())

        # Pre-generate stationary bootstrap resamples (fixed seeds for comparability).
        boot_list = HelperFunction.StationaryBootstrap(
            data=data, random_state=random_state, n_boot=n_boot
        )

        if sampler is None:
            sampler = optuna.samplers.TPESampler(
                seed=random_state, n_startup_trials=n_startup_trials
            )
        if pruner is None:
            pruner = optuna.pruners.MedianPruner(n_startup_trials=5)

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            study_name=study_name,
            load_if_exists=bool(storage and study_name),
        )

        def _pick_nearest_key(keys, target):
            """Pick an exact or nearest percent key from `keys` to mirror `target`."""
            if target in keys:
                return target
            try:
                t = float(target)
                return min(
                    ((abs(float(k) - t), k) for k in keys),
                    key=lambda x: x[0]
                )[1]
            except Exception:
                # Fallback: just take an arbitrary key deterministically
                return next(iter(keys))

        def _original_cost(lmbd: float, gamma: float, penalty: str, percent_key) -> float:
            """Compute the original-data cost for tie-breaking."""
            m0 = RegularizedJumpModel(k=K, lmbd=lmbd, gamma=gamma, penalty=penalty)
            m0.input_data(data)
            m0.initialize(k=K, random_state=random_state)

            chosen_key = _pick_nearest_key(m0.initial_state.keys(), percent_key)
            m0.initial_state = {chosen_key: m0.initial_state[chosen_key]}

            res0 = m0.calibrate(max_iter=max_iter)
            out = res0[chosen_key]
            mu0_df = out["mu"]                              # DataFrame
            s0 = out["state_sequence"]
            mu0 = mu0_df.to_numpy()                         # ndarray for math below

            Y = data.to_numpy()
            recon = float(np.sum((Y - mu0[s0 - 1]) ** 2))
            jumps = int(np.sum(s0[1:] != s0[:-1]))
            reg = float(m0.compute_penalty(mu0_df))         # accepts DF or ndarray
            return recon + lmbd * jumps + gamma * reg

        def objective(trial: optuna.trial.Trial) -> float:
            # Hyperparameters to tune
            lmbd = trial.suggest_float("lambda", lambda_range[0], lambda_range[1], log=True)
            gamma = trial.suggest_float("gamma", gamma_range[0], gamma_range[1], log=True)
            penalty = trial.suggest_categorical("penalty", list(penalty_candidates))

            percent2score: Dict[object, float] = {}
            percent2orig: Dict[object, float] = {}

            # Evaluate each initialization percent under bootstrap.
            for percent_key in percent_list:
                seqs: List[np.ndarray] = []

                for b in range(n_boot):
                    mdl = RegularizedJumpModel(k=K, lmbd=lmbd, gamma=gamma, penalty=penalty)
                    mdl.input_data(boot_list[b])
                    mdl.initialize(k=K, random_state=(random_state + b))

                    chosen_key = _pick_nearest_key(mdl.initial_state.keys(), percent_key)
                    # Force using only this initialization on this bootstrap sample
                    mdl.initial_state = {chosen_key: mdl.initial_state[chosen_key]}

                    res = mdl.calibrate(max_iter=max_iter)
                    seqs.append(res[chosen_key]["state_sequence"])

                # Lower is better
                score = HelperFunction.stability_score(seqs, K=K)
                if not np.isfinite(score):
                    score = 1.0  # Safe worst-case fallback

                percent2score[percent_key] = float(score)
                percent2orig[percent_key] = _original_cost(lmbd, gamma, penalty, percent_key)

            # For this trial, pick the percent with the best (lowest) stability;
            # break ties using the original-data cost.
            best_percent_this_trial = min(
                percent_list, key=lambda k: (percent2score[k], percent2orig[k])
            )

            trial.set_user_attr("best_percent", best_percent_this_trial)
            trial.set_user_attr("orig_cost", percent2orig[best_percent_this_trial])

            # The trial's target value is the best stability score across percents.
            return percent2score[best_percent_this_trial]

        # Run the optimization
        study.optimize(objective, n_trials=n_trials)

        # Pick the best completed trial; break ties with original-data cost.
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        best_trial = min(
            completed, key=lambda t: (t.value, t.user_attrs.get("orig_cost", np.inf))
        )

        best_params = {
            "lmbd": float(best_trial.params["lambda"]),
            "gamma": float(best_trial.params["gamma"]),
            "penalty": str(best_trial.params["penalty"]),
        }
        best_percent = best_trial.user_attrs.get("best_percent")

        return {
            "best_params": best_params,
            "best_score": float(best_trial.value),
            "best_percent": best_percent,
            "study": study,
        }

    def summarize_by_regime_from_daily_returns(
        self,
        daily_returns: Union[pd.Series, pd.DataFrame, Dict[str, Union[pd.Series, pd.DataFrame]]],
        pct_view: bool = False,
        default_asset_name: str = "asset",
        *,
        do_anova: bool = True,
        do_kruskal: bool = True,
        do_pairwise: bool = False,          # set True to run pairwise tests (if scipy available)
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Summarize per-regime stats for one or many assets, and run statistical tests
        to check whether returns differ across regimes.

        Returns
        -------
        summary_all : DataFrame
            Numeric table, MultiIndex (asset, regime).
        summary_all_pct : DataFrame
            Percentage string view.
        tests : dict[str, DataFrame]
            {
              'anova':    DataFrame by asset (F, pvalue, df_between, df_within, n_groups, n_total),
              'kruskal':  DataFrame by asset (H, pvalue, k),
              'pairwise': DataFrame by (asset, reg_i, reg_j) with Welch t-tests + Holm-adjusted p
            }
        """
        # ---------- Validate model state ----------
        if self.data is None:
            raise RuntimeError("Model has no data. Call input_data() first.")

        # Prefer calibrated regime_series; otherwise fall back to self.s
        if getattr(self, "regime_series", None) is not None:
            regime = self.regime_series.copy()
        else:
            if self.s is None:
                raise RuntimeError("No regime sequence. Run calibrate() first.")
            regime = pd.Series(self.s, index=self.data.index, name="regime")

        if not isinstance(regime.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            raise TypeError("Regime index must be datetime-like.")

        # ---------- Utilities ----------
        def _ensure_series(x: Union[pd.Series, pd.DataFrame]) -> pd.Series:
            if isinstance(x, pd.DataFrame):
                if x.shape[1] != 1:
                    raise ValueError("Input DataFrame must have exactly one column.")
                x = x.squeeze("columns")
            return x

        def _to_month_period_index(s: pd.Series) -> pd.Series:
            if isinstance(s.index, pd.PeriodIndex):
                return s.set_axis(s.index.asfreq("M"))
            if not isinstance(s.index, pd.DatetimeIndex):
                raise TypeError("Series must have DatetimeIndex or PeriodIndex.")
            return s.set_axis(s.index.to_period("M"))

        # Prepare monthly regime series
        regime_series = _ensure_series(regime).rename("regime")
        regime_series = _to_month_period_index(regime_series)
        try:
            regime_series = regime_series.astype("Int64")
        except Exception:
            pass

        # Single asset -> convert to dict for unified processing
        if not isinstance(daily_returns, dict):
            daily_returns = {default_asset_name: daily_returns}

        summary_frames: Dict[str, pd.DataFrame] = {}
        summary_pct_frames: Dict[str, pd.DataFrame] = {}

        # For tests we need to keep per-asset per-regime monthly returns vectors
        per_asset_groups: Dict[str, Dict[int, np.ndarray]] = {}

        for asset_name, dr in daily_returns.items():
            r = _ensure_series(dr).dropna().copy()
            if not isinstance(r.index, pd.DatetimeIndex):
                raise TypeError(f"daily_returns for '{asset_name}' must have a DatetimeIndex.")
            r.name = r.name or f"{asset_name}_daily_return"

            # Monthly aggregation
            m_ret = (1.0 + r).resample("M").apply(np.prod) - 1.0
            m_ret.name = "monthly_return"
            intra_month_vol = r.resample("M").std()
            intra_month_vol.name = "intra_month_vol"

            monthly_df = pd.concat([m_ret, intra_month_vol], axis=1).dropna()
            monthly_df.index = monthly_df.index.to_period("M")

            # Align with regimes
            aligned = monthly_df.join(regime_series, how="inner")
            if aligned.empty:
                raise ValueError(
                    f"No overlapping months between '{asset_name}' returns and model regime."
                )
            if aligned["regime"].isna().any():
                aligned = aligned.dropna(subset=["regime"])
            aligned["regime"] = aligned["regime"].astype(int)

            # Summary table
            summary = (
                aligned.groupby("regime")
                       .agg(mean_monthly_return=("monthly_return", "mean"),
                            mean_intra_month_vol=("intra_month_vol", "mean"),
                            count_months=("monthly_return", "size"))
                       .sort_index()
            )

            # Percentage view
            summary_pct = summary.copy()
            summary_pct["mean_monthly_return"] = (summary_pct["mean_monthly_return"] * 100).map("{:.2f}%".format)
            summary_pct["mean_intra_month_vol"] = (summary_pct["mean_intra_month_vol"] * 100).map("{:.2f}%".format)

            summary_frames[asset_name] = summary
            summary_pct_frames[asset_name] = summary_pct

            # Collect vectors by regime for tests
            groups = {
                reg: grp["monthly_return"].to_numpy(copy=True)
                for reg, grp in aligned.groupby("regime")
            }
            per_asset_groups[asset_name] = groups

        # Stack summaries
        summary_all = pd.concat(summary_frames, names=["asset", "regime"]).sort_index()
        summary_all_pct = pd.concat(summary_pct_frames, names=["asset", "regime"]).sort_index()

        # ---------- Statistical tests ----------
        tests: Dict[str, pd.DataFrame] = {}

        # Helpers for tests (scipy optional)
        try:
            from scipy.stats import f as f_dist
            from scipy.stats import kruskal as kruskal_test
            from scipy.stats import ttest_ind
            SCIPY_OK = True
        except Exception:
            SCIPY_OK = False

        def _anova_numpy(groups_dict: Dict[int, np.ndarray]):
            """Return F, pvalue (if scipy available), df_between, df_within, n_groups, n_total."""
            # Drop empty groups
            vals = [g[~np.isnan(g)] for g in groups_dict.values() if g is not None and g.size > 0]
            ns = [len(v) for v in vals]
            k = len(vals)
            n_total = sum(ns)
            if k < 2 or n_total <= k:
                return np.nan, np.nan, 0, 0, k, n_total

            all_concat = np.concatenate(vals)
            grand_mean = all_concat.mean()

            # Between-group SS
            ss_between = sum(n * (v.mean() - grand_mean) ** 2 for v, n in zip(vals, ns))
            # Within-group SS
            ss_within = sum(((v - v.mean()) ** 2).sum() for v in vals)

            df_between = k - 1
            df_within = n_total - k
            if df_within <= 0:
                return np.nan, np.nan, df_between, df_within, k, n_total

            ms_between = ss_between / df_between
            ms_within = ss_within / df_within
            if ms_within == 0:
                F = np.inf
            else:
                F = ms_between / ms_within

            if SCIPY_OK and np.isfinite(F):
                p = 1.0 - f_dist.cdf(F, df_between, df_within)
            else:
                p = np.nan
            return float(F), float(p), int(df_between), int(df_within), int(k), int(n_total)

        def _kruskal(groups_dict: Dict[int, np.ndarray]):
            """Return H, pvalue, k (groups)."""
            vals = [g[~np.isnan(g)] for g in groups_dict.values() if g is not None and g.size > 0]
            k = len(vals)
            if k < 2:
                return np.nan, np.nan, k
            if SCIPY_OK:
                H, p = kruskal_test(*vals)
                return float(H), float(p), int(k)
            else:
                # No scipy: cannot compute exact p; return NaN
                return np.nan, np.nan, k

        def _holm_bonferroni(pvals: np.ndarray) -> np.ndarray:
            """Holm step-down adjustment (returns adjusted p-values)."""
            m = len(pvals)
            order = np.argsort(pvals)
            adj = np.empty(m, dtype=float)
            prev = 0.0
            for i, idx in enumerate(order):
                rank = i + 1
                adj_p = (m - i) * pvals[idx]
                adj[idx] = max(adj_p, prev)  # ensure monotonicity
                prev = adj[idx]
            return np.minimum(adj, 1.0)

        pairwise_rows = []

        # Run tests per asset
        anova_rows = []
        kruskal_rows = []

        for asset, groups in per_asset_groups.items():
            if do_anova:
                F, p, dfb, dfw, k_groups, n_tot = _anova_numpy(groups)
                anova_rows.append({
                    "asset": asset,
                    "F": F,
                    "pvalue": p,
                    "df_between": dfb,
                    "df_within": dfw,
                    "n_groups": k_groups,
                    "n_total": n_tot,
                })

            if do_kruskal:
                H, p, k_groups = _kruskal(groups)
                kruskal_rows.append({
                    "asset": asset,
                    "H": H,
                    "pvalue": p,
                    "k": k_groups,
                })

            if do_pairwise and SCIPY_OK:
                # all pair (i<j) Welch t-tests with Holm correction
                regs = sorted(groups.keys())
                samples = {r: groups[r][~np.isnan(groups[r])] for r in regs}
                pairs = [(i, j) for idx, i in enumerate(regs) for j in regs[idx+1:]]
                t_list, p_list = [], []
                keep_pairs = []

                for i, j in pairs:
                    xi = samples[i]
                    xj = samples[j]
                    if len(xi) >= 2 and len(xj) >= 2:
                        t_stat, p_val = ttest_ind(xi, xj, equal_var=False, nan_policy="omit")
                        t_list.append(float(t_stat))
                        p_list.append(float(p_val))
                        keep_pairs.append((i, j))

                if p_list:
                    p_adj = _holm_bonferroni(np.array(p_list))
                    for (i, j), t_stat, p_raw, p_corr in zip(keep_pairs, t_list, p_list, p_adj):
                        pairwise_rows.append({
                            "asset": asset,
                            "reg_i": i,
                            "reg_j": j,
                            "t_stat": t_stat,
                            "pvalue_raw": p_raw,
                            "pvalue_holm": float(p_corr),
                            "method": "Welch t-test (Holm adjusted)"
                        })

        if do_anova:
            tests["anova"] = pd.DataFrame(anova_rows).set_index("asset") if anova_rows else pd.DataFrame()
        if do_kruskal:
            tests["kruskal"] = pd.DataFrame(kruskal_rows).set_index("asset") if kruskal_rows else pd.DataFrame()
        if do_pairwise and pairwise_rows:
            tests["pairwise"] = (
                pd.DataFrame(pairwise_rows)
                  .set_index(["asset", "reg_i", "reg_j"])
                  .sort_index()
            )

        return summary_all, summary_all_pct, tests

def backtest_centers_over_time(
    data: pd.DataFrame,
    k: int,
    first_cutoff: str | pd.Timestamp,
    step: str | pd.DateOffset = "M",
    *,
    end_date: str | pd.Timestamp | None = None,
    lmbd: float = 1.0,
    gamma: float = 1.0,
    penalty: str | PenaltyType = "lasso",
    max_iter: int = 10,
    init_random_state: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Rolling/expanding backtest for RegularizedJumpModel centers.

    At t = first_cutoff, fit the model on data[:t] and store centers μ(t).
    Then advance the cutoff by 'step' and repeat until end_date (or data end).
    Centers are aligned across time by greedy matching to minimize L2 drift.

    Parameters
    ----------
    data : (T, p) DataFrame
        Full normalized (or raw-to-be-normalized inside model) feature panel with
        DateTimeIndex (daily/monthly are both fine).
    k : int
        Number of regimes/states.
    first_cutoff : str | Timestamp
        First training cutoff (inclusive), e.g. "2020-01-31" or "2020-01".
    step : str | DateOffset, default "M"
        Step to advance the cutoff, e.g. "M" (1 month), "QS" (quarter start),
        or pd.DateOffset(months=3), etc.
    end_date : str | Timestamp | None
        Last cutoff date to consider. If None, uses the last index in `data`.
    lmbd, gamma, penalty, max_iter, init_random_state
        Model hyperparameters and calibration settings.
    verbose : bool
        If True, prints simple progress.

    Returns
    -------
    dict with keys:
        - 'centers_aligned': DataFrame
              MultiIndex rows = (cutoff, regime_id 1..K), columns = feature names.
              Centers are re-ordered each step to best match the previous step.
        - 'centers_raw': dict[cutoff -> DataFrame]
              Unaligned μ (as returned by each fit).
        - 'shifts': DataFrame
              Per-cutoff shift diagnostics (Frobenius & per-regime L2 after matching).
        - 'cutoffs': list[pd.Timestamp]
              The training cutoffs used.
    """
    if not isinstance(data.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise TypeError("`data` must have a DatetimeIndex or PeriodIndex.")
    if isinstance(data.index, pd.PeriodIndex):
        # Convert PeriodIndex to a Timestamp index at period end for slicing
        data = data.copy()
        data.index = data.index.to_timestamp(how="end")

    # Resolve dates
    first_cutoff = pd.Timestamp(first_cutoff)
    last_possible = pd.Timestamp(end_date) if end_date is not None else data.index.max()

    # Build the list of cutoffs: first_cutoff, first_cutoff+step, ...
    cutoffs: list[pd.Timestamp] = []
    t = first_cutoff
    step_offset = pd.tseries.frequencies.to_offset(step) if isinstance(step, str) else step
    while t <= last_possible:
        # Only keep cutoffs that exist after the first available date
        if t >= data.index.min():
            cutoffs.append(t)
        t = t + step_offset

    if len(cutoffs) == 0:
        raise ValueError("No valid cutoffs between first_cutoff and end_date for the given data.")

    # Storage
    centers_raw: "OrderedDict[pd.Timestamp, pd.DataFrame]" = OrderedDict()
    centers_aligned_frames: list[pd.DataFrame] = []
    shifts_rows: list[dict] = []

    prev_mu: Optional[pd.DataFrame] = None  # previous (aligned) centers
    feature_names: Optional[list[str]] = None

    # --- helper: greedy matching (K small) ---
    def _greedy_match(prev: np.ndarray, curr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match rows of 'curr' to rows of 'prev' to minimize pairwise L2 distances.
        Returns:
            perm : ndarray of length K, where prev[i] ~ curr[perm[i]]
            dists: per-row L2 distances after assignment
        """
        K = prev.shape[0]
        D = np.empty((K, K), dtype=float)
        # squared L2 distance matrix between centers
        for i in range(K):
            diff = curr - prev[i]
            D[i, :] = np.sum(diff * diff, axis=1)

        used_rows = set()
        used_cols = set()
        perm = np.empty(K, dtype=int)
        dists = np.empty(K, dtype=float)

        for _ in range(K):
            # pick the minimal remaining entry
            mask = np.full_like(D, True, dtype=bool)
            if used_rows:
                mask[list(used_rows), :] = False
            if used_cols:
                mask[:, list(used_cols)] = False
            # find global minimum among allowed pairs
            i_min, j_min = np.unravel_index(np.argmin(np.where(mask, D, np.inf)), D.shape)
            used_rows.add(i_min)
            used_cols.add(j_min)
            perm[i_min] = j_min
            dists[i_min] = np.sqrt(D[i_min, j_min])
        return perm, dists

    # --- rolling training ---
    for cutoff in cutoffs:
        if verbose:
            print(f"[{cutoff.date()}] fitting on data up to cutoff ...")

        sub = data.loc[:cutoff]
        if sub.empty:
            continue

        # Fit a fresh model on the expanding window
        mdl = RegularizedJumpModel(k=k, lmbd=lmbd, gamma=gamma, penalty=penalty)
        mdl.input_data(sub)                # includes normalization
        mdl.initialize(random_state=init_random_state)
        res = mdl.calibrate(max_iter=max_iter)

        mu_df: pd.DataFrame = res["mu"]    # (K x p), rows indexed 1..K, columns are features
        centers_raw[cutoff] = mu_df

        # First run: set baseline alignment
        if prev_mu is None:
            aligned = mu_df.copy()
            aligned.index = pd.Index(range(1, k + 1), name="regime")
            prev_mu = aligned.copy()
            feature_names = list(aligned.columns)

            # no shift for the very first step (fill with NaNs or zeros)
            shifts_rows.append({
                "cutoff": cutoff,
                "frobenius_shift": np.nan,
                **{f"regime_{i}_L2": np.nan for i in range(1, k + 1)}
            })
        else:
            # Align current centers to previous centers
            prev_arr = prev_mu.to_numpy()
            curr_arr = mu_df.to_numpy()

            perm, per_reg_L2 = _greedy_match(prev_arr, curr_arr)
            # Reorder current μ rows according to matching with prev
            aligned_arr = curr_arr[perm, :]
            aligned = pd.DataFrame(aligned_arr, index=pd.Index(range(1, k + 1), name="regime"),
                                   columns=prev_mu.columns)

            # Record shift metrics
            fro_shift = float(np.linalg.norm(aligned_arr - prev_arr))
            row = {"cutoff": cutoff, "frobenius_shift": fro_shift}
            row.update({f"regime_{i}_L2": float(per_reg_L2[i - 1]) for i in range(1, k + 1)})
            shifts_rows.append(row)

            # Update previous baseline
            prev_mu = aligned.copy()

        # Stack (with a top level index = cutoff)
        aligned.index.name = "regime"
        aligned["__cutoff__"] = cutoff
        centers_aligned_frames.append(aligned.set_index("__cutoff__", append=True))

    # Concatenate all aligned centers into one table
    centers_aligned = pd.concat(centers_aligned_frames).reorder_levels(["__cutoff__", "regime"]).sort_index()
    centers_aligned.index.names = ["cutoff", "regime"]

    shifts = pd.DataFrame(shifts_rows).set_index("cutoff").sort_index()

    return {
        "centers_aligned": centers_aligned,   # MultiIndex (cutoff, regime) x features
        "centers_raw": centers_raw,           # dict[cutoff -> DataFrame]
        "shifts": shifts,                     # per-cutoff drift diagnostics
        "cutoffs": cutoffs,
    }

def compute_frobenius_drift(
    centers_aligned: pd.DataFrame | None = None,
    backtest_out: Dict[str, Any] | None = None,
) -> pd.Series:
    """
    Compute Frobenius drift between consecutive training cutoffs.

    Parameters
    ----------
    centers_aligned : DataFrame, optional
        MultiIndex rows = (cutoff, regime), columns = features.
        If None, this function will try to read from backtest_out["centers_aligned"].
    backtest_out : dict, optional
        The output dict returned by `backtest_centers_over_time(...)`.

    Returns
    -------
    drift : pd.Series
        Index = cutoff (Timestamp), values = ||μ_t - μ_{t-1}||_F.
        The first cutoff has NaN because there is no previous step.
    """
    if centers_aligned is None:
        if backtest_out is None or "centers_aligned" not in backtest_out:
            raise ValueError("Provide `centers_aligned` or `backtest_out` with 'centers_aligned'.")
        centers_aligned = backtest_out["centers_aligned"]

    if not isinstance(centers_aligned.index, pd.MultiIndex) or \
       list(centers_aligned.index.names) != ["cutoff", "regime"]:
        raise ValueError("`centers_aligned` must have MultiIndex rows ['cutoff','regime'].")

    # Collect unique cutoffs in chronological order
    cutoffs = pd.Index(sorted(centers_aligned.index.get_level_values("cutoff").unique()))
    drifts = []

    prev = None
    for t in cutoffs:
        # (K x p) matrix for this cutoff ordered by regime 1..K
        mu_t = centers_aligned.loc[t].sort_index().to_numpy()  # ensure regime order
        if prev is None:
            drifts.append(np.nan)  # no previous step
        else:
            diff = mu_t - prev
            # Frobenius norm = sqrt(sum of squares over all entries)
            drifts.append(float(np.linalg.norm(diff)))
        prev = mu_t

    drift = pd.Series(drifts, index=cutoffs, name="frobenius_drift")
    return drift


def plot_frobenius_drift(
    drift: pd.Series,
    *,
    ax: Optional[plt.Axes] = None,
    title: str = "Frobenius Drift of Cluster Centers",
    rolling: Optional[int] = 3,
    mark_spikes: bool = True,
    zscore_threshold: float = 2.0,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot Frobenius drift as a time series with optional rolling mean and spike markers.

    Parameters
    ----------
    drift : pd.Series
        Output from `compute_frobenius_drift`. Index must be datetime-like.
    ax : plt.Axes, optional
        If provided, draw on this axes; otherwise create a new figure.
    title : str
        Plot title.
    rolling : int or None
        Window size for an optional rolling mean overlay (in number of points).
        Set to None to disable.
    mark_spikes : bool
        If True, mark large-move spikes where z-score > `zscore_threshold`.
    zscore_threshold : float
        Threshold for spike detection on the standardized series.

    Returns
    -------
    (fig, ax)
    """
    if not isinstance(drift.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        # try to convert if index looks like timestamps
        try:
            drift.index = pd.to_datetime(drift.index)
        except Exception as e:
            raise TypeError("`drift` index must be datetime-like.") from e

    if ax is None:
        fig, ax = plt.subplots(figsize=(10.5, 3.4))
    else:
        fig = ax.figure

    ax.plot(drift.index, drift.values, label="Frobenius drift", linewidth=2)

    # Optional rolling mean for smoother trend
    if rolling is not None and rolling > 1:
        roll = drift.rolling(rolling, min_periods=max(2, rolling // 2)).mean()
        ax.plot(roll.index, roll.values, label=f"Rolling mean ({rolling})", linewidth=2)

    # Optional spike markers using z-score
    if mark_spikes:
        x = drift.copy()
        mu = x.mean(skipna=True)
        sd = x.std(skipna=True)
        if sd and sd > 0:
            z = (x - mu) / sd
            spikes = z > zscore_threshold
            ax.scatter(
                x.index[spikes],
                x.values[spikes],
                marker="o",
                s=36,
                alpha=0.9,
                label=f"Spikes (z>{zscore_threshold:g})",
            )

    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("Cutoff date")
    ax.set_ylabel("||μₜ − μₜ₋₁||₍F₎")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig, ax
