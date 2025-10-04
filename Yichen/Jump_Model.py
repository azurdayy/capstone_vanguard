import sys
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional
import optuna

from pathlib import Path
import pandas as pd
import numpy as np

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
        self.state_seq = None
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
        self.data = data.copy()
        if not np.issubdtype(self.data.dtypes.values[0], np.number):
            raise TypeError("Data must be numeric")
        if (self.data.isna().sum().sum() > 0):
            raise ValueError("Data contains NaN values")
        if (self.data.std(ddof=0) == 0).any():
            raise ValueError("Data contains constant columns")
        self.data = normalized(data)
        self.T, self.p = self.data.shape
        self.feature_names = list(self.data.columns)

    def initialize(self, k: int = 2, random_state: int = 42):
        result = Initialization.regularized(self.data, k, random_state)
        self.feature_score = result["d"]
        # shift KMeans labels (0..k-1) to 1..k for internal consistency
        self.initial_state = {pct: (lbls.astype(int) + 1) 
                            for pct, lbls in result["state_sequences"].items()}

    def compute_penalty(self, mu: pd.DataFrame | np.ndarray) -> float:
        mu_arr = mu.to_numpy() if isinstance(mu, pd.DataFrame) else np.asarray(mu)
        return self.penalty.compute(mu_arr)
    
    def _update_mu(self, s: np.ndarray) -> np.ndarray:
        """
        Compute state-wise feature means:
            mu_k = sum_{t: s_t = k} y_t / count_{t: s_t = k}

        Args:
            data: (T, p) DataFrame, each row is an observation at time t,
                each column is a feature.
            s:    (T,) np.ndarray, state labels, values in 1..K.

        Returns:
            mus:  (K, p) np.ndarray, each row corresponds to the mean vector of features
                for state k (in order 1..K).
        """
        if len(s) != len(self.data):
            raise ValueError(f"len(s)={len(s)} must equal len(data)={len(self.data)}")
        if not np.issubdtype(np.asarray(s).dtype, np.integer):
            raise TypeError("s must be integer labels (1..K)")

        s = np.asarray(s).astype(int)
        K = self.k
        state = pd.Series(s, index=self.data.index, name="state")

        mu_df = self.data.groupby(state).mean(numeric_only=True)
        mu_df = mu_df.reindex(range(1, K + 1))  # ensure 1..K order; missing => NaN
        # enforce original column order
        mu_df = mu_df.reindex(columns=self.feature_names)

        return mu_df  # (K x p) DataFrame
    
    def _update_s(self, mu: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Update the state sequence s^(j) using dynamic programming (Eqs. 15.1a–15.2b).

        Parameters
        ----------
        mu : np.ndarray, shape (K, p)
            Row k is the mean vector μ_k of state k (1..K).

        Returns
        -------
        np.ndarray, shape (T,)
            Optimal state sequence with values in 1..K.
        """
        # Data and sizes
        Y = self.data.to_numpy()
        T, _ = Y.shape
        mu = mu.to_numpy() if isinstance(mu, pd.DataFrame) else np.asarray(mu)
        K = mu.shape[0]
        lam = float(self.lmbd)

        # Pre-compute emission costs: E[t, s] = ||y_t - μ_s||^2
        E = np.empty((T, K))
        for s in range(K):
            diff = Y - mu[s]  # broadcast over time
            E[:, s] = np.sum(diff * diff, axis=1)

        # DP table: V[t, s] corresponds to math V(t+1, s) due to 0-based indexing
        V = np.full((T, K), np.inf)

        # (15.1a) terminal condition at t = T  -> code index T-1
        V[T - 1, :] = E[T - 1, :]

        # (15.1b) backward recursion for t = T-1, ..., 1  -> code: T-2 down to 0
        for t in range(T - 2, -1, -1):
            next_row = V[t + 1, :]
            m_all = np.min(next_row + lam)  # min_j { V(t+1, j) + λ }
            stay = next_row                 # no penalty if staying in the same state
            V[t, :] = E[t, :] + np.minimum(stay, m_all)

        # (15.2a) choose s1  -> code index 0
        s_idx = np.empty(T, dtype=int)
        s_idx[0] = int(np.argmin(V[0, :]))

        # (15.2b) forward selection using the previously chosen state
        for t in range(1, T):
            costs = V[t, :] + lam * (np.arange(K) != s_idx[t - 1])
            s_idx[t] = int(np.argmin(costs))

        # Convert to labels in 1..K
        return s_idx + 1

    def calibrate(self, max_iter: int = 10) -> Dict[int, Dict[str, object]]:
        if self.data is None or self.initial_state is None:
            raise RuntimeError("Call input_data() and initialize() before calibrate().")

        out: Dict[int, Dict[str, object]] = {}
        for percent, s0 in self.initial_state.items():
            i = 0
            s = s0.copy()
            s_prev = None

            while i < max_iter and not np.array_equal(s, s_prev):
                s_prev = s.copy()
                mu_df = self._update_mu(s)   # DataFrame with named columns
                s = self._update_s(mu_df)
                i += 1

            out[percent] = {
                "mu": mu_df,                         # DataFrame (K x p)
                "mu_array": mu_df.to_numpy(),        # ndarray convenience
                "feature_names": self.feature_names, # list[str]
                "state_sequence": s,
                "iterations": i,
            }
        return out

    def predict(self, new_data: pd.DataFrame) -> np.ndarray:
        pass

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