import sys
from typing import List, Dict, Any
from abc import abstractmethod

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

PROJECT_DIR = Path(__file__).resolve()
sys.path.insert(0, str(PROJECT_DIR))

class Initialization:
    @staticmethod
    def regularized(
        data: pd.DataFrame,
        k: int = 2,
        random_state: int = 42
    ) -> Dict[str, object]:
        """
        Initialise state sequences for the Regularised Jump Model (Algorithm 7). 
        Returns:
            - d: feature importance scores (|| mu_{:, j} ||_2)
            - state_sequences: dict mapping {percent -> state_sequence (ndarray)}
        """
        X = data.to_numpy(copy=False)  # shape (T, p)
        T, p = X.shape

        # ----- Step 1: Standard K-means on all variables -----
        km_full = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        km_full.fit(X)  # fit before accessing cluster_centers_
        mu = km_full.cluster_centers_.astype(float)  # (k, p)

        # ----- Step 2: d_j = || mu_{:, j} ||_2 for j=1..p -----
        d = np.linalg.norm(mu, axis=0)        # shape (p,)
        sorted_idx = np.argsort(-d)           # descending indices

        # ----- Step 3: Run K-means on subsets at given percentages -----
        percents = [1, 2, 5, 10, 25, 50, 100]
        state_sequences: Dict[int, np.ndarray] = {}

        for perc in percents:
            m = max(1, int(np.ceil(p * perc / 100.0)))  # number of top features to use
            cols = sorted_idx[:m]                       # top m features by index
            X_sub = X[:, cols]

            km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            labels = km.fit_predict(X_sub)  # length T

            state_sequences[perc] = labels

        return {
            "d": d,
            "state_sequences": state_sequences
        }