import sys
from typing import List
from abc import abstractmethod
from scipy.optimize import linear_sum_assignment

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve()
sys.path.insert(0, str(PROJECT_DIR))

class HelperFunction:
    @staticmethod
    def StationaryBootstrap(data: pd.DataFrame, random_state: int = 42, n_boot: int = 10) -> List[pd.DataFrame]:
        """
        Perform stationary bootstrap on a multivariate time series.

        Args:
            data: (T, p) DataFrame, rows represent time steps, columns are features.
            random_state: int, random seed for reproducibility.
            n_boot: int, number of bootstrap replications to generate.

        Returns:
            boot_samples: list of length n, each element is a (T, p) DataFrame
                          corresponding to one stationary bootstrap sample.
        """
        rng = np.random.default_rng(random_state)
        T = len(data)
        
        # Choose p by rule of thumb: expected block size ~ N^(1/3)
        block_size = int(round(T ** (1/3)))
        p = 1.0 / block_size
        
        boot_samples = []
        data_values = data.values
        cols = data.columns

        for _ in range(n_boot):
            idx = []
            t = 0
            while t < T:
                # random start index
                start = rng.integers(0, T)
                # geometric random block length (at least 1)
                L = rng.geometric(p)
                for j in range(L):
                    idx.append((start + j) % T)  # wrap-around
                    t += 1
                    if t >= T:
                        break
            boot_df = pd.DataFrame(data_values[idx, :], columns=cols, index=data.index)
            boot_samples.append(boot_df)

        return boot_samples
    
    @staticmethod
    def disagreement_rate(s1: np.ndarray, s2: np.ndarray, K: int) -> float:
        """
        Compute the disagreement rate between two state sequences, after aligning labels.

        Args:
            s1: (T,) np.ndarray, state sequence with values in 1..K.
            s2: (T,) np.ndarray, state sequence with values in 1..K.
            K:  int, number of states.

        Returns:
            rate: float, proportion of time points where aligned s1 and s2 differ.
        """
        s1 = s1.astype(int)
        s2 = s2.astype(int)

        # Build confusion matrix C[a,b] = number of times state a in s1 aligns with state b in s2
        C = np.zeros((K, K), dtype=int)
        for a, b in zip(s1, s2):
            C[a - 1, b - 1] += 1

        # Solve assignment problem (maximize diagonal sum)
        cost = C.max() - C
        row_ind, col_ind = linear_sum_assignment(cost)

        # Construct permutation to align s2 with s1
        perm = np.zeros(K, dtype=int)
        for a_idx, b_idx in zip(row_ind, col_ind):
            perm[b_idx] = a_idx + 1
        s2_aligned = np.take(perm, s2 - 1)

        # Compute disagreement proportion
        return float(np.mean(s1 != s2_aligned))

    @staticmethod
    def stability_score(seqs: List[np.ndarray], K: int) -> float:
        """
        Compute the average pairwise disagreement score across bootstrap sequences.

        Args:
            seqs: list of (T,) np.ndarray, state sequences from bootstrap samples.
            K:    int, number of states.

        Returns:
            score: float, mean disagreement across all pairs.
        """
        B = len(seqs)
        pair = 0
        acc = 0.0
        for i in range(B):
            for j in range(i + 1, B):
                acc += HelperFunction.disagreement_rate(seqs[i], seqs[j], K)
                pair += 1
        return acc / pair

