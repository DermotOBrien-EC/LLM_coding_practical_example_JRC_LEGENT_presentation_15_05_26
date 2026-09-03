"""Incremental ridge over nested training prefixes.

Every backtest origin trains on a prefix of the same series, so the raw cross-product
moments can be accumulated once instead of refitting from scratch at each origin.
Algebraically identical to loadfc.RidgeLog with w=None.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Moments:
    n: int
    Sxx: np.ndarray
    Sx: np.ndarray
    Sxy: np.ndarray
    Sy: float


class MomentPath:
    """Accumulates X'X, X'1, X'y, sum(y) over an increasing sequence of prefix lengths."""

    def __init__(self, A: np.ndarray, t: np.ndarray) -> None:
        self.A = np.ascontiguousarray(A, dtype=float)
        self.t = np.ascontiguousarray(t, dtype=float)
        p = self.A.shape[1]
        self._pos = 0
        self._m = Moments(0, np.zeros((p, p)), np.zeros(p), np.zeros(p), 0.0)

    def upto(self, n: int) -> Moments:
        if n < self._pos:
            raise ValueError("prefix lengths must be non-decreasing")
        if n > self._pos:
            B, tb = self.A[self._pos : n], self.t[self._pos : n]
            self._m.Sxx += B.T @ B
            self._m.Sx += B.sum(axis=0)
            self._m.Sxy += B.T @ tb
            self._m.Sy += float(tb.sum())
            self._m.n = n
            self._pos = n
        return self._m


@dataclass
class RidgeFit:
    mu: np.ndarray
    sd: np.ndarray
    beta: np.ndarray
    b0: float

    def predict(self, rows: np.ndarray) -> np.ndarray:
        return np.exp(self.b0 + ((rows - self.mu) / self.sd) @ self.beta)


def fit_from_moments(m: Moments, alpha: float) -> RidgeFit:
    n = m.n
    mu = m.Sx / n
    var = np.diag(m.Sxx) / n - mu**2
    sd = np.sqrt(np.clip(var, 0.0, None))
    sd[sd <= 0] = 1.0
    inv = 1.0 / sd
    Gz = (m.Sxx - n * np.outer(mu, mu)) * inv[:, None] * inv[None, :]
    b0 = m.Sy / n
    rhs = (m.Sxy - mu * m.Sy) * inv
    beta = np.linalg.solve(Gz + alpha * n / 1000.0 * np.eye(len(mu)), rhs)
    return RidgeFit(mu=mu, sd=sd, beta=beta, b0=b0)
