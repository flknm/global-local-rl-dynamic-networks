"""Probability distributions used by Algorithm 1."""

from __future__ import annotations

import numpy as np


def _validate_probability(p: float) -> float:
    p = float(p)
    if not 0.0 < p <= 1.0:
        raise ValueError(f"p must lie in (0, 1], got {p}.")
    return p


def geometric_pmf(k: int, p: float) -> float:
    """Return ``P(X=k)=p(1-p)^k`` for a geometric variable on ``N_0``."""
    p = _validate_probability(p)
    if int(k) != k or k < 0:
        return 0.0
    return p * (1.0 - p) ** int(k)


def sample_geometric_n0(
    rng: np.random.Generator,
    p: float,
    size: int | tuple[int, ...] | None = None,
):
    """Sample the geometric law on ``N_0``."""
    p = _validate_probability(p)
    draws = rng.geometric(p, size=size) - 1
    if size is None:
        return int(draws)
    return draws.astype(np.int64, copy=False)
