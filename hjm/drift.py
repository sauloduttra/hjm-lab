"""The HJM no-arbitrage drift and the deflated-bond martingale condition.

THE headline of the framework: under the risk-neutral measure Q the
forward-rate drift is NOT free -- no-arbitrage forces

    alpha(t,T) = sum_k sigma_k(t,T) * int_t^T sigma_k(t,u) du = sum_k sigma_k S_k.

Equivalently the discounted bond P(t,T)/B(t) is a Q-martingale, which is the
condition that the integrated drift equals half the squared bond-price
volatility:

    int_t^T alpha(t,u) du = 1/2 * ||Sigma(t,T)||^2 = 1/2 * sum_k S_k(t,T)^2.

The norm is the SUM OF SQUARES of the per-factor bond-vol components -- NOT the
square of their scalar sum, which for K>=2 would inject spurious cross terms
sum_{i<j} S_i S_j (an adversarial catch; invisible in every one-factor case).

Two independent code paths keep the central test falsifiable: `alpha` as the
analytic product sigma_k*S_k, vs `alpha_quadrature` which re-integrates
sigma_k by Simpson without ever touching the closed-form S.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "hjm_drift", "bond_vol", "alpha_quadrature",
    "noarb_residual", "bond_drift_under_Q",
]


def _simpson(g, a: float, b: float, n: int) -> float:
    """Composite Simpson of a scalar function g on [a, b] with n panels."""
    if n % 2 == 1:
        n += 1
    x = np.linspace(a, b, n + 1)
    y = np.array([g(float(xi)) for xi in x])
    h = (b - a) / n
    return float(h / 3.0 * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum()
                            + 2.0 * y[2:-2:2].sum()))


def hjm_drift(vol, t: float, T: float) -> float:
    """alpha(t,T) = sum_k sigma_k(t,T) * S_k(t,T)  (the closed-form leg)."""
    return sum(f.sigma(t, T) * f.S(t, T) for f in vol.factors)


def bond_vol(vol, t: float, T: float):
    """Bond-price volatility Sigma(t,T) = -S(t,T) per factor.

    Scalar for a one-factor structure; the per-component vector (-S_1,...,-S_K)
    for a multi-factor one.
    """
    S = [f.S(t, T) for f in vol.factors]
    if len(S) == 1:
        return -S[0]
    return -np.array(S)


def alpha_quadrature(vol, t: float, T: float, n: int = 400) -> float:
    """alpha(t,T) with S_k re-integrated by Simpson (independent of the
    closed-form S) -- the falsifiable leg of the drift identity."""
    total = 0.0
    for f in vol.factors:
        S_quad = _simpson(lambda u: f.sigma(t, u), t, T, n)
        total += f.sigma(t, T) * S_quad
    return total


def noarb_residual(vol, t: float, T: float, n: int = 400) -> float:
    """-int_t^T alpha(t,u) du + 1/2 * sum_k S_k(t,T)^2.

    Zero iff the HJM drift condition holds (the deflated bond is a
    Q-martingale).  The integral is taken by independent Simpson quadrature so
    the two sides are genuinely separate numeric paths.
    """
    int_alpha = _simpson(lambda u: hjm_drift(vol, t, u), t, T, n)
    sum_sq = sum(f.S(t, T) ** 2 for f in vol.factors)
    return -int_alpha + 0.5 * sum_sq


def bond_drift_under_Q(vol, t: float, T: float, r_t: float, n: int = 400) -> float:
    """The instantaneous drift of dP(t,T)/P(t,T) under Q.

    Equals r_t exactly when the drift condition holds (it is r_t plus the
    `noarb_residual`, which vanishes), so the test asserts this returns r_t.
    """
    return r_t + noarb_residual(vol, t, T, n)
