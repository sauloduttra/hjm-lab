"""Gaussian-HJM analytics for a deterministic vol structure.

When sigma(t,T) is deterministic, the forward rates and log-bond are Gaussian,
so bond prices and ZCB options have closed forms.

A CRITICAL numerical split runs through this module:

  * the OPTION integrated variance v^2 = sigma^2 B(a,T_B-T_O)^2 B(2a,T_O) is a
    PRODUCT of B's -> expm1-stable, no Taylor branch (accurate down to a=1e-9);
  * the CONVEXITY self-block V(tau) = sigma^2/a^2 [tau-2B(a,tau)+B(2a,tau)] is a
    DIFFERENCE of O(1/a^3) terms that cancels catastrophically -> the Taylor
    series is MANDATORY below a*tau < 1e-2 (the naive form returns exactly 0.0
    at a=1e-6).  Same pattern as shortrate-lab vasicek.A and g2pp factors.V.

The ZCB call and put are each their OWN Black-style expression (the put is NOT
`call - forward`), so put-call parity is a genuine cross-check.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from hjm.curve import B, ZeroCurve
from hjm.vol import VolStructure

__all__ = [
    "convexity_self_block", "integrated_rate_variance", "option_variance",
    "gaussian_bond_price", "zcb_call", "zcb_put",
]

_TAYLOR_SWITCH = 1e-2


def convexity_self_block(sigma: float, a: float, tau: float) -> float:
    """One self block of the integrated-rate variance:

        sigma^2/a^2 [tau - 2 B(a,tau) + B(2a,tau)]  ==  sigma^2 int_0^tau B(a,s)^2 ds

    A catastrophic difference near a -> 0, so below a*tau < 1e-2 the derived
    Taylor series is used (leading order sigma^2 tau^3 / 3).
    """
    w = a * tau
    if w < _TAYLOR_SWITCH:
        t3 = tau * tau * tau
        series = (1.0 / 3.0 - w / 4.0 + 7.0 * w * w / 60.0 - w ** 3 / 24.0
                  + 31.0 * w ** 4 / 2520.0 - w ** 5 / 320.0)
        return sigma * sigma * t3 * series
    return sigma * sigma / (a * a) * (tau - 2.0 * B(a, tau) + B(2.0 * a, tau))


def integrated_rate_variance(vol: VolStructure, tau: float) -> float:
    """V(tau) = Var_Q[int_0^tau r du] for a structure of INDEPENDENT factors:
    the sum of each factor's self block (correlation between named factors is
    handled explicitly in `models.reproduce_g2pp_V`)."""
    total = 0.0
    for f in vol.factors:
        total += convexity_self_block(f.sigma_param, getattr(f, "a", 0.0), tau)
    return total


def option_variance(vol: VolStructure, T_O: float, T_B: float) -> float:
    """Integrated forward-bond-price variance v^2 = int_0^{T_O}
    [Sigma(s,T_B)-Sigma(s,T_O)]^2 ds.

    For each factor v_k^2 = sigma_k^2 B(a_k, T_B-T_O)^2 B(2 a_k, T_O) -- a
    PRODUCT of B's, expm1-stable with NO Taylor branch (independent factors
    add in quadrature; a_k=0 gives the Ho-Lee limit sigma^2 (T_B-T_O)^2 T_O).
    """
    total = 0.0
    for f in vol.factors:
        a = getattr(f, "a", 0.0)
        total += f.sigma_param ** 2 * B(a, T_B - T_O) ** 2 * B(2.0 * a, T_O)
    return total


def gaussian_bond_price(vol: VolStructure, curve: ZeroCurve, t: float, T: float,
                        f_state=0.0) -> float:
    """Curve-consistent Gaussian-HJM bond price

        P(t,T) = (P^M(0,T)/P^M(0,t)) exp(A(t,T)),
        A(t,T) = 1/2 [V(T-t) - V(T) + V(t)] - sum_k B(a_k, T-t) x_k,

    with x_k the realized factor states (`f_state`, scalar for one factor).
    A(0,T) = 0 and x = 0 give P(0,T) = P^M(0,T) exactly (curve consistency).
    Evaluated in log space (a difference of log-discounts), never a ratio of
    two tiny numbers.
    """
    factors = vol.factors
    if np.ndim(f_state) == 0:
        states = [f_state] * len(factors)
    else:
        states = list(f_state)
        if len(states) != len(factors):
            raise ValueError(
                "f_state must have one entry per factor "
                f"({len(factors)} factors, got {len(states)})")
    convexity = 0.5 * (integrated_rate_variance(vol, T - t)
                       - integrated_rate_variance(vol, T)
                       + integrated_rate_variance(vol, t))
    state_term = sum(B(getattr(f, "a", 0.0), T - t) * x
                     for f, x in zip(factors, states))
    A = convexity - state_term
    log_ratio = curve.log_discount(T) - curve.log_discount(t)
    return math.exp(log_ratio + A)


def _v_and_prices(vol, curve, T_O, T_B):
    """Return (option vol v, P(0,T_B), P(0,T_O)) -- the shared pieces of the
    ZCB call/put (the strike-dependent h is formed by each caller)."""
    v = math.sqrt(option_variance(vol, T_O, T_B))
    return v, curve.discount(T_B), curve.discount(T_O)


def zcb_call(vol: VolStructure, curve: ZeroCurve, T_O: float, T_B: float,
             K: float) -> float:
    """European call expiring T_O on a ZCB maturing T_B > T_O, strike K:
    Call = P(0,T_B) Phi(h) - K P(0,T_O) Phi(h - v).  Degenerate v == 0 returns
    the intrinsic max(P(0,T_B) - K P(0,T_O), 0)."""
    v, p_tb, p_to = _v_and_prices(vol, curve, T_O, T_B)
    if v <= 0.0:
        return max(p_tb - K * p_to, 0.0)
    h = math.log(p_tb / (K * p_to)) / v + 0.5 * v
    return p_tb * norm.cdf(h) - K * p_to * norm.cdf(h - v)


def zcb_put(vol: VolStructure, curve: ZeroCurve, T_O: float, T_B: float,
            K: float) -> float:
    """European put, from its OWN Black-style form (NOT call - forward), so
    parity is a genuine test: Put = K P(0,T_O) Phi(-(h-v)) - P(0,T_B) Phi(-h)."""
    v, p_tb, p_to = _v_and_prices(vol, curve, T_O, T_B)
    if v <= 0.0:
        return max(K * p_to - p_tb, 0.0)
    h = math.log(p_tb / (K * p_to)) / v + 0.5 * v
    return K * p_to * norm.cdf(-(h - v)) - p_tb * norm.cdf(-h)
