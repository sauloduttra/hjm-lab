"""Named special cases + the cross-model reproductions.

The HJM framework is only convincing if its special cases REPRODUCE the
sibling short-rate models -- so the reproductions are recomputed FROM SCRATCH
inline here (there is no `import shortrate` / `import g2pp` anywhere in
hjm-lab; the agreement is a theorem, not an echo):

  * ConstantVol  -> Ho-Lee (no mean reversion);
  * ExponentialVol -> Hull-White / Vasicek -- the Gaussian-HJM ZCB-option vol
    equals Vasicek's option_sigma_p (the flagship);
  * two exponential factors -> G2++ V(tau) = Var_Q[int_0^tau r].

The Ho-Lee bond price MUST carry the realized short-rate state r(t); the
t=0-only state-free form is wrong for t > 0 (it errs by up to ~13%).
"""
from __future__ import annotations

import math

from hjm.curve import B, ZeroCurve
from hjm.gaussian import convexity_self_block
from hjm.vol import ConstantVol, ExponentialVol, MultiFactorVol, VolStructure

__all__ = [
    "ho_lee", "hull_white", "two_factor_exponential",
    "ho_lee_bond_price", "reproduce_vasicek_sigma_p", "reproduce_g2pp_V",
    "implied_short_rate_drift",
]

_TAYLOR_SWITCH = 1e-2


def ho_lee(sigma: float) -> ConstantVol:
    """Ho-Lee (1986): constant vol; implied dr = theta(t)dt + sigma dW."""
    return ConstantVol(sigma)


def hull_white(sigma: float, a: float) -> ExponentialVol:
    """Hull-White (1990): exponential vol; implied dr = [theta(t)-a r]dt + sigma dW."""
    return ExponentialVol(sigma, a)


def two_factor_exponential(sigma: float, a: float, eta: float, b: float,
                           rho: float) -> MultiFactorVol:
    """Two exponential factors -> a G2++-style two-factor Gaussian model.

    The structure itself holds two INDEPENDENT factors (drift and option vol
    add in quadrature); the inter-factor correlation `rho` enters only the
    integrated-rate variance and is consumed by `reproduce_g2pp_V`.
    """
    return MultiFactorVol([ExponentialVol(sigma, a), ExponentialVol(eta, b)])


def ho_lee_bond_price(sigma: float, curve: ZeroCurve, t: float, T: float,
                      r_t: float) -> float:
    """Ho-Lee zero-coupon bond, carrying the realized short rate r(t):

        P(t,T) = (P^M(0,T)/P^M(0,t)) exp(-(T-t)(r(t)-f^M(0,t))
                                          - 1/2 sigma^2 t (T-t)^2).

    The state term -(T-t)(r(t)-f^M(0,t)) is mandatory; dropping it (the
    t=0-only form) errs by up to ~13% for t > 0.
    """
    tau = T - t
    f0t = curve.inst_forward(t)
    log_ratio = curve.log_discount(T) - curve.log_discount(t)
    A = -tau * (r_t - f0t) - 0.5 * sigma * sigma * t * tau * tau
    return math.exp(log_ratio + A)


def reproduce_vasicek_sigma_p(sigma: float, a: float, T_O: float,
                              T_B: float) -> float:
    """The Vasicek/Hull-White ZCB-option volatility, recomputed from scratch:

        sigma_p = sigma * B(a, T_B-T_O) * sqrt((1 - e^{-2a T_O})/(2a)).

    Equals sqrt(gaussian.option_variance(ExponentialVol(sigma,a),T_O,T_B)).
    The a -> 0 limit of the integrated-variance factor is T_O.
    """
    var_int = T_O if a == 0.0 else -math.expm1(-2.0 * a * T_O) / (2.0 * a)
    return sigma * B(a, T_B - T_O) * math.sqrt(var_int)


def _cross_block(sigma: float, a: float, eta: float, b: float, rho: float,
                 tau: float) -> float:
    """The rho-dependent cross block of the two-factor integrated-rate variance

        2 rho sigma eta/(a b) [tau - B(a,tau) - B(b,tau) + B(a+b,tau)],

    Taylor-branched below (a+b)*tau < 1e-2 (catastrophic difference)."""
    pref = 2.0 * rho * sigma * eta
    if (a + b) * tau < _TAYLOR_SWITCH:
        t3 = tau * tau * tau
        series = (1.0 / 3.0
                  - (a + b) * tau / 8.0
                  + (2.0 * a * a + 3.0 * a * b + 2.0 * b * b) * tau * tau / 60.0
                  - (a ** 3 + 2.0 * a * a * b + 2.0 * a * b * b + b ** 3) * tau ** 3 / 144.0)
        return pref * t3 * series
    return pref / (a * b) * (tau - B(a, tau) - B(b, tau) + B(a + b, tau))


def reproduce_g2pp_V(sigma: float, a: float, eta: float, b: float, rho: float,
                     tau: float) -> float:
    """G2++ V(tau) = Var_Q[int_0^tau r du] recomputed inline:

        sigma^2/a^2[tau-2B(a,tau)+B(2a,tau)] + eta^2/b^2[...b...]
        + 2 rho sigma eta/(a b)[tau - B(a,tau) - B(b,tau) + B(a+b,tau)].

    Self blocks via `convexity_self_block` (Taylor-branched); cross block via
    `_cross_block`.  Leading order (sigma^2+eta^2+2 rho sigma eta) tau^3/3.
    """
    return (convexity_self_block(sigma, a, tau)
            + convexity_self_block(eta, b, tau)
            + _cross_block(sigma, a, eta, b, rho, tau))


def implied_short_rate_drift(vol: VolStructure, curve: ZeroCurve, t: float,
                             h: float = 1e-5) -> float:
    """The implied short-rate drift theta(t) of the single-factor model that
    reproduces this curve:

        Ho-Lee   (a=0): theta(t) = f_t(0,t) + sigma^2 t
        Hull-White:     theta(t) = f_t(0,t) + a f(0,t) + (sigma^2/(2a))(1-e^{-2a t})

    f_t(0,t) is the curve forward's slope (central difference, O(h^2)).
    """
    if len(vol.factors) != 1:
        raise ValueError("implied_short_rate_drift is single-factor only")
    f = vol.factors[0]
    a = getattr(f, "a", 0.0)
    sig = f.sigma_param
    f0 = curve.inst_forward(t)
    f_t = (curve.inst_forward(t + h) - curve.inst_forward(t - h)) / (2.0 * h)
    if a == 0.0:
        return f_t + sig * sig * t
    return f_t + a * f0 + sig * sig / (2.0 * a) * (-math.expm1(-2.0 * a * t))
