"""The market curve P^M(0,T) that the HJM model auto-calibrates to.

Everything downstream is built on the instantaneous forward rate f^M(0,T) =
-d ln P^M(0,T)/dT and the discount factor P^M(0,T).  Each curve exposes a
CLOSED-FORM `inst_forward` (so curve-consistency carries zero differentiation
noise) plus `log_discount`; the model-agnostic central-difference
`fd_inst_forward` exists ONLY to cross-check the analytic forward at O(h^2).

`B(z, tau) = (1 - e^{-z*tau})/z` is the cancellation-free primitive reused
across the whole rates pillar (shortrate-lab, g2pp-lab, and here): implemented
as `-expm1(-z*tau)/z` with an explicit `z == 0 -> tau` branch.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np

__all__ = [
    "B", "ZeroCurve", "FlatCurve", "QuadraticForwardCurve",
    "NelsonSiegelCurve", "fd_inst_forward", "bond_from_forward",
]


def B(z: float, tau: float) -> float:
    """B(z, tau) = (1 - e^{-z tau})/z, via -expm1 (cancellation-free).

    Explicit z == 0 -> tau branch (so B(0, tau) == tau exactly and no
    ZeroDivisionError); B(z, 0) == 0 exactly; B -> 1/z as tau -> inf.
    """
    if z == 0.0:
        return tau
    return -math.expm1(-z * tau) / z


class ZeroCurve(ABC):
    """An initial discount curve P^M(0,T) with an analytic forward."""

    @abstractmethod
    def log_discount(self, T: float) -> float:
        """ln P^M(0, T)."""

    @abstractmethod
    def inst_forward(self, T: float) -> float:
        """The analytic instantaneous forward f^M(0, T) = -d ln P/dT."""

    def discount(self, T: float) -> float:
        """P^M(0, T) = exp(log_discount(T))."""
        return math.exp(self.log_discount(T))

    def zero_yield(self, T: float) -> float:
        """Continuously-compounded spot yield -ln P(0,T)/T; -> f(0,0) as T->0."""
        if T == 0.0:
            return self.inst_forward(0.0)
        return -self.log_discount(T) / T


class FlatCurve(ZeroCurve):
    """Flat forward curve f^M(0,T) = r (the Ho-Lee golden workhorse)."""

    def __init__(self, r: float):
        self.r = r

    def log_discount(self, T: float) -> float:
        return -self.r * T

    def inst_forward(self, T: float) -> float:
        return self.r


class QuadraticForwardCurve(ZeroCurve):
    """Affine forward f^M(0,T) = c0 + c1*T; integrates exactly under GL.

    -ln P(0,T) = int_0^T f du = c0*T + c1*T^2/2.
    """

    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def log_discount(self, T: float) -> float:
        return -(self.c0 * T + 0.5 * self.c1 * T * T)

    def inst_forward(self, T: float) -> float:
        return self.c0 + self.c1 * T


class NelsonSiegelCurve(ZeroCurve):
    """Nelson-Siegel (1987) forward f^M(0,T) = b0 + b1 e^{-T/lam}
    + b2 (T/lam) e^{-T/lam}.

    Has a nonzero third derivative of -ln P, so it genuinely exercises the
    O(h^2) finite-difference forward.  The forward integral is closed form:

      int_0^T f du = b0 T + b1 lam (1-e^{-T/lam})
                     + b2 [lam(1-e^{-T/lam}) - T e^{-T/lam}].
    """

    def __init__(self, beta0: float, beta1: float, beta2: float, lam: float):
        if lam <= 0.0:
            raise ValueError("lam must be > 0")
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.lam = lam

    def log_discount(self, T: float) -> float:
        lam = self.lam
        e = math.exp(-T / lam)
        integral = (self.beta0 * T
                    + self.beta1 * lam * (1.0 - e)
                    + self.beta2 * (lam * (1.0 - e) - T * e))
        return -integral

    def inst_forward(self, T: float) -> float:
        e = math.exp(-T / self.lam)
        return self.beta0 + self.beta1 * e + self.beta2 * (T / self.lam) * e


def fd_inst_forward(price_fn, T: float, h: float = 1e-5) -> float:
    """Model-agnostic instantaneous forward via a central difference of
    -ln(price):  (ln P(T-h) - ln P(T+h)) / (2h).  Truncation O(h^2), so it is
    banded at ~1e-6/1e-7, NEVER 1e-12."""
    return (math.log(price_fn(T - h)) - math.log(price_fn(T + h))) / (2.0 * h)


def bond_from_forward(curve: ZeroCurve, T: float, n_quad: int = 64) -> float:
    """P(0,T) = exp(-int_0^T f^M(0,u) du) via Gauss-Legendre quadrature.

    Exact (to rounding) for polynomial forwards; the round-trip
    bond_from_forward == curve.discount pins the forward<->bond consistency.
    """
    if T == 0.0:
        return 1.0
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    # map [-1, 1] -> [0, T]
    u = 0.5 * T * (nodes + 1.0)
    fu = np.array([curve.inst_forward(float(x)) for x in u])
    integral = 0.5 * T * float(np.dot(weights, fu))
    return math.exp(-integral)
