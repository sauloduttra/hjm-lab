"""Deterministic HJM volatility structures sigma(t,T).

Each structure exposes BOTH the instantaneous vol `sigma(t, T)` AND its
T-integral `S(t, T) = int_t^T sigma(t, u) du` -- the single primitive every
other module reuses (the bond-price volatility is just `-S`).  Routing
everything through `S` means a bug in the integral is caught once, centrally.

For `ExponentialVol`, `S = sigma * B(a, T-t)` is a PRODUCT through the
cancellation-free `B`, so no Taylor branch is needed here (the catastrophic
difference shows up later, in the integrated-rate convexity -- see
`gaussian.convexity_self_block`).

Multi-factor structures hold a list of independent single-factor `factors`;
single-factor structures report `factors == [self]`, so drift/no-arb code can
iterate `vol.factors` uniformly.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

from hjm.curve import B

__all__ = ["VolStructure", "ConstantVol", "ExponentialVol", "MultiFactorVol"]


class VolStructure(ABC):
    """A deterministic HJM vol structure (single- or multi-factor)."""

    @property
    def factors(self) -> list:
        """The independent single-factor components (just [self] for a leaf)."""
        return [self]

    @property
    def n_factors(self) -> int:
        return len(self.factors)


class _SingleFactor(VolStructure):
    """A one-factor leaf: defines sigma(t,T) and its integral S(t,T)."""

    @abstractmethod
    def sigma(self, t: float, T: float) -> float:
        """Instantaneous forward-rate volatility sigma(t, T)."""

    @abstractmethod
    def S(self, t: float, T: float) -> float:
        """int_t^T sigma(t, u) du  (= minus the bond-price volatility)."""


class ConstantVol(_SingleFactor):
    """Ho-Lee: sigma(t,T) = sigma (constant); S(t,T) = sigma*(T-t)."""

    def __init__(self, sigma: float):
        if sigma < 0.0:
            raise ValueError("sigma must be >= 0")
        self._sigma = sigma

    @property
    def sigma_param(self) -> float:
        return self._sigma

    def sigma(self, t: float, T: float) -> float:
        return self._sigma

    def S(self, t: float, T: float) -> float:
        return self._sigma * (T - t)


class ExponentialVol(_SingleFactor):
    """Hull-White: sigma(t,T) = sigma*e^{-a(T-t)}; S(t,T) = sigma*B(a, T-t)."""

    def __init__(self, sigma: float, a: float):
        if sigma < 0.0:
            raise ValueError("sigma must be >= 0")
        if a < 0.0:
            raise ValueError("a must be >= 0")
        self._sigma = sigma
        self.a = a

    @property
    def sigma_param(self) -> float:
        return self._sigma

    def sigma(self, t: float, T: float) -> float:
        return self._sigma * math.exp(-self.a * (T - t))

    def S(self, t: float, T: float) -> float:
        return self._sigma * B(self.a, T - t)


class MultiFactorVol(VolStructure):
    """K independent single-factor structures driven by independent dW_k.

    (Correlation between factors is modelled, equivalently, by independent
    Brownian motions plus the cross term in the integrated-rate variance --
    see `gaussian.integrated_rate_variance` / `models.two_factor_exponential`.)
    """

    def __init__(self, factors: list):
        if not factors:
            raise ValueError("need at least one factor")
        if not all(isinstance(f, _SingleFactor) for f in factors):
            raise TypeError("MultiFactorVol factors must be single-factor leaves "
                            "(ConstantVol / ExponentialVol)")
        self._factors = list(factors)

    @property
    def factors(self) -> list:
        return self._factors
