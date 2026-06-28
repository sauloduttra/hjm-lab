"""Drift / no-arbitrage identities (HJM-01, HJM-02, DRIFT-01).

Adversarial correction:
  * HJM-02 -- the no-arbitrage residual uses ||Sigma||^2 = sum_k S_k^2 (squared
    L2 norm of the bond-vol VECTOR), NOT (sum_k S_k)^2; the square-of-sum
    injects spurious cross terms for K>=2 (invisible in any one-factor case).
"""
from __future__ import annotations

import math

import pytest

from hjm.curve import B
from hjm.vol import ConstantVol, ExponentialVol, MultiFactorVol
from hjm.drift import (hjm_drift, alpha_quadrature, noarb_residual, bond_vol,
                       bond_drift_under_Q)


# HJM-01: the no-arbitrage drift (closed form + independent quadrature) -------

def test_hjm_drift_closed_form_and_quadrature():
    for t, T in [(0.0, 5.0), (1.0, 4.0), (0.5, 7.0)]:
        cv = ConstantVol(0.012)
        assert hjm_drift(cv, t, T) == pytest.approx(0.012 ** 2 * (T - t), rel=1e-14)
        ev = ExponentialVol(0.01, 0.1)
        assert hjm_drift(ev, t, T) == pytest.approx(
            0.01 ** 2 * math.exp(-0.1 * (T - t)) * B(0.1, T - t), rel=1e-12)
        # independent Simpson leg (does NOT touch the closed-form S)
        assert hjm_drift(ev, t, T) == pytest.approx(
            alpha_quadrature(ev, t, T, 400), rel=1e-9)


def test_hjm_drift_golden_and_holee_limit():
    assert hjm_drift(ExponentialVol(0.01, 0.1), 0, 5) == \
        pytest.approx(2.3865121854e-04, rel=1e-9)
    # a -> 0 collapses to Ho-Lee sigma^2 (T-t)
    assert hjm_drift(ExponentialVol(0.01, 1e-6), 0, 5) == \
        pytest.approx(0.01 ** 2 * 5, rel=1e-5)


# HJM-02: drift condition == no-arbitrage (sum of squares!) ------------------

def test_noarb_residual_single_and_two_factor():
    for vol in (ExponentialVol(0.01, 0.1), ConstantVol(0.012)):
        for t, T in [(0.0, 5.0), (1.0, 4.0)]:
            assert noarb_residual(vol, t, T, 400) == pytest.approx(0.0, abs=1e-8)
    mf = MultiFactorVol([ExponentialVol(0.011, 0.30), ExponentialVol(0.007, 0.80)])
    assert noarb_residual(mf, 0.0, 5.0, 600) == pytest.approx(0.0, abs=1e-8)


def test_noarb_is_sum_of_squares_not_square_of_sum():
    """The adversarial guard: with two factors the integrated drift equals
    0.5*(S1^2+S2^2); the square-of-sum injects a spurious cross term S1*S2."""
    mf = MultiFactorVol([ExponentialVol(0.011, 0.30), ExponentialVol(0.007, 0.80)])
    S1 = mf.factors[0].S(0.0, 5.0)
    S2 = mf.factors[1].S(0.0, 5.0)
    int_alpha = 0.5 * (S1 ** 2 + S2 ** 2)         # == int_0^5 alpha
    assert int_alpha == pytest.approx(4.42596e-4, rel=1e-4)
    # the residual built the RIGHT way vanishes; the wrong way is off by ~S1*S2
    wrong = 0.5 * (S1 + S2) ** 2
    assert abs(wrong - int_alpha) == pytest.approx(S1 * S2, rel=1e-12)
    assert abs(wrong - int_alpha) > 2e-4          # ~2.45e-4 spurious cross term


def test_bond_drift_equals_short_rate():
    ev = ExponentialVol(0.01, 0.1)
    assert bond_drift_under_Q(ev, 0.0, 5.0, r_t=0.03, n=400) == \
        pytest.approx(0.03, abs=1e-8)


# DRIFT-01: bond-price volatility --------------------------------------------

def test_bond_vol():
    for t, T in [(0.0, 5.0), (1.0, 4.0)]:
        assert bond_vol(ConstantVol(0.012), t, T) == pytest.approx(-0.012 * (T - t), rel=1e-14)
        assert bond_vol(ExponentialVol(0.01, 0.1), t, T) == \
            pytest.approx(-0.01 * B(0.1, T - t), rel=1e-14)
    mf = MultiFactorVol([ExponentialVol(0.011, 0.30), ExponentialVol(0.007, 0.80)])
    bv = bond_vol(mf, 0.0, 5.0)
    assert len(bv) == 2                            # per-component vector
    assert bv[0] == pytest.approx(-mf.factors[0].S(0.0, 5.0), rel=1e-14)
