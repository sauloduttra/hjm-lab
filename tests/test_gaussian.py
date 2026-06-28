"""Gaussian-HJM analytics (CURVE-04, GAUSS-01..04).

Adversarial corrections:
  * GAUSS-02 -- the convexity self-block is a catastrophic difference; the
    Taylor branch below a*tau<1e-2 is MANDATORY (naive form returns ~0 at a=1e-6).
  * GAUSS-04 -- the ATM option time value is O(sigma), not O(sigma^2), so a
    fixed 1e-12 band at sigma=1e-10 fails for ATM strikes.
"""
from __future__ import annotations

import math

import pytest

from hjm.curve import B, FlatCurve, QuadraticForwardCurve
from hjm.vol import ConstantVol, ExponentialVol, MultiFactorVol
from hjm.gaussian import (gaussian_bond_price, option_variance,
                          convexity_self_block, integrated_rate_variance,
                          zcb_call, zcb_put)


def test_integrated_rate_variance_holee_golden():
    """Ho-Lee integrated-rate variance V = sigma^2 tau^3/3."""
    assert integrated_rate_variance(ConstantVol(0.01), 5.0) == \
        pytest.approx(0.0041666666666667, rel=1e-12)


# CURVE-04: curve consistency P_HJM(0,T) == P^M(0,T) exactly ------------------

def test_curve_consistency_exact():
    qc = QuadraticForwardCurve(0.03, 0.005)
    for vol in (ConstantVol(0.01), ExponentialVol(0.013, 0.18)):
        for T in (1.0, 5.0, 20.0):
            p = gaussian_bond_price(vol, qc, 0.0, T, 0.0)
            assert abs(math.log(p) - qc.log_discount(T)) < 5e-16
    assert gaussian_bond_price(ExponentialVol(0.013, 0.18), qc, 0.0, 2.0, 0.0) == \
        pytest.approx(math.exp(-0.07), abs=5e-16)


# GAUSS-01: option variance is a B-product, no Taylor ------------------------

def test_option_variance_b_product():
    for a in (0.5, 0.1, 1e-3, 1e-9):
        from hjm.curve import B
        ev = ExponentialVol(0.012, a)
        expected = 0.012 ** 2 * B(a, 5 - 2) ** 2 * B(2 * a, 2)
        assert option_variance(ev, 2, 5) == pytest.approx(expected, rel=1e-12)
    # a -> 0 Ho-Lee limit v -> sigma(T_B-T_O)sqrt(T_O)
    v = math.sqrt(option_variance(ExponentialVol(0.01, 1e-9), 2, 5))
    assert v == pytest.approx(0.01 * 3 * math.sqrt(2), rel=1e-7)
    assert math.sqrt(option_variance(ConstantVol(0.01), 2, 5)) == \
        pytest.approx(0.0424264068711929, rel=1e-7)


# GAUSS-02: convexity self-block Taylor branch -------------------------------

def test_convexity_self_block_taylor_and_continuity():
    from hjm.curve import B
    # adversarial: at a=1e-6 the naive form collapses; Taylor recovers sigma^2 tau^3/3
    assert convexity_self_block(0.01, 1e-6, 2.0) == \
        pytest.approx(0.01 ** 2 * 2.0 ** 3 / 3.0, rel=1e-5)
    # branch continuity: just below the switch the Taylor branch matches the
    # naive closed form (evaluated at the SAME a, tau, where it has not yet
    # catastrophically cancelled)
    sigma, tau = 0.01, 2.0
    a = 0.999e-2 / tau                              # a*tau just below 1e-2 -> Taylor
    taylor = convexity_self_block(sigma, a, tau)
    naive = sigma ** 2 / a ** 2 * (tau - 2 * B(a, tau) + B(2 * a, tau))
    assert taylor == pytest.approx(naive, rel=1e-8)


# GAUSS-03: ZCB option closed form + put-call parity -------------------------

def test_put_call_parity_independent_forms():
    fc = FlatCurve(0.03)
    ev = ExponentialVol(0.01, 0.1)
    for K in (0.80, 0.88, 0.904, 0.95):
        c = zcb_call(ev, fc, 1, 4, K)
        p = zcb_put(ev, fc, 1, 4, K)
        assert c - p == pytest.approx(fc.discount(4) - K * fc.discount(1), abs=1e-13)
        assert 0.0 <= c <= fc.discount(4) + 1e-15
    # call strictly decreasing in K
    calls = [zcb_call(ev, fc, 1, 4, K) for K in (0.80, 0.88, 0.95)]
    assert calls[0] > calls[1] > calls[2]
    # v -> 0 intrinsic
    intrinsic = max(fc.discount(4) - 0.88 * fc.discount(1), 0.0)
    assert zcb_call(ExponentialVol(1e-13, 0.1), fc, 1, 4, 0.88) == \
        pytest.approx(intrinsic, abs=1e-12)


# GAUSS-04: sigma -> 0 deterministic limit -----------------------------------

def test_sigma_to_zero_bond_is_deterministic_forward():
    fc = FlatCurve(0.03)
    ev = ExponentialVol(1e-10, 0.1)
    assert gaussian_bond_price(ev, fc, 1.5, 4.0, 0.0) == \
        pytest.approx(fc.discount(4.0) / fc.discount(1.5), rel=1e-12)


def test_sigma_to_zero_option_atm_is_order_sigma():
    """ITM/OTM collapse to intrinsic at 1e-12; ATM gap is O(sigma) ~1.13*sigma."""
    fc = FlatCurve(0.03)
    P_TO, P_TB = fc.discount(2), fc.discount(5)
    # strictly OTM strike: intrinsic 0, option ~0 at tiny sigma
    K_otm = 1.5 * P_TB / P_TO
    assert zcb_call(ExponentialVol(1e-10, 0.1), fc, 2, 5, K_otm) == \
        pytest.approx(max(P_TB - K_otm * P_TO, 0.0), abs=1e-12)
    # ATM: the gap is linear in sigma, so abs<1e-12 at sigma=1e-10 would FAIL
    K_atm = P_TB / P_TO
    gap = abs(zcb_call(ExponentialVol(1e-10, 0.1), fc, 2, 5, K_atm)
              - max(P_TB - K_atm * P_TO, 0.0))
    assert gap > 1e-12                             # O(sigma), not O(sigma^2)
    assert gap < 2e-10                             # but bounded by ~2*sigma


# gaussian_bond_price at t>0 with nonzero state (the convexity + state term) --

def test_gaussian_bond_price_convexity_and_state_at_t_positive():
    fc = FlatCurve(0.03)
    ev = ExponentialVol(0.013, 0.18)
    t, T = 1.5, 4.0
    assert gaussian_bond_price(ev, fc, t, T, 0.0) == \
        pytest.approx(0.9271027306391553, rel=1e-13)
    conv = 0.5 * (integrated_rate_variance(ev, T - t)
                  - integrated_rate_variance(ev, T)
                  + integrated_rate_variance(ev, t))
    log_ratio = fc.log_discount(T) - fc.log_discount(t)
    for x in (-0.01, 0.01):
        expected = math.exp(log_ratio + conv - B(ev.a, T - t) * x)
        assert gaussian_bond_price(ev, fc, t, T, x) == pytest.approx(expected, rel=1e-13)
    # a higher state (= higher rates) gives a lower bond price
    assert (gaussian_bond_price(ev, fc, t, T, -0.01)
            > gaussian_bond_price(ev, fc, t, T, 0.0)
            > gaussian_bond_price(ev, fc, t, T, 0.01))


# multi-factor Gaussian analytics --------------------------------------------

def test_multi_factor_gaussian_analytics():
    from hjm.models import reproduce_g2pp_V
    mf = MultiFactorVol([ExponentialVol(0.011, 0.30), ExponentialVol(0.007, 0.80)])
    qc = QuadraticForwardCurve(0.03, 0.005)
    # option variance is the per-factor quadrature sum
    expected_ov = sum(f.sigma_param ** 2 * B(f.a, 5 - 2) ** 2 * B(2 * f.a, 2)
                      for f in mf.factors)
    assert option_variance(mf, 2, 5) == pytest.approx(expected_ov, rel=1e-12)
    # integrated-rate variance == sum of self blocks == reproduce_g2pp_V(rho=0)
    irv = integrated_rate_variance(mf, 4.0)
    assert irv == pytest.approx(
        convexity_self_block(0.011, 0.30, 4.0) + convexity_self_block(0.007, 0.80, 4.0),
        rel=1e-12)
    assert irv == pytest.approx(reproduce_g2pp_V(0.011, 0.30, 0.007, 0.80, 0.0, 4.0), rel=1e-12)
    # parity holds under two factors
    for K in (0.80, 0.90, 0.95):
        assert zcb_call(mf, qc, 1, 4, K) - zcb_put(mf, qc, 1, 4, K) == \
            pytest.approx(qc.discount(4) - K * qc.discount(1), abs=1e-13)
    # vector state matches an independent recomputation
    x1, x2 = 0.01, -0.005
    conv = 0.5 * (integrated_rate_variance(mf, 3) - integrated_rate_variance(mf, 4)
                  + integrated_rate_variance(mf, 1))
    log_ratio = qc.log_discount(4) - qc.log_discount(1)
    expected = math.exp(log_ratio + conv - B(0.30, 3) * x1 - B(0.80, 3) * x2)
    assert gaussian_bond_price(mf, qc, 1, 4, [x1, x2]) == pytest.approx(expected, rel=1e-13)
    # wrong-length state raises
    with pytest.raises(ValueError):
        gaussian_bond_price(mf, qc, 1, 4, [x1])
