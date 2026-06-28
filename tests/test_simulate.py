"""Musiela consistency + Monte Carlo martingales (REPRO-01, MC-01..03).

The MC is an honest forward-curve Euler in Musiela coordinates (exact
transport via the index shift, genuine O(dt) drift bias).  Martingale tests
use 4-sigma bands with reported z-scores -- never a point equality.

Note: for a flat curve P^M(0,S)/P^M(0,T) = exp(-r(S-T)) = exp(-0.09) here
(NOT exp(-0.15), which is P^M(0,S) itself -- a spec golden typo, corrected).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from hjm.curve import B, FlatCurve
from hjm.vol import ExponentialVol
from hjm.drift import hjm_drift
from hjm.simulate import (musiela_drift, mc_discount_factor, mc_forward_bond,
                          mc_zcb_option, discretization_bias_probe)
from hjm.gaussian import zcb_call


# REPRO-01: Musiela drift coupling == maturity-form alpha --------------------

def test_musiela_drift_equals_maturity_alpha():
    ev = ExponentialVol(0.01, 0.1)
    for x in (0.5, 2.0, 5.0):
        assert musiela_drift(ev, x) == pytest.approx(hjm_drift(ev, 0.0, x), rel=1e-11)


# MC-01: risk-neutral (bank-account) martingale ------------------------------

def test_mc_discount_factor_martingale():
    ev, fc = ExponentialVol(0.01, 0.1), FlatCurve(0.03)
    mean, se = mc_discount_factor(ev, fc, 3.0, 150, 40000, np.random.default_rng(0))
    target = math.exp(-0.09)                       # P^M(0,3)
    z = (mean - target) / se
    assert abs(z) < 4.0, f"z={z:.2f}"


def test_mc_no_systematic_bias():
    """Seed-averaging cancels the MC sampling noise that swamps a single run;
    the residual systematic (Euler) bias is then consistent with zero -- a
    falsifiable check on the drift condition (a wrong drift would survive the
    averaging as |bias| >> bias_se).  (The O(dt) bias is genuinely O(dt) and
    resolvable at higher vol -- see examples/forward_curve_mc.py.)"""
    ev, fc = ExponentialVol(0.01, 0.1), FlatCurve(0.03)
    probe = discretization_bias_probe(ev, fc, 3.0, n_steps=150, n_paths=12000,
                                      n_seeds=10, base_seed=100)
    assert abs(probe["bias"]) < 4 * probe["bias_se"]


# MC-02: forward-measure martingale ------------------------------------------

def test_mc_forward_bond_martingale():
    ev, fc = ExponentialVol(0.01, 0.1), FlatCurve(0.03)
    mean, se = mc_forward_bond(ev, fc, 2.0, 5.0, 150, 40000, np.random.default_rng(1))
    target = fc.discount(5) / fc.discount(2)       # = exp(-0.09)
    z = (mean - target) / se
    assert abs(z) < 4.0, f"z={z:.2f}"


# MC-03: MC option reproduces the Gaussian closed form -----------------------

def test_mc_zcb_option_matches_analytic():
    ev, fc = ExponentialVol(0.01, 0.1), FlatCurve(0.03)
    K = 0.90
    mean, se = mc_zcb_option(ev, fc, 1.0, 4.0, K, 150, 40000, np.random.default_rng(2))
    analytic = zcb_call(ev, fc, 1.0, 4.0, K)       # computed independently of the MC
    z = (mean - analytic) / se
    assert abs(z) < 4.0, f"z={z:.2f}"
