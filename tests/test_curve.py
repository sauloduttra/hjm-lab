"""Curve identities (CURVE-01..03).

Adversarial corrections:
  * CURVE-01 -- B needs an explicit z==0 -> tau branch (else B(0,tau) raises
    ZeroDivisionError), and z*B == -expm1 is rtol 1e-13, NOT bitwise-exact
    (the v/z then z*(v/z) reintroduces ~1 ulp on ~14% of inputs).
  * CURVE-03 -- the FD forward is a genuine O(h^2); 1e-12 is unjustifiable
    (fails ~83% of maturities at h=1e-5).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from hjm.curve import (B, FlatCurve, QuadraticForwardCurve, NelsonSiegelCurve,
                       fd_inst_forward, bond_from_forward)


# CURVE-01: the shared primitive B(z, tau) -----------------------------------

def test_B_defining_identity_and_limits():
    rng = np.random.default_rng(0)
    for _ in range(500):
        z = rng.uniform(-5, 5)
        tau = rng.uniform(0, 50)
        assert math.isclose(z * B(z, tau), -math.expm1(-z * tau), rel_tol=1e-13)
    # exact limits require the z==0 branch
    for tau in (0.0, 2.0, 30.0):
        assert B(0.0, tau) == tau
    for z in (-1.0, 0.0, 0.5, 3.0):
        assert B(z, 0.0) == 0.0
    assert B(0.0, 2.0) == 2.0          # does NOT raise ZeroDivisionError
    assert B(0.5, 3.0) == -math.expm1(-1.5) / 0.5 == pytest.approx(1.5537396797031404)


# CURVE-02: bond <-> forward round-trip --------------------------------------

def test_bond_from_forward_round_trip():
    curves = {
        "flat": (FlatCurve(0.03), 1e-12),
        "quad": (QuadraticForwardCurve(0.03, 0.005), 1e-12),
        "ns": (NelsonSiegelCurve(0.03, -0.01, 0.02, 1.5), 1e-10),
    }
    for _name, (c, rel) in curves.items():
        for T in (0.5, 2.0, 7.0, 30.0):
            assert bond_from_forward(c, T, 64) == pytest.approx(c.discount(T), rel=rel)
        assert c.inst_forward(0.0) == pytest.approx(c.zero_yield(0.0), rel=1e-12)


def test_quadratic_golden():
    c = QuadraticForwardCurve(0.03, 0.005)
    assert c.discount(2) == pytest.approx(math.exp(-0.07), rel=1e-12)
    assert c.discount(7) == pytest.approx(math.exp(-0.3325), rel=1e-12)


# CURVE-03: FD forward is O(h^2), not 1e-12 ----------------------------------

def test_fd_forward_matches_analytic_order_h2():
    c = NelsonSiegelCurve(0.03, -0.01, 0.02, 1.5)
    for T in (0.25, 1.0, 2.0, 5.0, 10.0, 30.0):
        fd = fd_inst_forward(c.discount, T, h=1e-5)
        assert fd == pytest.approx(c.inst_forward(T), abs=1e-6)
    # a 1e-12 band would be wrong: at least one maturity exceeds it at h=1e-5
    worst = max(abs(fd_inst_forward(c.discount, T, 1e-5) - c.inst_forward(T))
                for T in (0.25, 1, 2, 5, 10, 30))
    assert worst > 1e-12
    # O(h^2) scaling: error roughly /100 from h=1e-3 to h=1e-4
    e3 = abs(fd_inst_forward(c.discount, 2.0, 1e-3) - c.inst_forward(2.0))
    e4 = abs(fd_inst_forward(c.discount, 2.0, 1e-4) - c.inst_forward(2.0))
    assert e3 / e4 == pytest.approx(100.0, rel=0.3)
