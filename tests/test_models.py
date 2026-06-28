"""Cross-model reproductions (MODEL-01..03).

Each reproduction recomputes the target sibling formula FROM SCRATCH inline --
there is no `import shortrate` / `import g2pp` in hjm-lab.

Adversarial corrections:
  * MODEL-01 -- the Ho-Lee bond MUST carry the realized short rate r(t); the
    state-free form errs by up to ~13% for t>0.
  * MODEL-03 -- the two-factor V self-block horizon is the FIXED tau, not a
    generic bond maturity; the cross-block Taylor reaches ~1e-10.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from hjm.curve import B, FlatCurve
from hjm.vol import ConstantVol, ExponentialVol
from hjm.gaussian import option_variance
from hjm.models import (ho_lee_bond_price, reproduce_vasicek_sigma_p,
                        reproduce_g2pp_V, two_factor_exponential,
                        implied_short_rate_drift)


def test_implied_short_rate_drift_holee():
    """Ho-Lee implied drift theta(t) = f_t(0,t) + sigma^2 t (flat curve -> f_t=0)."""
    fc = FlatCurve(0.03)
    sigma = 0.01
    assert implied_short_rate_drift(ConstantVol(sigma), fc, 2.0) == \
        pytest.approx(sigma ** 2 * 2.0, rel=1e-6)


def _ns_fwd_slope(c, T):
    lam, e = c.lam, math.exp(-T / c.lam)
    return -c.beta1 / lam * e + c.beta2 * ((1.0 / lam) * e - (T / lam ** 2) * e)


@pytest.mark.parametrize("sigma,a,t", [(0.013, 0.18, 2.0), (0.02, 0.5, 1.0)])
def test_implied_short_rate_drift_hull_white_nonflat(sigma, a, t):
    """The Hull-White branch theta = f_t + a f0 + sigma^2/(2a)(1-e^{-2at}) on a
    curve with a genuinely nonzero forward slope."""
    from hjm.curve import NelsonSiegelCurve
    c = NelsonSiegelCurve(0.03, -0.01, 0.02, 1.5)
    expected = (_ns_fwd_slope(c, t) + a * c.inst_forward(t)
                + sigma ** 2 / (2 * a) * (1 - math.exp(-2 * a * t)))
    assert _ns_fwd_slope(c, t) != 0.0             # f_t actively exercised
    assert implied_short_rate_drift(ExponentialVol(sigma, a), c, t) == \
        pytest.approx(expected, rel=1e-6)


def test_multifactorvol_rejects_non_leaf_factors():
    from hjm.vol import MultiFactorVol
    with pytest.raises(TypeError):
        MultiFactorVol([1, 2, 3])
    inner = MultiFactorVol([ConstantVol(0.01), ExponentialVol(0.01, 0.1)])
    with pytest.raises(TypeError):
        MultiFactorVol([inner, ConstantVol(0.02)])   # no nesting


# MODEL-01: Ho-Lee bond carries the short-rate state -------------------------

def _f0(T):
    return 0.03 + 0.005 * math.sin(0.3 * T) + 0.002 * T


class _SmoothCurve:
    def inst_forward(self, T):
        return _f0(T)

    def log_discount(self, T):
        return -quad(_f0, 0, T)[0]

    def discount(self, T):
        return math.exp(self.log_discount(T))


def test_ho_lee_bond_carries_state():
    c = _SmoothCurve()
    sigma, t, T = 0.012, 0.7, 3.2
    for W in (0.0, 1.5, -2.0):
        r_t = _f0(t) + sigma ** 2 * t ** 2 / 2 + sigma * W
        f_tT = lambda u: _f0(u) + sigma ** 2 * (u * t - t ** 2 / 2) + sigma * W
        p_hjm = math.exp(-quad(f_tT, t, T)[0])
        assert ho_lee_bond_price(sigma, c, t, T, r_t) == pytest.approx(p_hjm, rel=1e-12)
        # the state-free form (drop the r(t) term) errs materially for t>0
        log_ratio = c.log_discount(T) - c.log_discount(t)
        state_free = math.exp(log_ratio - (T - t) * _f0(t)
                              - 0.5 * sigma ** 2 * t * (T - t) ** 2)
        assert abs(state_free - p_hjm) / p_hjm > 1e-2


# MODEL-02: Hull-White/Vasicek reproduction (FLAGSHIP) -----------------------

def _simpson(g, a, b, n):
    x = np.linspace(a, b, n + 1)
    y = np.array([g(float(xi)) for xi in x])
    h = (b - a) / n
    return float(h / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-2:2].sum()))


@pytest.mark.parametrize("sigma,a,T_O,T_B,golden", [
    (0.013, 0.18, 2.0, 5.0, 0.0359816674784861),
    (0.01, 0.5, 1.0, 3.0, 0.0100514766642051),
])
def test_hull_white_option_vol_reproduces_vasicek(sigma, a, T_O, T_B, golden):
    ev = ExponentialVol(sigma, a)
    v1 = math.sqrt(option_variance(ev, T_O, T_B))            # path 1: from S
    v2 = reproduce_vasicek_sigma_p(sigma, a, T_O, T_B)        # path 2: Vasicek closed form
    # path 3: raw-exponential Simpson of [Sigma(s,T_B)-Sigma(s,T_O)]^2 (no B)

    def integrand(s):
        sig_tb = -sigma / a * (math.exp(-a * (T_O - s)) - math.exp(-a * (T_B - s)))
        return sig_tb ** 2
    v3 = math.sqrt(_simpson(integrand, 0.0, T_O, 800))
    assert v1 == pytest.approx(v2, rel=1e-12)
    assert v1 == pytest.approx(v3, rel=1e-8)
    assert v1 == pytest.approx(golden, rel=1e-12)


def test_no_shortrate_import():
    """Static guard (via ast, so docstrings don't false-positive): hjm-lab
    reproduces, never imports, the siblings."""
    import ast
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent / "hjm"
    for f in root.glob("*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    assert not n.name.startswith(("shortrate", "g2pp"))
            elif isinstance(node, ast.ImportFrom):
                assert node.module is None or \
                    not node.module.startswith(("shortrate", "g2pp"))


# MODEL-03: two-factor V reproduces G2++ -------------------------------------

@pytest.mark.parametrize("sigma,a,eta,b,rho,taus,goldens", [
    (0.013, 0.18, 0.010, 0.05, -0.4, (1, 3, 7),
     (4.960274e-05, 1.110165e-03, 1.033552e-02)),
    (0.01, 0.5, 0.008, 0.1, -0.7, (1, 2, 5),
     (1.3050611078648e-05, 8.522420583158e-05, 9.709339619048e-04)),
])
def test_g2pp_V_reproduction(sigma, a, eta, b, rho, taus, goldens):
    for tau, g in zip(taus, goldens):
        v_closed = reproduce_g2pp_V(sigma, a, eta, b, rho, tau)
        assert v_closed == pytest.approx(g, rel=1e-6)
        # Ito-isometry quadrature: int_0^tau [S1^2+S2^2+2 rho S1 S2] ds
        def integ(s):
            S1 = sigma * B(a, tau - s)
            S2 = eta * B(b, tau - s)
            return S1 ** 2 + S2 ** 2 + 2 * rho * S1 * S2
        v_ito = _simpson(integ, 0.0, tau, 2000)
        assert v_closed == pytest.approx(v_ito, rel=1e-6)


def test_g2pp_V_rho_odd_decomposition():
    sigma, a, eta, b, tau = 0.013, 0.18, 0.010, 0.05, 3.0
    v_plus = reproduce_g2pp_V(sigma, a, eta, b, 0.4, tau)
    v_minus = reproduce_g2pp_V(sigma, a, eta, b, -0.4, tau)
    cross = (2 * 0.4 * sigma * eta / (a * b)
             * (tau - B(a, tau) - B(b, tau) + B(a + b, tau)))
    assert v_plus - v_minus == pytest.approx(2 * cross, rel=1e-12)
