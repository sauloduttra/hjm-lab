"""The HJM framework reproduces its short-rate special cases from scratch.

    PYTHONPATH=. python examples/reproduce_short_rate_models.py
"""
from __future__ import annotations

import math

import numpy as np

from hjm.curve import B
from hjm.vol import ExponentialVol
from hjm.gaussian import option_variance
from hjm.models import reproduce_vasicek_sigma_p, reproduce_g2pp_V

print("=" * 72)
print(" Hull-White / Vasicek reproduction (FLAGSHIP)")
print(" The Gaussian-HJM ZCB-option vol == Vasicek's option_sigma_p, three ways")
print("=" * 72)
sigma, a, T_O, T_B = 0.013, 0.18, 2.0, 5.0
v1 = math.sqrt(option_variance(ExponentialVol(sigma, a), T_O, T_B))     # HJM, from S
v2 = reproduce_vasicek_sigma_p(sigma, a, T_O, T_B)                       # Vasicek closed form


def _sigma(s, T):
    return -sigma / a * (math.exp(-a * (T_O - s)) - math.exp(-a * (T_B - s)))


grid = np.linspace(0, T_O, 4001)
v3 = math.sqrt(np.trapz(np.array([_sigma(s, 0) ** 2 for s in grid]), grid))  # raw quadrature
print(f"  (1) HJM option_variance (from S) : {v1:.15f}")
print(f"  (2) Vasicek option_sigma_p       : {v2:.15f}")
print(f"  (3) raw-exponential quadrature    : {v3:.15f}")
print(f"  max abs disagreement              : {max(abs(v1-v2), abs(v1-v3)):.2e}")

print()
print("=" * 72)
print(" G2++ reproduction: two exponential factors -> V(tau) = Var_Q[int_0^tau r]")
print("=" * 72)
print(f"{'tau':>5} | {'HJM Ito-isometry':>20} | {'G2++ closed form':>20}")
print("-" * 72)
sg, a1, et, b1, rho = 0.013, 0.18, 0.010, 0.05, -0.4
for tau in (1, 3, 7):
    g = np.linspace(0, tau, 8001)
    S1 = sg * np.array([B(a1, tau - s) for s in g])
    S2 = et * np.array([B(b1, tau - s) for s in g])
    v_ito = np.trapz(S1 ** 2 + S2 ** 2 + 2 * rho * S1 * S2, g)
    v_cf = reproduce_g2pp_V(sg, a1, et, b1, rho, tau)
    print(f"{tau:>5} | {v_ito:>20.12e} | {v_cf:>20.12e}")
