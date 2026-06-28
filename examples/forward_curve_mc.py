"""Monte Carlo of the forward curve under Q reprices the discount curve.

    PYTHONPATH=. python examples/forward_curve_mc.py
"""
from __future__ import annotations

import math

import numpy as np

from hjm.curve import FlatCurve
from hjm.vol import ExponentialVol
from hjm.simulate import mc_discount_factor, discretization_bias_probe

vol = ExponentialVol(0.01, 0.1)
curve = FlatCurve(0.03)

print("=" * 66)
print(" Musiela forward-curve Monte Carlo  (Hull-White vol, flat 3% curve)")
print(" E_Q[exp(-int_0^T r du)] should reprice P^M(0,T) = exp(-0.03 T)")
print("=" * 66)
print(f"{'T':>4} | {'MC mean':>12} | {'P^M(0,T)':>12} | {'SE':>10} | {'z':>7}")
print("-" * 66)
for T in (1.0, 2.0, 3.0, 5.0):
    mean, se = mc_discount_factor(vol, curve, T, 200, 60000, np.random.default_rng(7))
    target = curve.discount(T)
    print(f"{T:>4.1f} | {mean:>12.6f} | {target:>12.6f} | {se:>10.2e} | "
          f"{(mean-target)/se:>7.2f}")

print()
print(" The HJM Euler scheme has a genuine O(dt) drift bias (no exact")
print(" transition).  Seed-averaging cancels the MC noise; at higher vol the")
print(" bias is resolvable and HALVES when dt is halved (order O(dt)):")
hi_vol = ExponentialVol(0.08, 0.1)
for n in (40, 80, 160):
    pr = discretization_bias_probe(hi_vol, curve, 5.0, n_steps=n, n_paths=15000,
                                   n_seeds=20, base_seed=1000)
    print(f"   n_steps={n:>3} : seed-avg bias = {pr['bias']:.3e}  "
          f"+/- {pr['bias_se']:.1e}")
