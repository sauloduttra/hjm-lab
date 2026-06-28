"""The HJM no-arbitrage drift condition, and why it is a SUM OF SQUARES.

    PYTHONPATH=. python examples/drift_condition.py
"""
from __future__ import annotations

from hjm.vol import ConstantVol, ExponentialVol, MultiFactorVol
from hjm.drift import hjm_drift, alpha_quadrature, noarb_residual

print("=" * 70)
print(" HJM no-arbitrage drift  alpha(t,T) = sum_k sigma_k(t,T) int_t^T sigma_k")
print("=" * 70)
for name, vol in [("Ho-Lee  (const 1.2%)", ConstantVol(0.012)),
                  ("Hull-White (1%, a=0.1)", ExponentialVol(0.01, 0.1))]:
    a_cf = hjm_drift(vol, 0.0, 5.0)
    a_q = alpha_quadrature(vol, 0.0, 5.0, 400)
    print(f"  {name:24} alpha(0,5) = {a_cf:.10e}   "
          f"(indep. quadrature {a_q:.10e})")

print()
print(" Deflated-bond martingale:  int_0^T alpha = 1/2 ||Sigma||^2 = 1/2 sum_k S_k^2")
print(" -- the SUM OF SQUARES of the bond-vol vector, NOT the square of the sum.")
print("-" * 70)
mf = MultiFactorVol([ExponentialVol(0.011, 0.30), ExponentialVol(0.007, 0.80)])
S1 = mf.factors[0].S(0.0, 5.0)
S2 = mf.factors[1].S(0.0, 5.0)
print(f"  two factors:  S1 = {S1:.6e}   S2 = {S2:.6e}")
print(f"  no-arbitrage residual (sum of squares) = {noarb_residual(mf, 0, 5, 600):.3e}  (~0)")
print(f"  0.5*(S1^2+S2^2) = {0.5*(S1**2+S2**2):.6e}   <- correct")
print(f"  0.5*(S1+S2)^2   = {0.5*(S1+S2)**2:.6e}   <- WRONG: adds spurious cross")
print(f"  spurious cross term S1*S2 = {S1*S2:.6e}")
