# hjm-lab

**The Heath-Jarrow-Morton forward-rate framework from first principles** in
pure Python + NumPy/SciPy.  The no-arbitrage drift condition, Gaussian-HJM bond
prices and ZCB options, a Musiela forward-curve Monte Carlo, and the
reproductions of Ho-Lee / Hull-White / G2++ from the volatility structure —
every formula derived from its definition and pinned by an algebraic identity,
**no rates library underneath**.

The capstone of the portfolio's rates pillar:
[`tvm-lab`](https://github.com/sauloduttra/tvm-lab) →
[`fixedalt-lab`](https://github.com/sauloduttra/fixedalt-lab) →
[`shortrate-lab`](https://github.com/sauloduttra/shortrate-lab) →
[`g2pp-lab`](https://github.com/sauloduttra/g2pp-lab) → **HJM**.

**35/35 tests** verifying algebraic identities.

```
$ pytest tests/
=========================== 35 passed in 46s ============================
```

> Every mathematical identity in this repo was **adversarially verified by a
> multi-agent workflow before a line of code was written**.  18 identities
> were independently designed (three angles), synthesized, then attacked by
> one refutation agent each — **6 of 18 had a subtle error corrected**.
> Examples: the no-arbitrage residual is `½·Σₖ Sₖ²` (the **sum of squares** of
> the bond-vol vector), not `½·(Σₖ Sₖ)²` — the square-of-sum injects spurious
> cross terms for two or more factors (invisible in every one-factor case); the
> Ho-Lee bond price **must carry the realized short rate r(t)** (the state-free
> form errs by up to ~13% for t>0); the at-the-money ZCB-option time value is
> **O(σ), not O(σ²)**; and the shared primitive `B(z,τ)` needs an explicit
> `z==0` branch and a rtol-1e-13 (not bitwise) defining identity.  See "How
> this was built".

## Why this exists

The HJM framework is the deepest idea in interest-rate modelling — *no-arbitrage
forces the drift of the forward-rate curve to equal its volatility times its own
integral* — and it is the one most courses state without proving.  This repo is
the opposite: the drift condition is a function you can read, and every test
pins an **algebraic identity** rather than matching a library.

Its special cases **reproduce** the sibling short-rate models, recomputed from
scratch (there is no `import shortrate`/`import g2pp` anywhere):

- constant vol → **Ho-Lee**;
- exponential vol `σe^{−a(T−t)}` → **Hull-White / Vasicek** — the Gaussian-HJM
  ZCB-option volatility equals Vasicek's `option_sigma_p` (the flagship,
  agreeing three independent ways to ~1e-14);
- two exponential factors → **G2++** `V(τ) = Var_Q[∫₀^τ r]`.

## The 6 modules

| Module | Topic | Headline API |
|---|---|---|
| `curve.py` | initial curve P^M(0,T) + the shared `B(z,τ)` primitive | `B`, `FlatCurve`, `QuadraticForwardCurve`, `NelsonSiegelCurve`, `bond_from_forward` |
| `vol.py` | deterministic vol structures σ(t,T) and S=∫σ | `ConstantVol`, `ExponentialVol`, `MultiFactorVol` |
| `drift.py` | the no-arbitrage drift + deflated-bond martingale | `hjm_drift`, `bond_vol`, `noarb_residual`, `alpha_quadrature` |
| `gaussian.py` | Gaussian-HJM bond price, ZCB option, convexity | `gaussian_bond_price`, `option_variance`, `zcb_call`, `zcb_put` |
| `models.py` | named cases + the cross-model reproductions | `ho_lee`, `hull_white`, `reproduce_vasicek_sigma_p`, `reproduce_g2pp_V` |
| `simulate.py` | Musiela forward-curve Monte Carlo under Q | `mc_discount_factor`, `mc_forward_bond`, `mc_zcb_option` |

## API in 30 seconds

```python
import numpy as np
from hjm import (FlatCurve, ExponentialVol, hjm_drift, noarb_residual,
                 option_variance, zcb_call, reproduce_vasicek_sigma_p,
                 mc_discount_factor)

vol, curve = ExponentialVol(0.01, 0.1), FlatCurve(0.03)   # Hull-White vol

# the no-arbitrage drift, and that it makes the deflated bond a martingale
hjm_drift(vol, 0.0, 5.0)               # 2.3865121854e-04
noarb_residual(vol, 0.0, 5.0)          # ~0

# Gaussian-HJM ZCB option == Vasicek's closed form (flagship)
import math
math.sqrt(option_variance(ExponentialVol(0.013, 0.18), 2, 5))   # 0.03598166747...
reproduce_vasicek_sigma_p(0.013, 0.18, 2, 5)                    # the same, from scratch

# the forward-curve MC reprices the discount curve (returns (mean, se))
mean, se = mc_discount_factor(vol, curve, 3.0, 200, 60000, np.random.default_rng(0))
# mean ~ exp(-0.09)
```

## The algebraic identities we actually test

35 tests, every one an identity the formula must satisfy. Highlights:

### Drift & no-arbitrage
- **The drift condition** α(t,T) = Σₖ σₖ(t,T)∫ₜᵀσₖ, checked against an independent Simpson quadrature (not against its own S — that would be a tautology)
- **Drift == no-arbitrage**: ∫ₜᵀα = ½Σₖ Sₖ² (the deflated bond is a Q-martingale), with a guard pinning that the **square-of-sum is wrong** for ≥2 factors
- bond-price volatility Σ(t,T) = −∫ₜᵀσ; constant → −σ(T−t), exponential → −σB(a,T−t)

### Gaussian-HJM
- **Curve consistency** P_HJM(0,T) = P^M(0,T) exactly (A(0,T)=0 structurally)
- option variance is a B-product (stable to a=1e-9); the convexity self-block is a catastrophic difference (Taylor branch mandatory below aτ<1e-2)
- **Put-call parity** from independent Black forms; σ→0 deterministic & intrinsic limits (ATM time value is O(σ))

### Reproductions
- **Hull-White ZCB-option vol == Vasicek `option_sigma_p`**, three independent ways (flagship), `0.03598166747...`
- **Ho-Lee bond carries r(t)** (the state-free form errs >1% for t>0)
- **two-factor V == G2++** `V(τ)` (self + cross blocks), vs an Ito-isometry quadrature
- a static `ast` guard that no `shortrate`/`g2pp` import exists

### Monte Carlo (honest 4-sigma bands)
- `E_Q[exp(−∫₀ᵀr)] = P^M(0,T)` and `E^{Q^T}[P(T,S)] = P^M(0,S)/P^M(0,T)`
- MC option price reproduces the Gaussian closed form; a seed-averaged bias probe confirms no systematic bias (and the O(dt) Euler bias halving with dt at higher vol)

## Worked example

```bash
PYTHONPATH=. python examples/drift_condition.py             # sum-of-squares, not square-of-sum
PYTHONPATH=. python examples/reproduce_short_rate_models.py # HW->Vasicek, two-factor->G2++
PYTHONPATH=. python examples/forward_curve_mc.py            # the MC reprices the curve
```

`forward_curve_mc.py` output:

```
   T |      MC mean |     P^M(0,T) |         SE |       z
 1.0 |     0.970461 |     0.970446 |   2.20e-05 |    0.70
 3.0 |     0.914005 |     0.913931 |   1.00e-04 |    0.73
 5.0 |     0.860855 |     0.860708 |   1.90e-04 |    0.77
```

## How this was built

This repo was built with an **adversarial multi-agent workflow** before
implementation:

1. **Design panel** — three independent agents proposed the module layout and
   the algebraic identities from different angles (no-arbitrage theory,
   numerical, testing-first).
2. **Synthesis** — merged into one canonical spec of 18 identities.
3. **Adversarial verification** — one agent per identity tried to refute it,
   recomputing the sibling formulas from scratch in NumPy. **6 of 18 were
   corrected** before any code: the sum-of-squares-vs-square-of-sum no-arb
   residual, the state-free Ho-Lee bond, the O(σ) ATM option limit, the
   `B(z,τ)` branch, the G2++ cross-block Taylor order, and the FD-forward floor.
4. **Implementation** against the verified spec.
5. **Adversarial code review** — reviewers per dimension (math, numerical, API,
   tests), findings verified before applying.

The result: identities that are right because they were proven right, not
because they happened to pass.

## What's intentionally NOT here yet

- **v0.2.0** — caps/floors & European swaptions in the Gaussian-HJM closed form
- **v0.2.0 alt** — humped (Mercurio-Moraleda) and Hull-White-with-humped vol structures
- **v0.3.0** — the separable / Markovian-HJM reduction to a finite-state short-rate model
- **v0.3.0 alt** — the LIBOR Market Model (BGM) as the discrete-tenor cousin
- **v0.4.0** — multi-curve (OIS / forward-basis) HJM

## Related repos

- [`shortrate-lab`](https://github.com/sauloduttra/shortrate-lab) — Vasicek/CIR; HJM with exponential vol reproduces its Hull-White bond-option vol
- [`g2pp-lab`](https://github.com/sauloduttra/g2pp-lab) — the two-factor Gaussian model; HJM with two exponential factors reproduces its V(τ)
- [`fixedalt-lab`](https://github.com/sauloduttra/fixedalt-lab), [`tvm-lab`](https://github.com/sauloduttra/tvm-lab) — the deterministic-curve foundations
- [`monte-carlo-lab`](https://github.com/sauloduttra/monte-carlo-lab) — the MC simulation conventions reused here

## References

- Heath, D., Jarrow, R. & Morton, A. (1992). *Bond Pricing and the Term Structure of Interest Rates.* Econometrica 60:77–105.
- Ho, T. & Lee, S. (1986). *Term Structure Movements and Pricing Interest Rate Contingent Claims.* J. Finance 41:1011–1029.
- Hull, J. & White, A. (1990). *Pricing Interest-Rate-Derivative Securities.* RFS 3:573–592.
- Musiela, M. & Rutkowski, M. *Martingale Methods in Financial Modelling.* Springer.
- Brigo, D. & Mercurio, F. (2006). *Interest Rate Models — Theory and Practice.* Springer.

## License
MIT.
