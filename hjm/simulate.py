"""Forward-curve Monte Carlo under Q, in Musiela coordinates.

The Musiela reparametrization r(t,x) = f(t,t+x) (x = time-to-maturity) turns
the HJM SDE into

    dr(t,x) = [ d r/dx + sigma(x) * int_0^x sigma(s) ds ] dt + sigma(x) dW(t).

The transport term d r/dx is integrated EXACTLY by an index shift (the grid
step dx equals the time step dt, so a one-step roll is r(t+dt, x_i) =
r(t, x_{i+1})), leaving only the genuine O(dt) Euler bias on the drift-coupling
term and the O(dt^2) trapezoid on the discount integral.  UNLIKE the exact-OU
sibling MCs (shortrate-lab, g2pp-lab), HJM has no exact transition; so the
martingale tests use HONEST 4-sigma bands with a reported z-score, plus a
seed-averaged `discretization_bias_probe` that cancels the MC noise to confirm
the residual O(dt) bias is consistent with zero (it is resolvable, and halves
with dt, only at higher vol).  The MC never reuses the analytic bond price or
option vol, so a wrong drift surfaces as a band breach.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "musiela_drift", "simulate_short_and_curve",
    "mc_discount_factor", "mc_forward_bond", "mc_zcb_option",
    "discretization_bias_probe",
]


def musiela_drift(vol, x: float) -> float:
    """The Musiela drift-coupling term sigma(x) * int_0^x sigma(s) ds =
    sum_k sigma_k(0,x) S_k(0,x).  Equals the maturity-form alpha(t,t+x)."""
    return sum(f.sigma(0.0, x) * f.S(0.0, x) for f in vol.factors)


def simulate_short_and_curve(vol, curve, T: float, max_ttm: float,
                             n_steps: int, n_paths: int, rng):
    """Roll the Musiela curve to time T.  Returns (disc, r_terminal, x_grid):
    the path discount exp(-int_0^T r(t,0) dt) (trapezoid) and the terminal
    forward curve r(T, x) over a time-to-maturity grid covering [0, max_ttm].

    Only the current curve is held in memory (n_paths x nx), so memory stays
    O(n_paths * (T+max_ttm)/dt)."""
    dt = T / n_steps
    X = T + max_ttm
    nx = int(round(X / dt)) + 1
    x = np.arange(nx) * dt
    factors = vol.factors
    K = len(factors)

    sigx = np.array([[f.sigma(0.0, float(xi)) for xi in x] for f in factors])  # (K, nx)
    couple = (sigx * (np.array([[f.S(0.0, float(xi)) for xi in x]
                                for f in factors]))).sum(axis=0)               # (nx,)

    f0 = np.array([curve.inst_forward(float(xi)) for xi in x])
    r = np.tile(f0, (n_paths, 1))                                              # (n_paths, nx)

    sqdt = math.sqrt(dt)
    prev_short = r[:, 0].copy()
    integral = np.zeros(n_paths)
    for _ in range(n_steps):
        new = np.empty_like(r)
        new[:, :-1] = r[:, 1:]              # exact transport (index shift)
        new[:, -1] = r[:, -1]               # flat boundary (unread region)
        new += couple * dt                  # drift coupling (broadcast over paths)
        for j in range(K):
            new += np.outer(rng.standard_normal(n_paths) * sqdt, sigx[j])
        r = new
        cur_short = r[:, 0]
        integral += 0.5 * (prev_short + cur_short) * dt
        prev_short = cur_short

    disc = np.exp(-integral)
    return disc, r, x


def mc_discount_factor(vol, curve, T: float, n_steps: int, n_paths: int,
                       rng) -> tuple:
    """Monte Carlo E_Q[exp(-int_0^T r du)] = P^M(0,T).  Returns (mean, se)."""
    disc, _, _ = simulate_short_and_curve(vol, curve, T, 0.0, n_steps, n_paths, rng)
    return float(disc.mean()), float(disc.std(ddof=1) / math.sqrt(n_paths))


def _bond_from_terminal_curve(r: np.ndarray, x: np.ndarray, ttm: float) -> np.ndarray:
    """P(T, T+ttm) = exp(-int_0^ttm r(T,x) dx) per path (trapezoid on the
    simulated terminal forward curve).  `ttm` must be an integer multiple of
    the grid step dx (= dt): choose n_steps so (S-T)/(T/n_steps) is integral."""
    dx = x[1] - x[0]
    ratio = ttm / dx
    n = int(round(ratio))
    if abs(ratio - n) > 1e-9:
        raise ValueError(
            f"maturity ttm={ttm!r} is not an integer multiple of the grid step "
            f"dx={dx!r} (ttm/dx={ratio!r}); choose n_steps so (S-T)/(T/n_steps) "
            "is integral")
    seg = r[:, :n + 1].copy()
    w = np.full(n + 1, dx)
    w[0] *= 0.5
    w[n] *= 0.5
    return np.exp(-(seg * w).sum(axis=1))


def mc_forward_bond(vol, curve, T: float, S: float, n_steps: int, n_paths: int,
                    rng) -> tuple:
    """Monte Carlo E^{Q^T}[P(T,S)] = P^M(0,S)/P^M(0,T), via
    E_Q[exp(-int_0^T r) P(T,S)] = P^M(0,S) divided by the deterministic
    P^M(0,T).  Returns (mean, se)."""
    disc, r, x = simulate_short_and_curve(vol, curve, T, S - T, n_steps, n_paths, rng)
    p_ts = _bond_from_terminal_curve(r, x, S - T)
    payoff = disc * p_ts
    p0t = curve.discount(T)
    return float(payoff.mean() / p0t), float(payoff.std(ddof=1) / math.sqrt(n_paths) / p0t)


def mc_zcb_option(vol, curve, T_O: float, T_B: float, K: float, n_steps: int,
                  n_paths: int, rng) -> tuple:
    """Monte Carlo E_Q[exp(-int_0^{T_O} r)(P(T_O,T_B)-K)^+] -- compare to
    gaussian.zcb_call.  Returns (mean, se)."""
    disc, r, x = simulate_short_and_curve(vol, curve, T_O, T_B - T_O,
                                          n_steps, n_paths, rng)
    p_ob = _bond_from_terminal_curve(r, x, T_B - T_O)
    payoff = disc * np.maximum(p_ob - K, 0.0)
    return float(payoff.mean()), float(payoff.std(ddof=1) / math.sqrt(n_paths))


def discretization_bias_probe(vol, curve, T: float, n_steps: int, n_paths: int,
                              n_seeds: int, base_seed: int = 0) -> dict:
    """Estimate the Euler scheme's SYSTEMATIC bias by averaging the SIGNED gap
    (mc mean - P^M(0,T)) over `n_seeds` independent runs, which cancels the
    Monte Carlo sampling noise that swamps any single run.

    Returns {bias, bias_se, n_steps}.  A correct drift condition leaves the
    seed-averaged bias consistent with zero (the O(dt) Euler bias is far below
    a single run's SE at realistic vols); a wrong drift survives the averaging
    and shows up as |bias| >> bias_se.
    """
    target = curve.discount(T)
    biases = np.array([
        mc_discount_factor(vol, curve, T, n_steps, n_paths,
                           np.random.default_rng(base_seed + s))[0] - target
        for s in range(n_seeds)])
    return {"bias": float(biases.mean()),
            "bias_se": float(biases.std(ddof=1) / math.sqrt(n_seeds)),
            "n_steps": n_steps}
