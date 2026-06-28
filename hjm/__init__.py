"""hjm-lab: the Heath-Jarrow-Morton forward-rate framework, from first principles.

The no-arbitrage drift condition, Gaussian-HJM bond prices and ZCB options, and
the reproductions of Ho-Lee / Hull-White / G2++ from the volatility structure --
every formula derived from its definition and pinned by an algebraic identity,
no rates library underneath.  The capstone of the rates pillar
(tvm -> fixedalt -> shortrate -> g2pp -> HJM).
"""
from hjm import curve, drift, gaussian, models, simulate, vol

from hjm.curve import (
    B,
    FlatCurve,
    NelsonSiegelCurve,
    QuadraticForwardCurve,
    ZeroCurve,
    bond_from_forward,
    fd_inst_forward,
)
from hjm.vol import ConstantVol, ExponentialVol, MultiFactorVol, VolStructure
from hjm.drift import (
    alpha_quadrature,
    bond_drift_under_Q,
    bond_vol,
    hjm_drift,
    noarb_residual,
)
from hjm.gaussian import (
    convexity_self_block,
    gaussian_bond_price,
    integrated_rate_variance,
    option_variance,
    zcb_call,
    zcb_put,
)
from hjm.models import (
    ho_lee,
    ho_lee_bond_price,
    hull_white,
    implied_short_rate_drift,
    reproduce_g2pp_V,
    reproduce_vasicek_sigma_p,
    two_factor_exponential,
)
from hjm.simulate import (
    discretization_bias_probe,
    mc_discount_factor,
    mc_forward_bond,
    mc_zcb_option,
    musiela_drift,
    simulate_short_and_curve,
)

__version__ = "0.1.0"
