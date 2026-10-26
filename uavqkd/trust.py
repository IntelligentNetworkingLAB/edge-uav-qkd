"""Position-dependent trust / capture-risk model for a UAV trusted node.

The novelty hook: a UAV's physical-capture exposure depends on *where it flies*.
We model a scalar risk field r(x, y) >= 0 as a baseline plus Gaussian "threat"
bumps centred on adversary locations. Trust score is a decreasing function of risk,
and per-slot capture probability follows an exponential hazard. Cumulative exposure
(sum of per-slot risk along the trajectory) is the additive quantity the solver
trades off against key rate.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ThreatSource:
    xy: tuple[float, float]   # adversary location (m)
    intensity: float          # peak risk contribution
    sigma_m: float            # spatial spread (m)


@dataclass(frozen=True)
class TrustParams:
    baseline: float = 0.05                       # ambient risk everywhere
    sources: tuple[ThreatSource, ...] = field(default_factory=tuple)
    hazard_per_slot: float = 1.0                 # scales risk -> capture prob
    trust_decay: float = 1.0                     # scales risk -> trust score


def risk_field(xy, p: TrustParams):
    """Scalar capture-risk r at position(s) `xy` of shape (..., 2)."""
    xy = np.asarray(xy, dtype=float)
    r = np.full(xy.shape[:-1], p.baseline, dtype=float)
    for s in p.sources:
        d2 = np.sum((xy - np.asarray(s.xy)) ** 2, axis=-1)
        r = r + s.intensity * np.exp(-d2 / (2.0 * s.sigma_m**2))
    return r


def trust_score(xy, p: TrustParams):
    """Trust in [0, 1]: high where risk is low."""
    return np.exp(-p.trust_decay * risk_field(xy, p))


def capture_prob_per_slot(xy, p: TrustParams):
    """Per-slot probability the UAV is captured at position `xy`."""
    return 1.0 - np.exp(-p.hazard_per_slot * risk_field(xy, p))


def survival_probability(risk_along_path, p: TrustParams):
    """Probability the UAV is never captured over a trajectory, given its per-slot
    risk values (an array). Independent-slot hazard model."""
    risk_along_path = np.asarray(risk_along_path, dtype=float)
    p_cap = 1.0 - np.exp(-p.hazard_per_slot * risk_along_path)
    return float(np.prod(1.0 - p_cap))
