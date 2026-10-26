"""Scenario definition: ground stations, threat field, airspace grid, UAV kinematics.

A scenario is everything the solver needs that is *not* a decision variable. The
UAV flies at a fixed altitude over a 2D grid of candidate cells; each slot it sits
in one cell and serves one ground-station pair as a trusted-node relay.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from itertools import combinations

from .channel import ChannelParams, secret_key_rate
from .trust import TrustParams, ThreatSource, risk_field


@dataclass
class Scenario:
    # ground stations (x, y, z) in metres
    ground_stations: np.ndarray
    # airspace grid
    x_range: tuple[float, float] = (-3000.0, 3000.0)
    y_range: tuple[float, float] = (-3000.0, 3000.0)
    grid_n: int = 13                     # grid is grid_n x grid_n cells
    uav_altitude_m: float = 1500.0
    # kinematics / horizon
    slot_seconds: float = 1.0            # key integrated per slot uses bits/s * slot
    horizon_slots: int = 20
    max_speed_mps: float = 1500.0        # bound on per-slot displacement
    start_cell: int | None = None        # grid index; None -> from launch_xy / centre
    launch_xy: tuple[float, float] | None = None   # UAV base; None -> grid centre
    # sub-models
    channel: ChannelParams = field(default_factory=ChannelParams)
    trust: TrustParams = field(default_factory=TrustParams)

    # -- derived (filled by __post_init__) --
    cells_xy: np.ndarray = field(init=False)      # (Ncells, 2) horizontal positions
    pairs: list[tuple[int, int]] = field(init=False)

    def __post_init__(self):
        xs = np.linspace(*self.x_range, self.grid_n)
        ys = np.linspace(*self.y_range, self.grid_n)
        gx, gy = np.meshgrid(xs, ys)
        self.cells_xy = np.column_stack([gx.ravel(), gy.ravel()])
        n_gs = len(self.ground_stations)
        self.pairs = list(combinations(range(n_gs), 2))
        if self.start_cell is None:
            anchor = np.asarray(self.launch_xy if self.launch_xy is not None
                                else (0.0, 0.0))
            d = np.linalg.norm(self.cells_xy - anchor, axis=1)
            self.start_cell = int(np.argmin(d))

    @property
    def n_cells(self) -> int:
        return self.cells_xy.shape[0]

    def cell_pos3d(self) -> np.ndarray:
        """3D UAV position for each cell (Ncells, 3) at the flight altitude."""
        z = np.full((self.n_cells, 1), self.uav_altitude_m)
        return np.hstack([self.cells_xy, z])

    def link_distances(self) -> np.ndarray:
        """Slant distance UAV-cell -> ground station, shape (Ncells, n_gs)."""
        uav = self.cell_pos3d()[:, None, :]              # (Nc, 1, 3)
        gs = self.ground_stations[None, :, :]            # (1, Ngs, 3)
        return np.linalg.norm(uav - gs, axis=2)

    def link_key_rates(self) -> np.ndarray:
        """Per-link secret key rate bits/s, shape (Ncells, n_gs)."""
        return secret_key_rate(self.link_distances(), self.channel)

    def cell_keybits(self) -> tuple[np.ndarray, np.ndarray]:
        """Best achievable key bits per slot at each cell, and the chosen pair index.

        A trusted-node relay for pair (a,b) is bottlenecked by the weaker link, so
        the per-slot key for a cell is max over pairs of min(R_a, R_b) * slot_seconds.
        Returns (keybits[Ncells], best_pair_idx[Ncells]).
        """
        R = self.link_key_rates()                        # (Nc, Ngs)
        pair_rate = np.stack(
            [np.minimum(R[:, a], R[:, b]) for (a, b) in self.pairs], axis=1
        )                                                # (Nc, Npairs)
        best_pair = np.argmax(pair_rate, axis=1)
        keybits = pair_rate[np.arange(self.n_cells), best_pair] * self.slot_seconds
        return keybits, best_pair

    def cell_risk(self) -> np.ndarray:
        """Capture risk at each cell (Ncells,)."""
        return risk_field(self.cells_xy, self.trust)

    def reachable(self) -> np.ndarray:
        """Boolean (Ncells, Ncells): can the UAV move cell i -> j in one slot?"""
        d = np.linalg.norm(
            self.cells_xy[:, None, :] - self.cells_xy[None, :, :], axis=2
        )
        return d <= self.max_speed_mps * self.slot_seconds + 1e-6


def default_scenario(threat_intensity: float = 1.0, gs_separation: float = 4000.0) -> Scenario:
    """A canonical 3-ground-station scenario with one adversary region between two
    of the stations — so the key-rich corridor overlaps the high-risk zone, which
    is exactly what makes the key-rate vs capture-risk trade-off non-trivial."""
    s = gs_separation / 2.0
    gs = np.array(
        [
            [-s, -s, 0.0],
            [+s, -s, 0.0],
            [0.0, +s, 0.0],
        ]
    )
    trust = TrustParams(
        baseline=0.05,
        sources=(
            # adversary sits on the midline between the two southern stations,
            # i.e. on the most key-efficient hovering spot.
            ThreatSource(xy=(0.0, -s), intensity=1.5, sigma_m=1200.0),
        ),
        hazard_per_slot=1.0,
        trust_decay=1.0,
    )
    # apply threat intensity scaling
    trust = TrustParams(
        baseline=trust.baseline,
        sources=tuple(
            ThreatSource(xy=src.xy, intensity=src.intensity * threat_intensity,
                         sigma_m=src.sigma_m)
            for src in trust.sources
        ),
        hazard_per_slot=trust.hazard_per_slot,
        trust_decay=trust.trust_decay,
    )
    # UAV launches from a safe base in the low-risk north, far from the southern
    # adversary, then must decide how far to venture into the contested corridor.
    return Scenario(ground_stations=gs, trust=trust, launch_xy=(0.0, +2800.0))


def coverage_scenario(cluster_sep: float = 10000.0, threat_intensity: float = 1.0,
                      horizon: int = 22, speed: float = 2800.0,
                      grid_n: int = 11) -> Scenario:
    """COVERAGE regime: two ground-station pairs whose sweet-spots are far apart, so
    no single static location serves both well. With a per-pair key *demand*, a
    static node is starved on the distant pair while a mobile UAV shuttles between
    them. One pair (east) sits under an adversary, adding the risk trade-off."""
    h = cluster_sep / 2.0
    gs = np.array(
        [
            [-h, -900.0, 0.0], [-h, +900.0, 0.0],   # west pair (0,1)
            [+h, -900.0, 0.0], [+h, +900.0, 0.0],   # east pair (2,3)
        ]
    )
    trust = TrustParams(
        baseline=0.05,
        sources=(ThreatSource(xy=(+h, 0.0), intensity=1.5 * threat_intensity,
                              sigma_m=1800.0),),
        hazard_per_slot=1.0,
        trust_decay=1.0,
    )
    margin = 1500.0
    sc = Scenario(
        ground_stations=gs,
        x_range=(-h - margin, +h + margin),
        y_range=(-3000.0, 4000.0),
        grid_n=grid_n,
        uav_altitude_m=1200.0,
        horizon_slots=horizon,
        max_speed_mps=speed,
        launch_xy=(0.0, +3500.0),                  # safe base, north-centre
        trust=trust,
    )
    # keep only the two within-cluster pairs as serviceable demands
    sc.pairs = [(0, 1), (2, 3)]
    return sc
