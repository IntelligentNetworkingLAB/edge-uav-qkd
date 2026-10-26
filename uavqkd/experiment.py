"""Main experiment: the coverage contribution + its risk boundary.

Setup: two ground-station pairs whose sweet-spots are far apart, with a per-pair key
*demand* and an adversary over the eastern pair. The deliverable:

  * a single STATIC trusted node cannot serve both distant pairs at all (infeasible);
  * ONE mobile UAV trusted node shuttles between them and meets both demands, at a
    capture-risk cost that grows with the demand -- the key-rate <-> capture-risk
    boundary;
  * the natural deployment alternative, TWO static nodes (one per pair), is feasible
    but needs two devices -- so the figure is "1 mobile UAV vs 2 fixed nodes".

Run:  python -m uavqkd.experiment
"""
from __future__ import annotations

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .scenario import Scenario, coverage_scenario
from .solver import (
    solve_milp_demand,
    static_demand_min_risk,
    two_static_nodes_min_risk,
)
from .trust import risk_field

OUT = os.path.join(os.path.dirname(__file__), "..", "out")


def run(cluster_sep=8000.0, horizon=18, speed=3000.0, grid_n=9,
        demands_mbit=(4, 6, 8, 10, 12, 14), threat_intensity=1.0):
    os.makedirs(OUT, exist_ok=True)
    s = coverage_scenario(cluster_sep=cluster_sep, threat_intensity=threat_intensity,
                          horizon=horizon, speed=speed, grid_n=grid_n)

    rows = []
    traj_for_plot = None
    for Dm in demands_mbit:
        D = Dm * 1e6
        m = solve_milp_demand(s, demand=D)                 # exact
        lp = solve_milp_demand(s, demand=D, relax=True)    # LP lower bound
        one_risk, _, one_feas = static_demand_min_risk(s, demand=D)
        two_risk, _, two_feas = two_static_nodes_min_risk(s, demand=D)
        lp_bound = lp.total_risk if lp else np.nan
        gap = (100.0 * (m.total_risk - lp_bound) / m.total_risk
               if (m and lp and m.total_risk > 0) else np.nan)
        rows.append(
            dict(
                demand_mbit=Dm,
                mobile_risk=(m.total_risk if m else np.nan),
                lp_lower_bound=lp_bound,
                opt_gap_pct=gap,
                mobile_key_mbit=(m.total_key / 1e6 if m else np.nan),
                mobile_survival=(m.survival if m else np.nan),
                one_static_risk=(one_risk if one_feas else np.nan),
                two_static_risk=(two_risk if two_feas else np.nan),
            )
        )
        if m and Dm == demands_mbit[len(demands_mbit) // 2]:
            traj_for_plot = m
        fb_one = f"{one_risk:.2f}" if one_feas else "INFEASIBLE"
        fb_two = f"{two_risk:.2f}" if two_feas else "INFEASIBLE"
        print(f"D={Dm:>3}Mb/pair | mobile risk="
              f"{(m.total_risk if m else float('nan')):6.2f} (LP lb {lp_bound:5.2f}, "
              f"gap {gap:4.1f}%) | 1-static {fb_one:>10} | 2-static {fb_two:>10}",
              flush=True)

    _plot_boundary(rows, os.path.join(OUT, "coverage_boundary.png"))
    if traj_for_plot is not None:
        _plot_trajectory(s, traj_for_plot, os.path.join(OUT, "trajectory.png"))

    with open(os.path.join(OUT, "coverage.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    feas = [r for r in rows if not np.isnan(r["mobile_risk"])]
    if feas:
        dmax = max(r["demand_mbit"] for r in feas)
        print(f"\nmobile UAV feasible up to {dmax} Mbit/pair on T={horizon};"
              f" single static node infeasible at every demand.")
    print(f"figures -> {os.path.abspath(OUT)}")
    return rows


def _plot_boundary(rows, path):
    d = [r["demand_mbit"] for r in rows]
    mob = [r["mobile_risk"] for r in rows]
    lb = [r["lp_lower_bound"] for r in rows]
    two = [r["two_static_risk"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    # LP lower bound -> certified near-optimal band under the MILP curve
    ax.fill_between(d, lb, mob, color="#1f77b4", alpha=0.15,
                    label="MILP–LP gap (≤ optimality gap)")
    ax.plot(d, lb, ":", color="#1f77b4", lw=1, alpha=0.7)
    ax.plot(d, mob, "o-", color="#1f77b4", label="1 mobile UAV (MILP, exact)")
    ax.plot(d, two, "s--", color="#2ca02c", label="2 static nodes (one per pair)")
    # single static node is infeasible across the board
    ax.axhspan(0, max([m for m in mob if not np.isnan(m)] + [1]) * 1.1,
               xmin=0, xmax=0, alpha=0)  # keep axes sane
    ax.text(0.04, 0.92, "1 static node: infeasible at every demand\n"
            "(cannot cover both distant pairs)",
            transform=ax.transAxes, fontsize=8.5, color="#d62728",
            va="top")
    ax.set_xlabel("per-pair key demand (Mbit)")
    ax.set_ylabel("cumulative capture risk to meet demand")
    ax.set_title("Coverage boundary: 1 mobile UAV vs static placement")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_trajectory(s: Scenario, traj, path):
    xs = np.linspace(*s.x_range, 140)
    ys = np.linspace(*s.y_range, 140)
    gx, gy = np.meshgrid(xs, ys)
    field = risk_field(np.dstack([gx, gy]), s.trust)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    im = ax.contourf(gx, gy, field, levels=20, cmap="Reds", alpha=0.85)
    fig.colorbar(im, ax=ax, label="capture risk", shrink=0.85)
    gs = s.ground_stations
    ax.scatter(gs[:, 0], gs[:, 1], marker="^", s=150, c="navy",
               edgecolors="white", zorder=5, label="ground station")
    p = s.cells_xy[traj.cells]
    ax.plot(p[:, 0], p[:, 1], "-o", c="#1f77b4", ms=4, lw=1.6,
            zorder=6, label="UAV trajectory")
    ax.scatter(p[0, 0], p[0, 1], marker="*", s=240, c="lime",
               edgecolors="black", zorder=7, label="launch base")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Shuttle trajectory  (key={traj.total_key/1e6:.0f} Mbit, "
                 f"risk={traj.total_risk:.2f})")
    ax.legend(loc="upper center", ncol=3, fontsize=7.5)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
