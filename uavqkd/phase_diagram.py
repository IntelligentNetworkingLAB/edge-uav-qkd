"""Phase diagram: the single-UAV coverage boundary over (demand x horizon).

For each (per-pair demand, flight horizon T) we solve the coverage MILP and record
the minimum capture risk a single mobile UAV needs to meet the demand -- or mark the
cell INFEASIBLE (the UAV physically cannot shuttle enough key in T slots). The
feasibility frontier is the phase boundary: below/left a single UAV suffices; above/
right you must add a second node (or a second UAV).

Run:  python -m uavqkd.phase_diagram
"""
from __future__ import annotations

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .scenario import coverage_scenario
from .solver import solve_milp_demand, two_static_nodes_min_risk

OUT = os.path.join(os.path.dirname(__file__), "..", "out")


def run(cluster_sep=8000.0, speed=3000.0, grid_n=7,
        demands_mbit=(4, 6, 8, 10, 12, 14, 16),
        horizons=(10, 12, 14, 16, 18, 20, 22),
        threat_intensity=1.0):
    os.makedirs(OUT, exist_ok=True)
    risk_grid = np.full((len(horizons), len(demands_mbit)), np.nan)
    two_needed = np.zeros_like(risk_grid, dtype=bool)
    rows = []

    for i, T in enumerate(horizons):
        s = coverage_scenario(cluster_sep=cluster_sep, threat_intensity=threat_intensity,
                              horizon=T, speed=speed, grid_n=grid_n)
        line = []
        for j, Dm in enumerate(demands_mbit):
            D = Dm * 1e6
            m = solve_milp_demand(s, demand=D)
            two_risk, _, two_feas = two_static_nodes_min_risk(s, demand=D)
            if m is not None:
                risk_grid[i, j] = m.total_risk
                tag = f"{m.total_risk:5.2f}"
            else:
                two_needed[i, j] = two_feas      # static pair could still do it
                tag = "  inf"
            rows.append(dict(horizon=T, demand_mbit=Dm,
                             mobile_risk=(m.total_risk if m else np.nan),
                             two_static_risk=(two_risk if two_feas else np.nan)))
            line.append(tag)
        print(f"T={T:>3} | " + "  ".join(line), flush=True)

    _plot(risk_grid, two_needed, demands_mbit, horizons,
          os.path.join(OUT, "phase_diagram.png"))
    with open(os.path.join(OUT, "phase.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"figures -> {os.path.abspath(OUT)}")
    return risk_grid


def _plot(risk_grid, two_needed, demands, horizons, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    masked = np.ma.masked_invalid(risk_grid)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#dddddd")                       # infeasible cells -> grey
    im = ax.pcolormesh(masked, cmap=cmap, edgecolors="white", linewidth=0.5)
    fig.colorbar(im, ax=ax, label="min capture risk (1 mobile UAV)")

    # annotate cells
    for i in range(len(horizons)):
        for j in range(len(demands)):
            if np.isnan(risk_grid[i, j]):
                txt = "2-node" if two_needed[i, j] else "✗"
                ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center",
                        fontsize=7, color="#b00")
            else:
                ax.text(j + 0.5, i + 0.5, f"{risk_grid[i, j]:.1f}",
                        ha="center", va="center", fontsize=7, color="white")

    ax.set_xticks(np.arange(len(demands)) + 0.5)
    ax.set_xticklabels(demands)
    ax.set_yticks(np.arange(len(horizons)) + 0.5)
    ax.set_yticklabels(horizons)
    ax.set_xlabel("per-pair key demand (Mbit)")
    ax.set_ylabel("flight horizon T (slots)")
    ax.set_title("Single-UAV coverage phase diagram\n(grey/✗ = infeasible → need a 2nd node)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
