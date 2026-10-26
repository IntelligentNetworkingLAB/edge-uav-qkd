"""SIMPLE performance comparison (no MILP, no optimization jargon).

Story in one line:
    Two ground-station pairs sit far apart. A FIXED node can only sit next to ONE
    of them, so the other pair gets almost no key. A MOVING UAV flies back and
    forth and serves BOTH.

We just simulate both and draw a bar chart of "key delivered to each pair".

Run:  python -m uavqkd.simple_compare
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .scenario import coverage_scenario

OUT = os.path.join(os.path.dirname(__file__), "..", "out")


def run(horizon=18, speed=3000.0, grid_n=9):
    os.makedirs(OUT, exist_ok=True)
    s = coverage_scenario(horizon=horizon, speed=speed, grid_n=grid_n)
    T = s.horizon_slots

    # key rate (Mbit per time-step) the UAV/node gets for each pair, at every cell
    R = s.link_key_rates()                                   # (cells, stations)
    rateA = np.minimum(R[:, 0], R[:, 1]) / 1e6               # west pair (Mbit/step)
    rateB = np.minimum(R[:, 2], R[:, 3]) / 1e6               # east pair (Mbit/step)

    sweetA = int(np.argmax(rateA))                           # best spot for west pair
    sweetB = int(np.argmax(rateB))                           # best spot for east pair

    # transit time (steps) to fly between the two sweet spots
    dist = np.linalg.norm(s.cells_xy[sweetA] - s.cells_xy[sweetB])
    hop = int(np.ceil(dist / (speed * s.slot_seconds)))

    # --- FIXED node: sits at the best single spot the whole time ---
    # it parks next to the west pair -> west gets everything, east gets ~nothing
    fixed_A = T * rateA[sweetA]
    fixed_B = T * rateB[sweetA]                              # east rate from west spot (~0)

    # --- MOVING UAV: split its time between the two spots (minus flying time) ---
    serve = max(T - hop, 2)
    nA = serve // 2
    nB = serve - nA
    move_A = nA * rateA[sweetA]
    move_B = nB * rateB[sweetB]

    # Single summary number = key to the worst-served pair (the bottleneck).
    # NOTE: a plain average would be misleading -- a fixed node dumps everything on
    # the near pair, so its average looks fine while the far pair gets ~0 (a dead
    # link). The worst pair is the honest one-number summary of coverage.
    fixed_worst = min(fixed_A, fixed_B)
    move_worst = min(move_A, move_B)

    print("=== Secret key delivered over the mission (Mbit) ===")
    print(f"{'Fixed node':<14} worst-served pair = {fixed_worst:6.1f} Mbit")
    print(f"{'Moving UAV':<14} worst-served pair = {move_worst:6.1f} Mbit")
    print()
    print(f"Moving UAV gives the worst-off pair {move_worst:.1f} Mbit, "
          f"vs only {fixed_worst:.1f} Mbit for the fixed node "
          f"({move_worst/max(fixed_worst,0.1):.0f}x better coverage).")

    _plot(fixed_worst, move_worst, os.path.join(OUT, "simple_compare.png"))
    print(f"\nfigure -> {os.path.join(os.path.abspath(OUT), 'simple_compare.png')}")
    return dict(fixed_worst=fixed_worst, move_worst=move_worst)


def _plot(fixed_worst, move_worst, path):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    bars = ax.bar(["Fixed node", "Moving UAV"], [fixed_worst, move_worst],
                  color=["#888", "#1f77b4"], width=0.5)
    ax.set_ylabel("secret key to the worst-served pair (Mbit)")
    ax.set_title("Moving UAV covers both pairs; fixed node does not")
    for b, v in zip(bars, [fixed_worst, move_worst]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.3, f"{v:.1f}",
                ha="center", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
