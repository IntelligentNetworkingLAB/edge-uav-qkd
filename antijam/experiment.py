"""Run every controller against every jammer (multi-seed) and draw the win-grid.

The deliverable: a goodput heat-map (controllers x jammer-adaptivity ladder) that
shows the boundary -- where learning (Q-learning) wins, and where robust hedging
(Uniform / RandomHop / RobustDRO) wins because a co-adaptive jammer out-learns the
learner.

Run:  python -m antijam.experiment
"""
from __future__ import annotations

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .env import JammingEnv, JAMMER_LADDER
from .controllers import ALL_CONTROLLERS

OUT = os.path.join(os.path.dirname(__file__), "..", "out")


def run_one(ctrl_cls, jammer_cls, M, T, seed):
    rng = np.random.default_rng(1000 + seed)
    env = JammingEnv(M=M, jammer=jammer_cls(M), seed=seed)
    ctrl = ctrl_cls()
    ctrl.reset(M, rng)
    good = 0
    # discard a warm-up so we measure steady-state behaviour
    warm = T // 5
    for t in range(T):
        a = ctrl.act()
        ack, jammed = env.step(a)
        ctrl.update(a, ack, jammed)
        if t >= warm:
            good += ack
    return good / (T - warm)


def run(M=8, T=4000, seeds=20):
    os.makedirs(OUT, exist_ok=True)
    jammers = JAMMER_LADDER
    ctrls = ALL_CONTROLLERS
    grid = np.zeros((len(ctrls), len(jammers)))

    for ci, c in enumerate(ctrls):
        for ji, j in enumerate(jammers):
            vals = [run_one(c, j, M, T, s) for s in range(seeds)]
            grid[ci, ji] = np.mean(vals)

    cnames = [c.name for c in ctrls]
    jnames = [j(M).name for j in jammers]

    # print table
    print(f"goodput (M={M}, T={T}, {seeds} seeds)")
    print(f"{'':<11}" + "".join(f"{n:>10}" for n in jnames))
    for ci, cn in enumerate(cnames):
        print(f"{cn:<11}" + "".join(f"{grid[ci,ji]:>10.3f}" for ji in range(len(jnames))))

    # who wins each jammer
    print("\nbest controller per jammer:")
    for ji, jn in enumerate(jnames):
        bi = int(np.argmax(grid[:, ji]))
        print(f"  {jn:<10} -> {cnames[bi]:<10} ({grid[bi,ji]:.3f})")

    _plot(grid, cnames, jnames, os.path.join(OUT, "antijam_wingrid.png"))
    with open(os.path.join(OUT, "antijam.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["controller"] + jnames)
        for ci, cn in enumerate(cnames):
            w.writerow([cn] + [f"{grid[ci,ji]:.4f}" for ji in range(len(jnames))])
    print(f"\nfigure -> {os.path.abspath(OUT)}")
    return grid


def _plot(grid, cnames, jnames, path):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="goodput")
    ax.set_xticks(range(len(jnames))); ax.set_xticklabels(jnames)
    ax.set_yticks(range(len(cnames))); ax.set_yticklabels(cnames)
    ax.set_xlabel("jammer  (less adaptive  →  more adaptive)")
    ax.set_ylabel("controller")
    ax.set_title("Anti-jamming goodput: learning vs robust hedging")
    # mark the winner of each column
    for ji in range(len(jnames)):
        bi = int(np.argmax(grid[:, ji]))
        ax.add_patch(plt.Rectangle((ji - 0.5, bi - 0.5), 1, 1, fill=False,
                                   edgecolor="black", lw=2.2))
    for ci in range(len(cnames)):
        for ji in range(len(jnames)):
            ax.text(ji, ci, f"{grid[ci,ji]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
