"""Continuous-adaptivity crossover: goodput vs jammer adaptivity alpha.

Uses BlendJammer (sweep at alpha=0 -> predictive at alpha=1) and sweeps alpha to
trace where the LEARNER (Q-learning) stops winning and ROBUST hedging takes over.
The crossing of the Q-learning and RobustDRO curves is the phase boundary.

Run:  python -m antijam.crossover
"""
from __future__ import annotations

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .env import JammingEnv, BlendJammer
from .controllers import Uniform, RandomHop, DRO, RobustDRO, QLearning

OUT = os.path.join(os.path.dirname(__file__), "..", "out")


def run_one(ctrl_cls, alpha, M, T, seed):
    rng = np.random.default_rng(1000 + seed)
    env = JammingEnv(M=M, jammer=BlendJammer(M, alpha=alpha), seed=seed)
    ctrl = ctrl_cls(); ctrl.reset(M, rng)
    good, warm = 0, T // 5
    for t in range(T):
        a = ctrl.act()
        ack, jammed = env.step(a)
        ctrl.update(a, ack, jammed)
        if t >= warm:
            good += ack
    return good / (T - warm)


def run(M=8, T=4000, seeds=20, alphas=None):
    os.makedirs(OUT, exist_ok=True)
    if alphas is None:
        alphas = np.linspace(0, 1, 11)
    ctrls = [Uniform, RandomHop, DRO, RobustDRO, QLearning]
    curves = {c.name: [] for c in ctrls}
    for a in alphas:
        for c in ctrls:
            vals = [run_one(c, a, M, T, s) for s in range(seeds)]
            curves[c.name].append(np.mean(vals))
        row = "  ".join(f"{c.name}:{curves[c.name][-1]:.3f}" for c in ctrls)
        print(f"alpha={a:.2f} | {row}", flush=True)

    _plot(alphas, curves, os.path.join(OUT, "antijam_crossover.png"))
    with open(os.path.join(OUT, "antijam_crossover.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha"] + list(curves.keys()))
        for i, a in enumerate(alphas):
            w.writerow([f"{a:.3f}"] + [f"{curves[k][i]:.4f}" for k in curves])

    # find crossover between QLearning and RobustDRO
    ql = np.array(curves["QLearning"]); rd = np.array(curves["RobustDRO"])
    diff = ql - rd
    cross = None
    for i in range(len(alphas) - 1):
        if diff[i] > 0 and diff[i + 1] <= 0:
            cross = float(np.interp(0, [diff[i + 1], diff[i]],
                                    [alphas[i + 1], alphas[i]]))
            break
    if cross is not None:
        print(f"\nQ-learning beats RobustDRO for alpha < {cross:.2f}; "
              f"robust hedging wins above it.")
    print(f"figure -> {os.path.abspath(OUT)}")
    return curves


def _plot(alphas, curves, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    styles = {"Uniform": ("#999", "-"), "RandomHop": ("#bbb", "--"),
              "DRO": ("#2ca02c", "-."), "RobustDRO": ("#d62728", "-"),
              "QLearning": ("#1f77b4", "-")}
    for name, ys in curves.items():
        col, ls = styles.get(name, ("k", "-"))
        lw = 2.4 if name in ("QLearning", "RobustDRO") else 1.3
        ax.plot(alphas, ys, ls, color=col, lw=lw, marker="o", ms=3, label=name)
    ax.set_xlabel("jammer adaptivity  α   (sweep → predictive / co-adaptive)")
    ax.set_ylabel("goodput")
    ax.set_title("Learning wins on the left; robust hedging wins on the right")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
