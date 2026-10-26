"""Q-learning learning curves: it learns to BEAT a predictable jammer over time,
but a co-adaptive jammer drives its goodput back DOWN as it learns the learner.

Run:  python -m antijam.learning_curve
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .env import JammingEnv, SweepJammer, PredictiveJammer
from .controllers import QLearning, RobustDRO

OUT = os.path.join(os.path.dirname(__file__), "..", "out")


def trace(ctrl_cls, jammer_cls, M, T, seeds, win=200):
    """Return the seed-averaged rolling goodput over time."""
    acc = np.zeros(T)
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        env = JammingEnv(M=M, jammer=jammer_cls(M), seed=s)
        ctrl = ctrl_cls(); ctrl.reset(M, rng)
        acks = np.zeros(T)
        for t in range(T):
            a = ctrl.act()
            ack, jammed = env.step(a)
            ctrl.update(a, ack, jammed)
            acks[t] = ack
        # rolling mean
        c = np.cumsum(np.insert(acks, 0, 0))
        roll = (c[win:] - c[:-win]) / win
        acc[:len(roll)] += roll
    return acc[:T - win] / seeds


def run(M=8, T=6000, seeds=15):
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    series = [
        (QLearning, SweepJammer, "Q-learning vs sweep (learnable)", "#1f77b4", "-"),
        (QLearning, PredictiveJammer, "Q-learning vs predictive (co-adaptive)", "#1f77b4", "--"),
        (RobustDRO, PredictiveJammer, "RobustDRO vs predictive", "#d62728", "-"),
    ]
    for ctrl, jam, label, col, ls in series:
        y = trace(ctrl, jam, M, T, seeds)
        ax.plot(np.arange(len(y)), y, ls, color=col, lw=2, label=label)
        print(f"{label:<42} final goodput {y[-1]:.3f}", flush=True)

    ax.axhline(1 - 1 / M, color="grey", ls=":", lw=1, label="blind random (1-1/M)")
    ax.set_xlabel("time slot")
    ax.set_ylabel("rolling goodput (window 200)")
    ax.set_title("Learning helps — until the adversary learns the learner")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "antijam_learning_curve.png"), dpi=150)
    plt.close(fig)
    print(f"figure -> {os.path.abspath(OUT)}")


if __name__ == "__main__":
    run()
