"""Controllers (channel-selection policies) compared against the jammer ladder.

All share the same interface and the same observation each slot: the ACK bit and the
sensed jammed channel. They differ in how they turn that history into the next
channel:

  Uniform     - blind uniform random (maximally unpredictable, ignores everything)
  RandomHop   - uniform but never repeat the last channel
  DRO         - Wasserstein-robust: avoid channels the jammer histogram says are hot,
                inflated by an uncertainty radius (distributionally robust)
  RobustDRO   - proposed: softmax over robust scores with an ACK-driven temperature
                (hedge harder when ACKs drop -> adapts how random to be)
  QLearning   - tabular RL: state = (last own channel, last jammed channel),
                learns Q(state, channel) online by epsilon-greedy
"""
from __future__ import annotations

import numpy as np


class Controller:
    name = "base"
    def reset(self, M, rng): self.M, self.rng = M, rng
    def act(self) -> int: raise NotImplementedError
    def update(self, action, ack, jammed): ...


class Uniform(Controller):
    name = "Uniform"
    def act(self): return int(self.rng.integers(self.M))


class RandomHop(Controller):
    name = "RandomHop"
    def reset(self, M, rng): super().reset(M, rng); self.last = None
    def act(self):
        if self.last is None:
            self.last = int(self.rng.integers(self.M)); return self.last
        choices = [c for c in range(self.M) if c != self.last]
        self.last = int(self.rng.choice(choices)); return self.last


class DRO(Controller):
    """Distributionally-robust: pick the channel maximizing worst-case goodput over a
    Wasserstein ball of radius `eps` around the empirical jammer-channel histogram."""
    name = "DRO"
    def __init__(self, eps=0.15): self.eps = eps
    def reset(self, M, rng):
        super().reset(M, rng)
        self.counts = np.ones(M)               # Laplace-smoothed jammer histogram
    def act(self):
        phat = self.counts / self.counts.sum()
        # worst-case jam prob on each channel: inflate by eps (robustness)
        worst = np.minimum(phat + self.eps, 1.0)
        goodput = 1.0 - worst                  # expected success if we pick channel c
        return int(np.argmax(goodput + 1e-9 * self.rng.standard_normal(self.M)))
    def update(self, action, ack, jammed):
        self.counts[jammed] += 1


class RobustDRO(Controller):
    """Proposed: softmax over robust goodput scores with an ACK-driven temperature.
    When recent ACK rate is high -> low temperature (exploit the safe channel);
    when ACKs collapse (an adaptive jammer is tracking us) -> high temperature
    (hedge / randomize). This is the two-timescale 'hedging degree' idea."""
    name = "RobustDRO"
    def __init__(self, eps=0.15, tau_min=0.03, tau_max=1.0):
        self.eps, self.tau_min, self.tau_max = eps, tau_min, tau_max
    def reset(self, M, rng):
        super().reset(M, rng)
        self.counts = np.ones(M)
        self.ack_ewma = 1.0
    def act(self):
        phat = self.counts / self.counts.sum()
        worst = np.minimum(phat + self.eps, 1.0)
        score = 1.0 - worst
        tau = self.tau_min + (self.tau_max - self.tau_min) * (1.0 - self.ack_ewma)
        z = score / max(tau, 1e-3)
        z -= z.max()
        p = np.exp(z); p /= p.sum()
        return int(self.rng.choice(self.M, p=p))
    def update(self, action, ack, jammed):
        self.counts[jammed] += 1
        self.ack_ewma = 0.95 * self.ack_ewma + 0.05 * ack


class QLearning(Controller):
    """Tabular Q-learning. State = (last own channel, last jammed channel); action =
    next channel; reward = ACK. Learns to predict and dodge structured jammers."""
    name = "QLearning"
    def __init__(self, alpha=0.2, gamma=0.9, eps0=0.3, eps_decay=0.9995):
        self.alpha, self.gamma, self.eps0, self.eps_decay = alpha, gamma, eps0, eps_decay
    def reset(self, M, rng):
        super().reset(M, rng)
        self.Q = np.zeros((M, M, M))           # [last_action, last_jammed, action]
        self.eps = self.eps0
        self.s = (0, 0)
        self.last_action = 0
    def act(self):
        la, lj = self.s
        if self.rng.random() < self.eps:
            a = int(self.rng.integers(self.M))
        else:
            q = self.Q[la, lj]
            a = int(np.argmax(q + 1e-9 * self.rng.standard_normal(self.M)))
        self.last_action = a
        return a
    def update(self, action, ack, jammed):
        la, lj = self.s
        s2 = (action, jammed)
        best_next = self.Q[s2[0], s2[1]].max()
        td = ack + self.gamma * best_next - self.Q[la, lj, action]
        self.Q[la, lj, action] += self.alpha * td
        self.s = s2
        self.eps = max(0.02, self.eps * self.eps_decay)


ALL_CONTROLLERS = [Uniform, RandomHop, DRO, RobustDRO, QLearning]
