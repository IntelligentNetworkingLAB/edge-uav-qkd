"""Anti-jamming channel-hopping environment + a ladder of jammers ordered by adaptivity.

One Tx-Rx link, M frequency channels. Each slot the Tx picks a channel; the jammer
picks a channel. If they collide the packet is lost (NACK), else it gets through
(ACK). The controller's utility is goodput (fraction of slots that get through).

The jammers are ordered from non-adaptive to fully co-adaptive -- this ladder is the
x-axis of the whole study: we ask which controller wins at each rung.
"""
from __future__ import annotations

import numpy as np


# ----------------------------- jammers -------------------------------------
class Jammer:
    name = "base"
    def reset(self, rng): ...
    def choose(self, t, tx_last):    # returns channel to jam this slot
        raise NotImplementedError
    def observe(self, tx_action):    # sees which channel the Tx used (sensing)
        ...


class FixedJammer(Jammer):
    """Rung 0 — jams one fixed channel forever. Trivially learnable."""
    name = "fixed"
    def __init__(self, M, ch=0): self.M, self.ch = M, ch
    def reset(self, rng): self.rng = rng
    def choose(self, t, tx_last): return self.ch
    def observe(self, a): ...


class SweepJammer(Jammer):
    """Rung 1 — sweeps channels in a fixed cycle. Deterministic, learnable."""
    name = "sweep"
    def __init__(self, M): self.M = M
    def reset(self, rng): self.rng = rng
    def choose(self, t, tx_last): return t % self.M
    def observe(self, a): ...


class SlowLearnerJammer(Jammer):
    """Rung 2 — jams the Tx's most-used channel over a sliding window (lagged)."""
    name = "slow"
    def __init__(self, M, window=40): self.M, self.W = M, window
    def reset(self, rng):
        self.rng = rng; self.hist = []
    def choose(self, t, tx_last):
        if not self.hist: return self.rng.integers(self.M)
        recent = self.hist[-self.W:]
        return int(np.bincount(recent, minlength=self.M).argmax())
    def observe(self, a): self.hist.append(a)


class ReactiveJammer(Jammer):
    """Rung 3 — jams whatever channel the Tx used last slot (1-step memory)."""
    name = "reactive"
    def __init__(self, M): self.M = M
    def reset(self, rng): self.rng = rng; self.last = None
    def choose(self, t, tx_last):
        return tx_last if tx_last is not None else self.rng.integers(self.M)
    def observe(self, a): ...


class PredictiveJammer(Jammer):
    """Rung 4 — co-adaptive: learns the Tx's transition pattern P(next | last) and
    jams the channel it predicts the Tx will hop to. Punishes any *predictable*
    policy, including a converged learner."""
    name = "predictive"
    def __init__(self, M): self.M = M
    def reset(self, rng):
        self.rng = rng
        self.count = np.ones((self.M, self.M))    # Laplace-smoothed transition counts
        self.prev = None
    def choose(self, t, tx_last):
        if tx_last is None:
            return self.rng.integers(self.M)
        return int(self.count[tx_last].argmax())     # predicted next channel
    def observe(self, a):
        if self.prev is not None:
            self.count[self.prev, a] += 1
        self.prev = a


class BlendJammer(Jammer):
    """Continuous adaptivity knob alpha in [0,1]: each slot, act PREDICTIVELY
    (co-adaptive) with probability alpha, else follow a fixed SWEEP pattern.

      alpha = 0  -> pure sweep      : predictable structure, a learner exploits it
      alpha = 1  -> pure predictive : co-adaptive, it learns the learner

    Sweeping alpha traces the phase transition from 'learning wins' to
    'robust hedging wins'."""
    name = "blend"
    def __init__(self, M, alpha=0.5):
        self.M, self.alpha = M, alpha
    def reset(self, rng):
        self.rng = rng
        self.count = np.ones((self.M, self.M))
        self.prev = None
    def choose(self, t, tx_last):
        if self.rng.random() < self.alpha:                 # predictive component
            if tx_last is None:
                return int(self.rng.integers(self.M))
            return int(self.count[tx_last].argmax())
        return t % self.M                                  # sweep component
    def observe(self, a):
        if self.prev is not None:
            self.count[self.prev, a] += 1
        self.prev = a


JAMMER_LADDER = [FixedJammer, SweepJammer, SlowLearnerJammer,
                 ReactiveJammer, PredictiveJammer]


# ----------------------------- environment ---------------------------------
class JammingEnv:
    def __init__(self, M=8, jammer: Jammer | None = None, seed=0):
        self.M = M
        self.jammer = jammer if jammer is not None else FixedJammer(M)
        self.rng = np.random.default_rng(seed)
        self.jammer.reset(np.random.default_rng(seed + 1))
        self.tx_last = None
        self.t = 0

    def step(self, action):
        """Tx transmits on `action`. Returns (ack, jammed_channel)."""
        j = self.jammer.choose(self.t, self.tx_last)
        ack = int(action != j)
        self.jammer.observe(action)       # jammer senses the Tx channel
        self.tx_last = action
        self.t += 1
        return ack, j
