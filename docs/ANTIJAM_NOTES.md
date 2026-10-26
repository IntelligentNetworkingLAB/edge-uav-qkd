# Anti-jamming: RL vs robust-DRO — notes

## The story (one paragraph)
Across a ladder of jammers ordered by adaptivity, **learning (tabular Q-learning)
dominates when the jammer is predictable, but a *co-adaptive* jammer that learns the
Tx's policy drives the learner's goodput BELOW blind random — and only
distributionally-robust hedging holds up across the whole range.** The proposed
`RobustDRO` (robust scores + ACK-driven hedging temperature) is the only controller
strong at both ends.

## Three figures (`out/`)
- `antijam_wingrid.png` — controllers × 5 jammers, winner boxed per column.
  Winners: fixed→DRO 1.00, sweep→QL .97, slow→QL .91, reactive→RandomHop 1.00,
  predictive→RobustDRO .88 (QL collapses to .62).
- `antijam_crossover.png` — continuous adaptivity α (sweep→predictive). QL falls
  .97→.60; RobustDRO flat ~.875. **Crossover at α ≈ 0.21.**
- `antijam_learning_curve.png` — QL vs sweep climbs (.92→.98); QL vs predictive
  *decays* (.71→.60) as it converges and becomes predictable; RobustDRO flat.

## Why it's not "obvious"
A learner doing WORSE than blind random is counter-intuitive — it happens because a
converged Q-policy is low-entropy, and a co-adaptive adversary exploits exactly that.
This is the QKD/DRO-flavoured point: against adaptive adversaries, *robustness/
randomization beats optimization-or-learning*.

## Honest caveats
- Tabular Q-learning, not deep RL. A stronger learner pushes the crossover right but
  the collapse persists against a sufficiently co-adaptive predictor (any low-entropy
  policy is exploitable). M=8, single Tx-Rx, goodput-only utility.
- The predictive jammer is a simple Markov P(next|last) predictor; a smarter one
  shifts the boundary left.
- RobustDRO never *wins big* — its value is never *losing big* (flat ~0.875).

## Next steps
- **Entropy-regularized / max-entropy RL**: does forcing the learner to stay random
  recover robustness? This directly *bridges RL and DRO* and could be a 2nd result
  ("the fix for an exploited learner is to make it more like a robust hedger").
- Add the Lyapunov power + finite-queue layer (utility = -backlog, not just goodput).
- Non-stationary / switching jammers (adaptivity changes mid-run) — the ACK-trigger
  re-adaptation angle.
