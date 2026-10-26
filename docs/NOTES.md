# Research notes — UAV mobile trusted-node QKD (apnoms26)

## WHERE THE CONTRIBUTION LANDED (read this first)

We tried three framings for "mobility helps" and kept the one with a decisive,
well-motivated result:

| Framing | Result | Verdict |
|---|---|---|
| **Coverage / per-pair demand** | 1 static node *cannot* cover two distant pairs (infeasible); 1 mobile UAV meets both demands, risk grows with demand; 2 static nodes feasible but cost 2 devices | **KEPT — centerpiece** |
| Pure key↔risk time-sharing vs *idealized* static placement | marginal (−10%…+6%), transit-limited, sensitive to launch distance/horizon | demoted to honest footnote |
| Threat-intensity as a phase axis | nearly invariant (uniform risk scaling cancels) | discarded |

The decisive advantage of a mobile trusted node is **coverage of spatially-separated
demand**, not risk/key time-sharing. The honest boundary lives *inside* the coverage
story: the feasibility frontier in (demand × horizon) and the risk premium the single
UAV pays versus two fixed nodes.

## Solver / method choice (decided on performance, 9×9 grid, T=18)

Benchmarked four ways to solve the coverage min-risk problem; kept the two that won.

| Method | D=6 | D=10 | D=14 | runtime | verdict |
|---|---|---|---|---|---|
| **MILP (HIGHS, exact)** | 3.72 | 6.43 | 9.03 | 20–50 s | **KEPT — reference** |
| **LP relaxation (lower bound)** | 3.43 | 6.14 | 8.86 | ~7 s | **KEPT — certifies MILP within 2–8%** |
| Lagrangian-DP (dualize demand) | — | — | — | <0.1 s | DROPPED — multiplier convergence fails on the bi-cluster geometry (never returns a feasible integral shuttle) |
| Greedy shuttle (domain heuristic) | +124% | +62% | infeasible at D=14 | <1 ms | DROPPED — pays 60–120% more risk, brittle |

**Decision: report MILP + LP gap.** The LP relaxation is a valid lower bound, so the
MILP↔LP gap (≤ 8%, tightening to ~2% at high demand) *certifies* the trajectory is
near-optimal — a publishable claim with no extra solver. The problem is offline
trajectory *design*, so 20–50 s/instance is acceptable; GUROBI/MOSEK (swap via
`solver=`) cut this ~10×. The fast heuristics lose too much risk to be worth it.

## QKD key-rate model: hand-rolled vs library

Surveyed simulators. **TNO `tno.quantum.communication.qkd-key-rate`** (Apache-2.0,
pip, finite-key BB84, callable in a loop) is the only clean drop-in for the *rate*
kernel; SeQUeNCe/NetSquid are event simulators (too heavy for an inner loop), Qrackling
is MATLAB. **Decision: keep the hand-rolled `channel.py`** — its equations go straight
into the paper (transparent) and no library provides the FSO channel anyway. Optionally
use TNO as an independent cross-check of the key-rate numbers. `pip install
tno.quantum.communication.qkd-key-rate`.

## The mechanism (why mobility can help, and when it can't)

The key-rate ↔ capture-risk trade-off lives in the cloud of per-cell points
`(risk_c, keybits_c)`. A **static** node sits at one cell → it can only realize a
point on the Pareto front of that cloud. A **mobile** node spends its `T` slots
across several cells → it realizes (reachability permitting) *convex combinations*
of cell points. So:

- **Mobility wins** exactly where the cell cloud's Pareto front is **concave**:
  no single cell offers the desired risk/key blend, but time-sharing a high-key/
  high-risk cell with a low-risk cell averages to a point *above* the static front.
- **Mobility is neutral** where the front is convex / a single cell is already
  optimal (loose or trivial risk budgets).
- **Mobility loses** when **transit cost** (slots spent flying from the launch base
  through low-key cells) exceeds the time-sharing gain. This is the UAV-specific
  penalty the model must keep honest — it grows with launch distance and shrinks as
  the horizon `T` amortizes it.

## Two baselines (be explicit about which)

1. **Idealized static pre-placement** — node sited at the optimal cell, *no transit,
   no launch constraint*. This is the literature assumption and the **hard**
   baseline. Mobile only beats it when concavity gain > transit cost.
2. **Fly-and-park** (single fixed serving cell, same kinematics) — mobile strictly
   generalizes it, so mobile ≥ this always. Cleaner but weaker claim.

The honest contribution is the **phase boundary**: the region of operating
parameters (threat intensity, horizon, UAV speed, launch distance) where the mobile
trajectory beats baseline (1), and where it does not.

## Linear scalarization degenerates — use the risk budget

Maximizing `Σ key − λ·Σ risk` (DP, `solve_dp`) is *linear* per slot, so its optimum
puts every slot in the single argmax cell → it collapses to static. The Pareto
front's interior (the time-sharing points where mobility pays off) is only revealed
by the **hard risk-budget MILP** (`solve_milp`). Keep the DP for the convex-hull
vertices / fast checks; use the MILP for the real front.

## Honest limitations of v0

- 2D grid at fixed altitude; altitude/elevation not yet a decision variable.
- Asymptotic key rate (no finite-key correction).
- Independent-slot capture hazard; no adversary mobility/learning.
- Trust enters only via risk; no Byzantine/partial-trust relay model.
- Energy modeled implicitly via horizon + speed bound, not a battery state.

## Next steps

- DONE: coverage boundary (`out/coverage_boundary.png`) with MILP–LP gap band;
  shuttle trajectory (`out/trajectory.png`); single-UAV phase diagram over
  demand × horizon (`out/phase_diagram.png`) — clean staircase feasibility frontier,
  "2-node" region where one UAV can't keep up.
- Note: MILP slows markedly at T=22 / 7×7 grid (HIGHS) — motivates GUROBI/MOSEK for
  the larger sweeps.
- TODO (paper on hold): finite-key regime; altitude as a 3rd spatial DoF;
  multi-UAV / hand-off scheduling; trajectory path-smoothing regularizer for cleaner
  figures.
