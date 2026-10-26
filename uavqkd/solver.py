"""Trajectory + scheduling optimization over a time-expanded grid.

Decision: which cell the UAV occupies each of T slots (movement bounded by speed),
and implicitly which ground-station pair it serves (best per cell, precomputed).

Objective: maximize cumulative key bits minus lambda * cumulative capture risk.

Two solvers:
  * `solve_dp`     -- exact dynamic program for the lambda-scalarized objective
                      (longest path in a layered DAG). No external solver. Sweeping
                      lambda traces the convex hull of the Pareto front.
  * `solve_milp`   -- cvxpy MILP that maximizes key subject to a hard risk budget B.
                      Recovers non-hull Pareto points too. Solver is swappable
                      (HIGHS locally; GUROBI/MOSEK with a license).

`static_best_hover` is the honest baseline: a *statically pre-placed* node that
hovers at one fixed cell for the whole horizon -- the assumption prior trusted-node
work makes. Comparing the two fronts shows where mobility actually helps.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .scenario import Scenario


@dataclass
class TrajectoryResult:
    cells: np.ndarray          # (T,) grid indices visited
    total_key: float           # cumulative key bits
    total_risk: float          # cumulative capture risk (additive exposure)
    survival: float            # P(never captured) along the path
    label: str = ""
    runtime: float = 0.0       # solver wall-clock (s)
    bound: float = np.nan      # risk lower bound (LP/dual), where available
    feasible: bool = True


def _path_metrics(s: Scenario, cells: np.ndarray, keybits, risk) -> TrajectoryResult:
    total_key = float(keybits[cells].sum())
    total_risk = float(risk[cells].sum())
    from .trust import survival_probability
    survival = survival_probability(risk[cells], s.trust)
    return TrajectoryResult(cells=cells, total_key=total_key,
                            total_risk=total_risk, survival=survival)


def solve_dp(s: Scenario, lam: float) -> TrajectoryResult:
    """Exact DP for max sum_t (keybits - lam*risk) over feasible trajectories."""
    keybits, _ = s.cell_keybits()
    risk = s.cell_risk()
    reach = s.reachable()                      # (Nc, Nc) bool, reach[p, c]
    v = keybits - lam * risk                   # per-cell scalarized reward
    T, Nc = s.horizon_slots, s.n_cells

    NEG = -1e30
    f = np.full((T, Nc), NEG)
    back = np.full((T, Nc), -1, dtype=int)
    f[0, s.start_cell] = v[s.start_cell]

    for t in range(1, T):
        prev = f[t - 1]                        # (Nc,)
        for c in range(Nc):
            preds = np.where(reach[:, c])[0]   # cells p that can reach c
            if preds.size == 0:
                continue
            vals = prev[preds]
            j = int(np.argmax(vals))
            if vals[j] <= NEG / 2:
                continue
            f[t, c] = vals[j] + v[c]
            back[t, c] = preds[j]

    end = int(np.argmax(f[T - 1]))
    cells = np.empty(T, dtype=int)
    cells[T - 1] = end
    for t in range(T - 1, 0, -1):
        cells[t - 1] = back[t, cells[t]]

    res = _path_metrics(s, cells, keybits, risk)
    res.label = f"mobile-DP(lam={lam:g})"
    return res


def static_best_hover(s: Scenario, lam: float) -> TrajectoryResult:
    """Best statically pre-placed node: hover one fixed cell for all T slots."""
    keybits, _ = s.cell_keybits()
    risk = s.cell_risk()
    v = keybits - lam * risk
    c = int(np.argmax(v))                       # best fixed cell for this lambda
    cells = np.full(s.horizon_slots, c, dtype=int)
    res = _path_metrics(s, cells, keybits, risk)
    res.label = f"static-hover(lam={lam:g})"
    return res


def pareto_front(s: Scenario, lambdas, method="dp") -> list[TrajectoryResult]:
    """Trace a Pareto front by sweeping the trade-off weight lambda."""
    solver = solve_dp if method == "dp" else static_best_hover
    seen, front = set(), []
    for lam in lambdas:
        r = solver(s, lam)
        key = (round(r.total_key, 3), round(r.total_risk, 6))
        if key not in seen:
            seen.add(key)
            front.append(r)
    return front


def solve_milp(s: Scenario, risk_budget: float, solver: str = "HIGHS") -> TrajectoryResult:
    """Maximize cumulative key bits s.t. cumulative risk <= risk_budget.

    Modeled in cvxpy; pass solver='GUROBI' or 'MOSEK' with a license.
    """
    import cvxpy as cp

    keybits, _ = s.cell_keybits()
    risk = s.cell_risk()
    reach = s.reachable()
    T, Nc = s.horizon_slots, s.n_cells

    x = cp.Variable((T, Nc), boolean=True)
    cons = [cp.sum(x, axis=1) == 1]                      # one cell per slot
    cons += [x[0, s.start_cell] == 1]                    # start fixed
    for t in range(1, T):                                # movement feasibility
        for c in range(Nc):
            preds = np.where(reach[:, c])[0]
            cons.append(x[t, c] <= cp.sum(x[t - 1, preds]))
    cons.append(cp.sum(x @ risk) <= risk_budget)         # risk budget

    obj = cp.Maximize(cp.sum(x @ keybits))
    prob = cp.Problem(obj, cons)
    prob.solve(solver=solver)

    cells = np.array([int(np.argmax(x.value[t])) for t in range(T)])
    res = _path_metrics(s, cells, keybits, risk)
    res.label = f"mobile-MILP(B={risk_budget:g},{solver})"
    return res


def _active_pairrate(s: Scenario):
    """Per-cell key bits for each ground-station pair, plus indices of pairs that
    actually carry capacity (prunes far cross-cluster pairs with ~0 rate)."""
    R = s.link_key_rates()
    pairrate = np.stack(
        [np.minimum(R[:, a], R[:, b]) for (a, b) in s.pairs], axis=1
    ) * s.slot_seconds                                    # (Nc, Npairs)
    cap = pairrate.max(axis=0)
    active = [j for j in range(len(s.pairs)) if cap[j] > 1e-6 * cap.max()]
    return pairrate, active


def solve_milp_demand(s: Scenario, demand, solver="HIGHS", relax=False):
    """COVERAGE model: each slot the UAV picks a cell AND one served pair; it must
    deliver at least `demand` key bits to *every* active pair, minimizing cumulative
    capture risk. A static node stuck at one cell can serve spatially-separated pairs
    only poorly -- this is where a mobile trajectory genuinely wins.

    `demand` is a scalar (same for all active pairs) or a dict {pair_index: bits}.
    `relax=True` drops integrality -> LP relaxation (fractional slot time-sharing),
    a fast lower bound on the achievable risk.
    Returns a TrajectoryResult (cells = cell visited each slot) or None if infeasible.
    """
    import cvxpy as cp
    import time

    pairrate, active = _active_pairrate(s)
    risk = s.cell_risk()
    reach = s.reachable()
    T, Nc, Na = s.horizon_slots, s.n_cells, len(active)
    if isinstance(demand, dict):
        D = {j: demand.get(active[j], 0.0) for j in range(Na)}
    else:
        D = {j: float(demand) for j in range(Na)}

    # z[t, c*Na + jj] = 1  <=>  at slot t, UAV in cell c serving active-pair jj
    if relax:
        z = cp.Variable((T, Nc * Na))
        box = [z >= 0, z <= 1]
    else:
        z = cp.Variable((T, Nc * Na), boolean=True)
        box = []
    rate_flat = np.array([pairrate[c, active[jj]] for c in range(Nc) for jj in range(Na)])
    risk_flat = np.array([risk[c] for c in range(Nc) for jj in range(Na)])

    cons = list(box)
    cons += [cp.sum(z, axis=1) == 1]                     # one (cell,pair) per slot
    cons += [cp.sum(z[0, s.start_cell * Na:(s.start_cell + 1) * Na]) == 1]
    # cell marginal x[t,c] = sum over served pairs
    def xcol(t, c):
        return cp.sum(z[t, c * Na:(c + 1) * Na])
    for t in range(1, T):                                # movement feasibility
        for c in range(Nc):
            preds = np.where(reach[:, c])[0]
            cons.append(xcol(t, c) <= cp.sum([xcol(t - 1, p) for p in preds]))
    for jj in range(Na):                                 # per-pair demand
        cols = z[:, jj::Na]                              # (T, Nc) for this pair
        cons.append(cp.sum(cols @ pairrate[:, active[jj]]) >= D[jj])

    prob = cp.Problem(cp.Minimize(cp.sum(z @ risk_flat)), cons)
    t0 = time.perf_counter()
    prob.solve(solver=solver)
    runtime = time.perf_counter() - t0
    if z.value is None:
        return None

    occ = z.value.reshape(T, Nc, Na).sum(axis=2)         # (T, Nc) cell occupancy
    cells = occ.argmax(axis=1)
    keybits, _ = s.cell_keybits()
    res = _path_metrics(s, cells, keybits, risk)
    res.total_risk = float(prob.value)                   # exact (fractional) objective
    res.total_key = float((z.value @ rate_flat).sum())   # key actually delivered
    res.runtime = runtime
    res.label = f"{'LP' if relax else 'MILP'}({solver})"
    if relax:
        res.bound = float(prob.value)                    # LP is a valid lower bound
    return res


def static_demand_min_risk(s: Scenario, demand):
    """Best static (single fixed cell) node under the coverage demand: from one cell
    it time-shares its slots among pairs. Returns (min_risk, cell, feasible)."""
    pairrate, active = _active_pairrate(s)
    risk = s.cell_risk()
    T = s.horizon_slots
    Na = len(active)
    D = (demand if isinstance(demand, dict)
         else {active[j]: float(demand) for j in range(Na)})
    best = (np.inf, -1, False)
    for c in range(s.n_cells):
        slots = 0
        ok = True
        for j in active:
            r = pairrate[c, j]
            if r <= 0:
                ok = False
                break
            slots += int(np.ceil(D[j] / r))
        if not ok or slots > T:
            continue
        cost = slots * risk[c]                            # risk only for slots used
        if cost < best[0]:
            best = (cost, c, True)
    return best


def two_static_nodes_min_risk(s: Scenario, demand):
    """Deployment baseline: TWO statically-placed nodes, one dedicated to each pair.
    Each independently picks the cell minimizing its own risk*slots to meet demand.
    Returns (total_risk, [cell_per_pair], feasible) -- costs two devices."""
    pairrate, active = _active_pairrate(s)
    risk = s.cell_risk()
    T = s.horizon_slots
    D = (demand if isinstance(demand, dict)
         else {active[j]: float(demand) for j in range(len(active))})
    total, cells, feasible = 0.0, [], True
    for j in active:
        best = (np.inf, -1)
        for c in range(s.n_cells):
            r = pairrate[c, j]
            if r <= 0:
                continue
            slots = int(np.ceil(D[j] / r))
            if slots > T:
                continue
            cost = slots * risk[c]
            if cost < best[0]:
                best = (cost, c)
        if best[1] < 0:
            feasible = False
        else:
            total += best[0]
            cells.append(best[1])
    return (total, cells, feasible)


if __name__ == "__main__":
    from .scenario import default_scenario

    sc = default_scenario()
    for lam in [0.0, 5e4, 1e5, 5e5, 1e6, 1e7]:
        m = solve_dp(sc, lam)
        st = static_best_hover(sc, lam)
        print(f"lam={lam:8.0f}  mobile key={m.total_key/1e3:9.1f}k risk={m.total_risk:5.2f} "
              f"surv={m.survival:.3f} | static key={st.total_key/1e3:9.1f}k risk={st.total_risk:5.2f}")
