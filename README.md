# Edge UAV-QKD — Secure Edge Links with a Mobile UAV Trusted Node

Two studies on securing wireless links at the network edge.

**Mobile trusted node for QKD.** Two pairs of ground stations far apart each need a given amount of secret key. A UAV acts as a trusted relay for quantum key distribution (decoy-state BB84) over free-space optical links. One pair sits near an adversary, so hovering there raises the risk of the UAV being captured. The UAV's trajectory and the pair it serves in each time slot are chosen jointly to meet both demands with the least cumulative capture risk.

**Anti-jamming channel hopping.** One edge link hops among M channels against a ladder of jammers ordered by adaptivity (fixed, sweep, slow learner, reactive, predictive). Q-learning wins against predictable jammers, but a co-adaptive jammer that learns the learner pushes it below blind random hopping. A distributionally robust controller with an ACK-driven hedging temperature (`RobustDRO`) holds up across the whole ladder.

## Main result

| Deployment | Can it meet both demands? | Devices |
|---|---|---|
| 1 static trusted node | no, at every demand (one location cannot reach both pairs) | 1 |
| 1 mobile UAV | yes, risk grows with the demand | 1 |
| 2 static trusted nodes | yes | 2 |

One mobile UAV replaces two fixed nodes, and the report gives the capture risk it pays to do so. The MILP solution is reported together with its LP relaxation, a lower bound on the risk, so the gap certifies how close the trajectory is to optimal.

## Getting started

```bash
pip install -r requirements.txt
python run.py                    # quick preset, console report
python run.py --all              # also writes out/report.md, out/result.json and figures
python run.py --preset full      # paper setting (9x9 grid, 18 slots; MILP takes minutes)
python run.py --only antijam     # simple | coverage | antijam
python run.py --phase            # also the (demand x horizon) phase diagram (slow)
```

Scenario and presets are in `config/experiment.json`. Each module can also be run on its own, for example `python -m uavqkd.experiment` or `python -m antijam.crossover`.

## Model

| Layer | File | Summary |
|---|---|---|
| FSO channel + key rate | `uavqkd/channel.py` | transmittance from beam spreading, atmospheric loss and detector efficiency; decoy-state BB84 asymptotic key rate (GLLP) |
| Capture risk | `uavqkd/trust.py` | risk field from adversary locations; per-slot capture hazard |
| Scenario | `uavqkd/scenario.py` | ground stations, threat sources, airspace grid, UAV speed and horizon |
| Solver | `uavqkd/solver.py` | time-expanded grid MILP (cvxpy, HIGHS by default; GUROBI/MOSEK via `solver=`), LP lower bound, static baselines |
| Experiments | `uavqkd/experiment.py`, `phase_diagram.py`, `simple_compare.py` | demand sweep, feasibility phase diagram, fixed-vs-mobile bar chart |
| Anti-jamming | `antijam/` | jammer ladder, controllers (Uniform, RandomHop, DRO, RobustDRO, Q-learning), win grid, adaptivity crossover, learning curves |

A trusted relay serving pair (a, b) is limited by its weaker link, `min(R(uav,a), R(uav,b))`.

## Limitations

Single UAV on a 2D grid at fixed altitude, asymptotic key rate (no finite-key correction), independent per-slot capture hazard, energy modeled only through the horizon and speed bound. The anti-jamming learner is tabular. Design notes and the methods that were tried and dropped are in [`docs/NOTES.md`](docs/NOTES.md) and [`docs/ANTIJAM_NOTES.md`](docs/ANTIJAM_NOTES.md).

## Project layout

```
run.py            runs the experiments and writes the report
uavqkd/           channel, risk, scenario, solver, experiments, report
antijam/          anti-jamming environment, controllers, experiments
config/           scenario and presets
docs/             research notes
tests/            unittest suite
```

## Tests

```bash
python -m unittest discover -s tests
```

## Acknowledgement

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.RS-2022-00155911, Artificial Intelligence Convergence Innovation Human Resources Development (Kyung Hee University)),(No.2019-0-01287, Evolvable Deep Learning Model Generation Platform for Edge Computing)

## License

Apache-2.0
