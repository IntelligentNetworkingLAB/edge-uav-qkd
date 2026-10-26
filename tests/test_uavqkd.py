import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from antijam.controllers import ALL_CONTROLLERS        # noqa: E402
from antijam.env import JAMMER_LADDER, JammingEnv      # noqa: E402
from uavqkd import report                              # noqa: E402
from uavqkd.channel import ChannelParams, secret_key_rate   # noqa: E402
from uavqkd.scenario import coverage_scenario          # noqa: E402
from uavqkd.solver import solve_milp_demand, static_demand_min_risk, two_static_nodes_min_risk  # noqa: E402
from uavqkd.trust import TrustParams, ThreatSource, risk_field  # noqa: E402


class TestChannel(unittest.TestCase):
    def test_key_rate_falls_with_distance(self):
        r = secret_key_rate(np.array([500.0, 2000.0, 8000.0]), ChannelParams())
        self.assertTrue(np.all(r >= 0))
        self.assertTrue(r[0] > r[1] > r[2])


class TestTrust(unittest.TestCase):
    def test_risk_peaks_at_threat(self):
        p = TrustParams(baseline=0.05, sources=(ThreatSource((0.0, 0.0), 1.0, 500.0),))
        near, far = risk_field(np.array([[0.0, 0.0], [5000.0, 0.0]]), p)
        self.assertGreater(near, far)
        self.assertAlmostEqual(far, 0.05, places=3)


class TestCoverage(unittest.TestCase):
    def test_mobile_uav_covers_both_pairs(self):
        s = coverage_scenario(cluster_sep=8000.0, horizon=10, speed=3000.0, grid_n=5)
        D = 2e6
        lp = solve_milp_demand(s, demand=D, relax=True)
        self.assertIsNotNone(lp)
        self.assertFalse(static_demand_min_risk(s, D)[2])
        self.assertTrue(two_static_nodes_min_risk(s, D)[2])


class TestAntijam(unittest.TestCase):
    def test_controllers_run_against_all_jammers(self):
        for c in ALL_CONTROLLERS:
            for j in JAMMER_LADDER:
                env = JammingEnv(M=4, jammer=j(4), seed=0)
                ctrl = c()
                ctrl.reset(4, np.random.default_rng(0))
                for _ in range(50):
                    a = ctrl.act()
                    ack, jammed = env.step(a)
                    ctrl.update(a, ack, jammed)
                    self.assertIn(ack, (0, 1))


class TestReport(unittest.TestCase):
    def test_markdown_sections(self):
        res = {'simple': {'fixed_worst': 0.1, 'move_worst': 20.0},
               'coverage': [dict(demand_mbit=4, mobile_risk=2.6, lp_lower_bound=2.4,
                                 opt_gap_pct=7.0, mobile_key_mbit=8.0, mobile_survival=0.1,
                                 one_static_risk=float('nan'), two_static_risk=1.5)],
               'antijam': {'controllers': ['A', 'B'], 'jammers': ['x'], 'goodput': [[0.5], [0.9]]}}
        md = report.to_markdown(res)
        self.assertIn('infeasible', md)
        self.assertIn('**0.900**', md)


if __name__ == '__main__':
    unittest.main()
