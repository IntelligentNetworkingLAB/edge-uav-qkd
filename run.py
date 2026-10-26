#!/usr/bin/env python3
# 엣지 무선 보안 실험: 이동 UAV 신뢰 노드 QKD + 안티재밍 채널 호핑.
#
#   python run.py                  quick 설정, 콘솔 보고서
#   python run.py --preset full    논문 설정 (MILP 가 길다)
#   python run.py --all            마크다운, JSON, 그림을 out/ 에 저장
#   python run.py --only coverage  simple | coverage | antijam 중 일부만
#   python run.py --phase          (demand x horizon) 위상도까지 (오래 걸림)

import argparse
import contextlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from uavqkd import report                                         # noqa: E402

OUT = os.path.join(HERE, 'out')
DEF_CFG = os.path.join(HERE, 'config', 'experiment.json')
PARTS = ('simple', 'coverage', 'antijam')


def save(path, body):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(body)
    return os.path.relpath(path, HERE)


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding='utf-8')
        except Exception:
            pass
    ap = argparse.ArgumentParser(description='엣지 무선 보안 실험')
    ap.add_argument('-c', '--config', default=DEF_CFG)
    ap.add_argument('--preset', default='quick')
    ap.add_argument('--only', nargs='+', choices=PARTS)
    ap.add_argument('--phase', action='store_true', help='위상도도 계산한다')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--md', nargs='?', const=os.path.join(OUT, 'report.md'))
    ap.add_argument('--json', nargs='?', const=os.path.join(OUT, 'result.json'))
    ap.add_argument('--quiet', action='store_true', help='진행 로그를 숨긴다')
    args = ap.parse_args(argv)

    with open(args.config, encoding='utf-8') as f:
        exp = json.load(f)
    if args.preset not in exp['presets']:
        ap.error('preset 은 %s 중 하나' % ', '.join(exp['presets']))
    p, cov = exp['presets'][args.preset], exp['coverage']
    parts = args.only or PARTS
    log = open(os.devnull, 'w') if args.quiet else sys.stderr

    t0 = time.time()
    res = {}
    with contextlib.redirect_stdout(log):
        if 'simple' in parts:
            from uavqkd import simple_compare
            res['simple'] = simple_compare.run(horizon=p['horizon'], speed=cov['uav_speed_mps'],
                                               grid_n=p['grid_n'])
        if 'coverage' in parts:
            from uavqkd import experiment
            res['coverage'] = experiment.run(
                cluster_sep=cov['cluster_sep_m'], horizon=p['horizon'],
                speed=cov['uav_speed_mps'], grid_n=p['grid_n'],
                demands_mbit=tuple(p['demands_mbit']),
                threat_intensity=cov['threat_intensity'])
        if 'antijam' in parts:
            from antijam import experiment as aj
            from antijam.controllers import ALL_CONTROLLERS
            from antijam.env import JAMMER_LADDER
            M = p['antijam_channels']
            grid = aj.run(M=M, T=p['antijam_slots'], seeds=p['antijam_seeds'])
            res['antijam'] = {'controllers': [c.name for c in ALL_CONTROLLERS],
                              'jammers': [j(M).name for j in JAMMER_LADDER],
                              'goodput': grid.tolist()}
        if args.phase:
            from uavqkd import phase_diagram
            phase_diagram.run(cluster_sep=cov['cluster_sep_m'], speed=cov['uav_speed_mps'],
                              threat_intensity=cov['threat_intensity'])

    meta = {'preset': args.preset, 'elapsed': '%.0fs' % (time.time() - t0)}
    print(report.to_markdown(res, meta))
    if args.all:
        args.md = args.md or os.path.join(OUT, 'report.md')
        args.json = args.json or os.path.join(OUT, 'result.json')
    if args.md:
        print('saved', save(args.md, report.to_markdown(res, meta)))
    if args.json:
        print('saved', save(args.json, report.to_json(res, meta)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
