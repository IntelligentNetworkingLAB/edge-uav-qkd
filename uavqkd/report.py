# 결과 보고서 (마크다운 / JSON).
#
# run.py 가 모은 세 결과를 표로 정리한다.
#   simple   : 고정 노드 vs 이동 UAV, 가장 적게 받은 쌍의 키 (Mbit)
#   coverage : 요구 키량별 최소 포획 위험 (UAV 1대 MILP, LP 하한, 고정 노드 1대/2대)
#   antijam  : 제어기 x 재머 goodput 표

import json
import math


def _f(x, fmt='%.2f'):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '-'
    return fmt % x


def to_markdown(res, meta=None):
    L = ['# Edge UAV-QKD report', '']
    if meta:
        L += [', '.join('%s: %s' % kv for kv in meta.items()), '']

    s = res.get('simple')
    if s:
        L += ['## Coverage: fixed node vs mobile UAV', '',
              '| Relay | Key to the worst-served pair (Mbit) |', '|---|---:|',
              '| Fixed node | %s |' % _f(s['fixed_worst'], '%.1f'),
              '| Mobile UAV | %s |' % _f(s['move_worst'], '%.1f'), '']

    rows = res.get('coverage')
    if rows:
        L += ['## Capture risk to meet a per-pair key demand', '',
              '| Demand (Mbit/pair) | 1 mobile UAV (MILP) | LP lower bound | Gap | 1 static node | 2 static nodes |',
              '|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            L.append('| %d | %s | %s | %s | %s | %s |' % (
                r['demand_mbit'], _f(r['mobile_risk']), _f(r['lp_lower_bound']),
                _f(r['opt_gap_pct'], '%.1f%%'),
                _f(r['one_static_risk']) if not math.isnan(r['one_static_risk']) else 'infeasible',
                _f(r['two_static_risk']) if not math.isnan(r['two_static_risk']) else 'infeasible'))
        L += ['', 'Risk is the cumulative capture risk along the flight. The LP relaxation is a '
              'lower bound, so the gap bounds how far the MILP trajectory is from optimal.', '']

    aj = res.get('antijam')
    if aj:
        jn, cn, g = aj['jammers'], aj['controllers'], aj['goodput']
        L += ['## Anti-jamming goodput (jammers ordered by adaptivity)', '',
              '| Controller | ' + ' | '.join(jn) + ' |', '|---|' + '---:|' * len(jn)]
        for ci, c in enumerate(cn):
            best = [max(range(len(cn)), key=lambda k: g[k][ji]) == ci for ji in range(len(jn))]
            L.append('| %s | ' % c + ' | '.join(
                ('**%.3f**' if b else '%.3f') % g[ci][ji] for ji, b in enumerate(best)) + ' |')
        L += ['', 'Bold marks the best controller for each jammer.', '']
    return '\n'.join(L)


def to_json(res, meta=None):
    return json.dumps({'meta': meta or {}, **res}, indent=2, ensure_ascii=False, default=float)
