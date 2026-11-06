#!/usr/bin/env python3
"""저장된 궤적만 읽어서 독립 재검사 (플래너 내부 상태 사용 안 함).

  1) CSV 946행이 전부 궤적의 target 노드로 들어 있는지 (인덱스 대조)
  2) target 노드 FK vs CSV 자세 (URDF 배치 변환 적용) 오차
  3) 모든 간선을 촘촘히 샘플링해 정확 거리 최소값 (50 mm 미만 쌍은 전부 정확 계산)
  4) 관절 리밋, 도장 플래그 일치
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cell_model import DIR, Cell  # noqa: E402

OUT = Path(__file__).resolve().parent / (sys.argv[1] if len(sys.argv) > 1 else 'out')
val = json.loads((OUT / 'validation.json').read_text())
WP = val.get('workpiece', 'bumper')                                # 계획한 작업물 (범퍼 / 펜더)
z = np.load(OUT / f'trajectory_{WP}.npz')
X, kind, csv_idx, paint = z['X'], z['kind'], z['csv_idx'], z['paint']
names = [str(n) for n in z['names']]
gantry = [n for n in names if n.startswith('gantry_')]          # 2축 또는 9축(상하 Z 포함) 궤적
cell = Cell(val['urdf'], name=WP, gun='on' if val['tool_state']['gun_lift'] > 0 else 'off', gantry=gantry)
for jn, v in val['tool_state'].items():                           # 고정 관절(건, 고정 Z 등)은 계획 때 값 그대로
    cell.q[cell.model.joints[cell.model.getJointId(jn)].idx_q] = v
assert cell.names == names, (cell.names, names)
for a, b, why in val['collision_exclusions']:
    k = [i for i in range(len(cell.ia)) if set(cell.pair_names(i)) == {a, b}]
    cell.exclude_pairs(k, why)

d = np.genfromtxt(val.get('csv') or DIR / {'bumper': 'ES300h_front_bumper', 'fender': 'ES300h_left_fender'}[WP] / 'painting_trajectory.csv', delimiter=',', names=True)   # 계획 때 쓴 경로 파일
n_csv = len(d)
tgt = np.flatnonzero(kind == 'target')
present = sorted(set(csv_idx[tgt].tolist()))
print(f'1) target nodes {len(tgt)} / CSV rows {n_csv}; unique CSV idx covered {len(present)}; '
      f'missing {sorted(set(range(n_csv)) - set(present))[:10]}')
# 순서: CSV 인덱스가 증가 순으로 방문되는가
print('   visit order monotonic:', bool(np.all(np.diff(csv_idx[tgt]) > 0)))

Tw = cell.T_wp
ep, ea = [], []
for k in tgt:
    i = int(csv_idx[k])
    q = np.array([d['orientation_x'][i], d['orientation_y'][i], d['orientation_z'][i], d['orientation_w'][i]])
    R = pin.Quaternion(q[3], q[0], q[1], q[2]).normalized().toRotationMatrix()
    pt = Tw[:3, :3] @ [d['position_x'][i], d['position_y'][i], d['position_z'][i]] + Tw[:3, 3]
    Rt = Tw[:3, :3] @ R
    p, Rf = cell.tcp(X[k])
    ep.append(np.linalg.norm(p - pt))
    ea.append(np.linalg.norm(pin.log3(Rf.T @ Rt)))
    assert paint[k] == int(d['paint'][i])
print(f'2) node error max {max(ep) * 1000:.4f} mm / {math.degrees(max(ea)):.5f} deg ; paint flags match')

lo, hi = cell.lower, cell.upper
print(f'3) joint limits ok: {bool(np.all(X >= lo - 1e-9) and np.all(X <= hi + 1e-9))}')

dmin, worst = float('inf'), None
n = 0
for k in range(len(X) - 1):
    a, b = X[k], X[k + 1]
    m = max(4, int(np.ceil(np.max(np.abs(b - a) / np.r_[np.full(len(gantry), .01), np.full(6, .02)]))))
    for u in np.linspace(0, 1, m + 1):
        # AABB 간격(거리 하한) >= 50 mm 인 쌍은 실제로도 >= 50 mm -> 50 mm 미만 판정은 정확
        dd = cell.distances((1 - u) * a + u * b, near=.05)
        n += 1
        j = int(np.argmin(dd))
        if dd[j] < dmin:
            dmin, worst = float(dd[j]), (k, round(float(u), 3), cell.pair_names(j))
print(f'4) dense exact check: {n} samples, min distance {dmin * 1000:.2f} mm at {worst}')
print('   (certified lower bound reported by planner: '
      + (f"{val['metrics']['certified_clearance_lower_bound_mm']:.3f} mm)" if val['metrics'].get('certified_clearance_lower_bound_mm') is not None else "계획 때 전 구간 보장 안 함)"))
