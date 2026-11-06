#!/usr/bin/env python3
"""CSV 경로를 순서대로 풀어 RViz 재생 자료를 만든다. 시각화 전용, 충돌 미검증."""
import csv
import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
from transition_curves import lane_curve

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--csv', default='ES300h_front_bumper/painting_trajectory.csv')
args = ap.parse_args()
start = time.perf_counter()
first = json.loads((ROOT / 'first_point.json').read_text())
if first['tool_mechanical_state'] != 'on':
    raise SystemExit('전개 상태의 첫 점 계산이 필요합니다.')
model = pin.buildModelFromUrdf(str(ROOT / first['scene']))
data = model.createData()
fid = model.getFrameId('gun_nozzle')
q0 = np.array(first['home_q'], dtype=float)
q = np.array(first['goal_q'], dtype=float)
pin.framesForwardKinematics(model, data, q)
part = data.oMf[model.getFrameId('bumper_link')].copy()
groups = defaultdict(list)
with (ROOT / args.csv).open(newline='') as f:
    for i, r in enumerate(csv.DictReader(f)):
        p = np.array([float(r['position_' + axis]) for axis in 'xyz'])
        rot = Rotation.from_quat([float(r['orientation_' + axis]) for axis in 'xyzw'])
        groups[int(r['row_index'])].append((p, rot, int(r['paint']), i))

targets = []
keys = sorted(groups)
for row in keys:
    group = groups[row]
    if targets:
        prev = groups[row - 1]
        A, B = prev[-1], group[0]
        if row in (2, 13):
            curve = lane_curve([x[0] for x in prev], [x[0] for x in group], 1 if row == 2 else 2)
        else:
            n = max(1, math.ceil(np.linalg.norm(B[0]-A[0])/.04))
            curve = np.linspace(A[0], B[0], n+1)
        lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
        fractions = lengths / lengths[-1] if lengths[-1] else np.zeros(len(curve))
        interpolate = Slerp([0, 1], Rotation.concatenate([A[1], B[1]]))
        for p, u in zip(curve[1:-1], fractions[1:-1]):
            targets.append((p, interpolate(float(u)), 0, -1, row))
    targets.extend((p, rot, paint, idx, row) for p, rot, paint, idx in group)

lo, hi = model.lowerPositionLimit[:8], model.upperPositionLimit[:8]
step_offsets = [(0.,0.)] + sorted([(x,y) for x in (-.4,-.2,-.1,0.,.1,.2,.4)
                                    for y in (-.4,-.2,-.1,0.,.1,.2,.4) if x or y],
                                   key=lambda v: v[0]**2 + v[1]**2)
path, paints, indexes, errors = [q0.copy(), q.copy()], [0, 0], [-1, 0], []
failure = None
for k, (p, rot, paint, csv_idx, row) in enumerate(targets):
    if k == 0:
        continue
    goal = part * pin.SE3(rot.as_matrix(), p)
    previous = q.copy()
    def err(arm, xy):
        test = previous.copy()
        test[:2], test[2:8] = xy, arm
        pin.framesForwardKinematics(model, data, test)
        current = data.oMf[fid]
        return np.r_[current.translation-goal.translation, pin.log3(current.rotation.T @ goal.rotation)]
    found = False
    for dx,dy in step_offsets:
        xy = previous[:2] + [dx,dy]
        if np.any(xy < lo[:2]) or np.any(xy > hi[:2]):
            continue
        result = least_squares(err, previous[2:8], args=(xy,), bounds=(lo[2:8],hi[2:8]),
                               max_nfev=60, ftol=1e-7, xtol=1e-7, gtol=1e-7)
        e = err(result.x, xy)
        if np.linalg.norm(e[:3]) <= .001 and np.linalg.norm(e[3:]) <= np.deg2rad(.2):
            if np.max(np.abs(result.x-previous[2:8])) > np.deg2rad(40):
                continue
            q[:2], q[2:8] = xy, result.x
            found = True
            break
    if not found:
        failure = {'target_index': k, 'csv_index': csv_idx, 'row': row}
        print(f'IK 실패: {failure}', flush=True)
        break
    path.append(q.copy()); paints.append(paint); indexes.append(csv_idx)
    errors.append([float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))])
    if k % 100 == 0:
        print(f'계산 중 {k}/{len(targets)} ({time.perf_counter()-start:.1f}s)', flush=True)

path = np.array(path)
times = [0., max(15., 1.5*np.linalg.norm(path[1,:2]-path[0,:2])/.5)]
for k in range(2,len(path)):
    dq = path[k] - path[k-1]
    prev = targets[k-2][0]; cur = targets[k-1][0]
    dt = max(np.linalg.norm(dq[:2])/.5, np.max(np.abs(dq[2:8])/model.velocityLimit[2:8]),
             np.linalg.norm(cur-prev)/.2, .02)
    times.append(times[-1]+float(dt))
elapsed = time.perf_counter()-start
np.savez(ROOT/'follow_trajectory.npz', q=path, t=np.array(times), paint=np.array(paints), csv_index=np.array(indexes))
report = {'status': 'complete' if failure is None else 'partial', 'failure': failure, 'scene': first['scene'],
          'source_csv': args.csv,
          'total_csv_points': sum(len(v) for v in groups.values()), 'reached_csv_points': sum(i >= 0 for i in indexes),
          'total_targets_including_transitions': len(targets), 'completed_targets': len(path)-1,
          'planning_seconds': elapsed, 'playback_seconds': times[-1],
          'max_node_position_error_m': max((e[0] for e in errors), default=0),
          'max_node_orientation_error_deg': math.degrees(max((e[1] for e in errors), default=0)),
          'collision_checked': False, 'interpolation_checked': False,
          'acceleration_checked': False, 'assumed_tcp_speed_m_s': .2}
(ROOT/'follow_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
