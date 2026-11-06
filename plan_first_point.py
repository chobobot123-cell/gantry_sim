#!/usr/bin/env python3
"""첫 CSV 점에 도달하는 캐리지 후보 + 6축 IK. 충돌 없는 경로를 보장하지 않는다."""
import csv
import json
import argparse
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--gun', choices=['on', 'off'], default='on', help='건 전개(on) 또는 수납(off); 분사 신호와 별도')
parser.add_argument('--scene', default='paint_cell.urdf')
args = parser.parse_args()
model = pin.buildModelFromUrdf(str(ROOT / args.scene))
data = model.createData()
fid_tcp = model.getFrameId('gun_nozzle')
fid_part = model.getFrameId('bumper_link')
q_home = np.zeros(model.nq)
q_home[:2] = model.upperPositionLimit[:2]
q_home[3:5] = [-np.pi / 2, np.pi / 2]
# 기계적으로 전개된 건. 분사 신호와는 별도다.
q_home[8:10] = [0.0736, 0.0736 * 17.3773] if args.gun == 'on' else [0., 0.]
pin.framesForwardKinematics(model, data, q_home)
part = data.oMf[fid_part].copy()
with (ROOT / 'ES300h_front_bumper/painting_trajectory.csv').open(newline='') as f:
    row = next(csv.DictReader(f))
quat = np.array([float(row['orientation_' + a]) for a in 'xyzw'])
quat /= np.linalg.norm(quat)
target_local = pin.SE3(Rotation.from_quat(quat).as_matrix(),
                       np.array([float(row['position_' + a]) for a in 'xyz']))
# 기존 CSV와 PLY가 동일 스캔 좌표라는 기존 프로젝트의 가정.
target = part * target_local
lower, upper = model.lowerPositionLimit[2:8], model.upperPositionLimit[2:8]
seeds = [q_home[2:8].copy(), np.zeros(6),
         np.array([0., -1., 1., 0., 1., 0.]),
         np.array([0., 1., -1., 0., -1., 0.])]
step = .5
offsets = np.arange(-2, 2.01, step)
candidates = sorted({(round(float(x), 6), round(float(y), 6))
                     for x in target.translation[0] + offsets
                     for y in target.translation[1] + offsets
                     if model.lowerPositionLimit[0] <= x <= model.upperPositionLimit[0]
                     and model.lowerPositionLimit[1] <= y <= model.upperPositionLimit[1]},
                    key=lambda xy: np.linalg.norm(np.array(xy) - q_home[:2]))

def residual(arm, xy):
    q = q_home.copy()
    q[:2], q[2:8] = xy, arm
    pin.framesForwardKinematics(model, data, q)
    actual = data.oMf[fid_tcp]
    return np.r_[actual.translation - target.translation,
                 pin.log3(actual.rotation.T @ target.rotation)]

best = None
attempts = 0
closest = (float('inf'), None, None)
for xy in candidates:
    for seed in seeds:
        attempts += 1
        result = least_squares(residual, np.clip(seed, lower + 1e-7, upper - 1e-7),
                               args=(xy,), bounds=(lower, upper), max_nfev=120,
                               ftol=1e-9, xtol=1e-9, gtol=1e-9)
        err = residual(result.x, xy)
        if np.linalg.norm(err) < closest[0]:
            closest = (float(np.linalg.norm(err)), xy, err.copy())
        if np.linalg.norm(err[:3]) < 1e-3 and np.linalg.norm(err[3:]) < np.deg2rad(.2):
            best = (xy, result.x, err)
            break
    if best is not None:
        break
if best is None:
    raise SystemExit(f'후보 {len(candidates)}개에서 IK 해를 찾지 못했습니다. 가장 작은 잔차: 위치 {np.linalg.norm(closest[2][:3]):.4f} m, 자세 {np.rad2deg(np.linalg.norm(closest[2][3:])):.2f} deg, X/Y {closest[1]}. 전체 공간의 불가능 증명은 아닙니다.')
xy, arm, err = best
q_goal = q_home.copy()
q_goal[:2], q_goal[2:8] = xy, arm
out = {'source': 'ES300h_front_bumper/painting_trajectory.csv',
       'scene': args.scene,
       'row_index': int(row['row_index']), 'point_index': int(row['point_index']),
       'paint': int(row['paint']), 'tool_mechanical_state': args.gun,
       'target_world_xyz_m': target.translation.tolist(),
       'target_world_quat_xyzw': Rotation.from_matrix(target.rotation).as_quat().tolist(),
       'home_q': q_home.tolist(), 'goal_q': q_goal.tolist(),
       'position_error_m': float(np.linalg.norm(err[:3])),
       'orientation_error_deg': float(np.rad2deg(np.linalg.norm(err[3:]))),
       'grid_step_m': step, 'candidate_count': len(candidates), 'ik_attempts': attempts,
       'collision_checked': False, 'interpolation_checked': False}
(ROOT / 'first_point.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=2))
