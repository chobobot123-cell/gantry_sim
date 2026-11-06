#!/usr/bin/env python3
"""영상 동작 순서를 참고한 접근: 팔 내림, 이동 중 팔 접기, 첫 점 정렬.
끝 자세는 기존 전체 계획의 최초 paint=1 목표이며, 시작 팔 자세만 변경한다.
실제 영상 관절각의 복원이 아니며 충돌/실제 제어 검사는 수행하지 않는다.
"""
import json
from pathlib import Path
import numpy as np
from cell_model import Cell

root = Path(__file__).resolve().parent
source = root / 'out_gun_off_231_speed/trajectory_bumper.npz'
destination = root / 'out_approach'
destination.mkdir(exist_ok=True)
with np.load(source) as z:
    data = {k: z[k].copy() for k in z.files}
end = int(np.flatnonzero((data['kind'] == 'target') & (data['paint'] == 1))[0])
goal = data['X'][end].copy()
g0, g1 = data['X'][0, :2].copy(), goal[:2].copy()
q1 = goal[2:].copy()
# 최종 자세와 같은 분기에서 팔을 거의 아래로 펴는 시뮬레이션 준비 자세.
q0 = q1.copy()
q0[[1, 2, 4]] = np.deg2rad([10., -20., 30.])
vmax, amax, sample_dt = .5, 1., .02
length = float(np.linalg.norm(g1 - g0))
if length <= 1e-9:
    raise ValueError('This approach requires a nonzero gantry translation.')
vpeak = min(vmax, np.sqrt(length * amax))
ta = vpeak / amax
cruise = max(0., (length - vpeak * ta) / vpeak)
duration = 2 * ta + cruise
t = np.linspace(0., duration, int(np.ceil(duration / sample_dt)) + 1)
distance = np.where(t < ta, .5 * amax * t**2,
                    np.where(t <= ta + cruise, .5 * amax * ta**2 + vpeak * (t - ta),
                             length - .5 * amax * (duration - t)**2))
progress = np.clip(distance / length, 0., 1.)
G = g0 + progress[:, None] * (g1 - g0)
# 처음 출발 위치부터 팔도 함께 접기 시작한다. 이동 중간까지 기다리지 않는다.
fold_start, fold_end = 0., .45
u = np.clip((progress - fold_start) / (fold_end - fold_start), 0., 1.)
blend = 10*u**3 - 15*u**4 + 6*u**5
Q = q0 + blend[:, None] * (q1 - q0)
X = np.c_[G, Q]
X[-1] = goal
cell = Cell(root / 'paint_cell_collision_231.urdf', gun='off')
assert np.all(X >= cell.lower - 1e-8) and np.all(X <= cell.upper + 1e-8)
dt = np.diff(t)
velocity = np.diff(X, axis=0) / dt[:, None]
acceleration = np.diff(velocity, axis=0) / ((dt[1:] + dt[:-1])[:, None] / 2)
gantry_peak = float(np.linalg.norm(velocity[:, :2], axis=1).max())
arm_peak = np.max(np.abs(velocity[:, 2:]), axis=0)
assert gantry_peak <= vmax + 1e-8
assert np.max(np.linalg.norm(acceleration[:, :2], axis=1)) <= amax + 1e-7
assert np.all(arm_peak <= np.minimum(cell.vel[2:], np.deg2rad(120)) + 1e-8)
assert np.max(np.abs(acceleration[:, 2:])) <= np.deg2rad(600) + 1e-8
tcp = np.array([cell.tcp(x)[0] for x in X])
assert np.array_equal(X[-1], goal)
assert np.linalg.norm(tcp[-1] - data['tcp'][end]) < 1e-7
n = len(t)
kind = np.full(n, 'transit', dtype='<U7'); kind[0] = 'home'; kind[-1] = 'target'
row = np.full(n, -1); row[-1] = int(data['row'][end])
csv_idx = np.full(n, -1); csv_idx[-1] = int(data['csv_idx'][end])
np.savez(destination / 'trajectory_bumper.npz', X=X, t=t, tcp=tcp,
         names=data['names'], tool_names=data['tool_names'], tool_values=data['tool_values'],
         paint=np.zeros(n, dtype=int), kind=kind, row=row, csv_idx=csv_idx)
report = {
    'method': 'gantry trapezoid with a quintic arm fold during travel',
    'video_reference': '/home/jhin/비디오/Screencasts/Screencast from 10-31-2026 02:03:40 PM.webm',
    'source_of_final_pose': str(source), 'scene': 'paint_cell_231.urdf',
    'target_row': int(row[-1]), 'target_csv_index': int(csv_idx[-1]),
    'duration_s': float(t[-1]), 'gantry_xy_speed_limit_m_s': vmax,
    'gantry_xy_peak_segment_speed_m_s': gantry_peak,
    'assumed_gantry_acceleration_m_s2': amax,
    'gantry_start_m': g0.tolist(), 'gantry_end_m': g1.tolist(),
    'initial_arm_deg': np.degrees(q0).tolist(), 'final_arm_deg': np.degrees(q1).tolist(),
    'arm_fold_start_travel_fraction': fold_start, 'arm_fold_end_travel_fraction': fold_end,
    'arm_fold_start_s': float(np.interp(fold_start, progress, t)),
    'arm_fold_end_s': float(np.interp(fold_end, progress, t)),
    'max_arm_velocity_deg_s': float(np.degrees(arm_peak.max())),
    'max_arm_acceleration_deg_s2': float(np.degrees(np.abs(acceleration[:, 2:]).max())),
    'tcp_start_m': tcp[0].tolist(), 'tcp_end_m': tcp[-1].tolist(),
    'tcp_height_range_m': [float(tcp[:, 2].min()), float(tcp[:, 2].max())],
    'spray': 'OFF throughout approach and final hold',
    'collision_checked': False, 'physical_controller_validated': False,
    'note': 'Video-inspired sequence, not measured reconstruction. Start arm pose is new. Original full painting plan unchanged.'
}
(destination / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
