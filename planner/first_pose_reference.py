#!/usr/bin/env python3
"""첫 분사점 시연: 참조 팔 자세를 가깝게 유지하며 높이와 방향을 맞춘 뒤 X/Y를 계산한다.
충돌/접근경로/나머지 도장점은 검사하지 않는다.
"""
import json
import time
from pathlib import Path
import numpy as np
import pinocchio as pin
from scipy.optimize import minimize
import plan_bumper as pb
from cell_model import Cell, DIR

out = DIR / 'planner/out_first_reference'
out.mkdir(exist_ok=True)
cell = Cell(DIR / 'planner/paint_cell_collision_208.urdf', gun='off')
pb.CSV = DIR / 'ES300h_front_bumper/painting_trajectory_adjusted.csv'
row = next(r for r in pb.load_targets(cell) if r['row'] == 6)
i = int(np.flatnonzero(row['paint'])[0])
T = row['T'][i]
# 이전 결과의 첫 경로점(분사 OFF)을 참조 자세로만 사용한다.
prior = np.load(DIR / 'planner/out_row6_208_speed/trajectory_bumper.npz')
j = int(np.flatnonzero(prior['row'] == 6)[0])
q_ref = prior['X'][j, 2:].copy()
g_ref = np.zeros(2)

def residual(q):
    p, R = cell.tcp(np.r_[g_ref, q])
    return np.r_[p[2] - T[2, 3], pin.log3(R.T @ T[:3,:3])]

def base(q):
    p, _ = cell.tcp(np.r_[g_ref, q])
    return T[:2, 3] - p[:2]

def base_bounds(q):
    g = base(q)
    return np.r_[g - cell.lower[:2], cell.upper[:2] - g]

start = time.perf_counter()
sol = minimize(lambda q: np.sum((q - q_ref)**2), q_ref, method='SLSQP',
               bounds=list(zip(cell.lower[2:], cell.upper[2:])),
               constraints=[{'type':'eq','fun':residual}, {'type':'ineq','fun':base_bounds}],
               options={'ftol':1e-11, 'maxiter':300})
q = sol.x
g = base(q)
p, R = cell.tcp(np.r_[g,q])
pos = float(np.linalg.norm(p-T[:3,3])); ang = float(np.linalg.norm(pin.log3(R.T @ T[:3,:3])))
valid = bool(sol.success and pos < 1e-4 and ang < np.deg2rad(.1) and np.all(base_bounds(q)>=-1e-7))
report = {'description':'fixed reference posture; optimize arm z and orientation, then compute gantry XY; no collision checking',
          'reference':'existing row 6 first path point (paint OFF)', 'reference_q_deg':np.degrees(q_ref).tolist(),
          'row':6,'point_index':i,'csv_idx':int(row['csv_idx'][i]),'target_world_m':T[:3,3].tolist(),
          'gantry_xy_m':g.tolist(),'q_deg':np.degrees(q).tolist(), 'position_error_mm':pos*1000,
          'orientation_error_deg':np.degrees(ang), 'arm_deviation_deg_norm':float(np.linalg.norm(np.degrees(q-q_ref))),
          'elapsed_s':time.perf_counter()-start,'iterations':int(sol.nit),'valid':valid,'optimizer_message':str(sol.message)}
(out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(report,indent=2,ensure_ascii=False))
if not valid: raise SystemExit('No valid solution; no visualization generated.')
x = np.r_[g,q]
np.savez(out/'trajectory_bumper.npz',X=np.array([x,x]),t=np.array([0.,1.]),names=np.array(cell.names),
         tcp=np.array([p,p]),paint=np.array([0,0]),tool_names=np.array(['gun_lift','gun_swing']),tool_values=np.array([0.,0.]))
