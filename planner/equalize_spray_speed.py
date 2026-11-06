#!/usr/bin/env python3
"""도장과 두 긴 우회의 노드 간 TCP 평균 속도를 같게 만드는 시뮬레이션 시간 배분."""
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
src = root/'out_gun_off_231_speed'
dst = root/'out_equal_speed'
dst.mkdir(exist_ok=True)
with np.load(src/'trajectory_bumper.npz') as z:
    data = {k:z[k] for k in z.files}
q,t,p,kind,rows,paint = (data[k] for k in ('X','t','tcp','kind','row','paint'))
ds = np.linalg.norm(np.diff(p,axis=0),axis=1)
dt = np.diff(t)
green = (paint[:-1] == 1) & (paint[1:] == 1)
reference = float(ds[green].sum()/dt[green].sum())
curves = []
mask = green.copy()
for a,b in ((1,2),(12,13)):
    start = int(np.flatnonzero((kind=='target') & (rows==a))[-1])
    end = int(np.flatnonzero((kind=='target') & (rows==b))[0])
    mask[start:end] = True
    curves.append((a,b,start,end))
if np.any(ds[mask] < 1e-8):
    raise ValueError('노즐 이동이 없는 구간의 시간 배분을 따로 정의해야 합니다.')
dt[mask] = ds[mask]/reference
vel = np.diff(q,axis=0)/dt[:,None]
acc = np.diff(vel,axis=0)/((dt[:-1]+dt[1:])/2)[:,None]
# 초기 값은 실제 장비 확정 사양이 아닌 기존 프로젝트 가정이다.
factors = {'xy_speed':float(np.linalg.norm(vel[:,:2],axis=1).max()/.5),
           'gantry_acceleration':float(np.sqrt(np.abs(acc[:,:2]).max()/1.0)),
           'arm_speed':float(np.degrees(np.abs(vel[:,2:]).max())/120.),
           'arm_acceleration':float(np.sqrt(np.degrees(np.abs(acc[:,2:]).max())/600.))}
factor = max(1., *factors.values())
dt *= factor
data['t'] = np.r_[0., np.cumsum(dt)]
output = dst/'trajectory_bumper.npz'
np.savez(output, **data)
val = json.loads((src/'validation.json').read_text())
val['equal_speed'] = {'tcp_segment_speed_m_s':reference/factor,
    'scope':'paint=1 양쪽 노드 및 1→2, 12→13 우회 간선',
    'speed_meaning':'노드 간 직선거리 / 구간시간; 관절 보간 중 실시간 TCP 속도는 다를 수 있음',
    'joint_velocity_deg_s_assumption':120., 'joint_acceleration_deg_s2_assumption':600.,
    'gantry_acceleration_m_s2_assumption':1., 'global_slowdown':factor, 'slowdown_factors':factors,
    'curves':[{'from_row':a,'to_row':b,'length_m':float(ds[i:j].sum()),
               'time_s':float(dt[i:j].sum())} for a,b,i,j in curves]}
val['metrics']['cycle_time_s'] = float(data['t'][-1])
val['trajectory_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
(dst/'validation.json').write_text(json.dumps(val,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(val['equal_speed'],indent=2,ensure_ascii=False))
print(f"cycle={data['t'][-1]:.2f} s; max XY speed={np.linalg.norm(np.diff(q[:,:2],axis=0)/dt[:,None],axis=1).max():.4f} m/s")
