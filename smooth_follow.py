#!/usr/bin/env python3
"""기존 캐리지 계단 경로를 평활화하고 고정된 캐리지 위치마다 팔 IK를 다시 계산한다."""
import json
import math
import time
from pathlib import Path
import numpy as np
import pinocchio as pin
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
start = time.perf_counter()
report = json.loads((ROOT/'follow_report.json').read_text())
if report['status'] != 'complete':
    raise SystemExit('원본 계획이 완성되지 않아 평활화하지 않습니다.')
saved = np.load(ROOT/'follow_trajectory.npz')
original = saved['q'].copy()
paint, indexes = saved['paint'].copy(), saved['csv_index'].copy()
model = pin.buildModelFromUrdf(str(ROOT/report['scene']))
data = model.createData()
fid = model.getFrameId('gun_nozzle')
lower, upper = model.lowerPositionLimit[2:8], model.upperPositionLimit[2:8]
goals = []
for q in original:
    pin.framesForwardKinematics(model,data,q)
    goals.append(data.oMf[fid].copy())

# 정지·계단 모양을 완화. 시각화용 경로이므로 정확한 속도나 가속도 계획은 아님.
filtered = gaussian_filter1d(original[1:,:2], sigma=6, axis=0, mode='nearest')
filtered[0] = original[1,:2]
strengths = (1.0, .75, .5, .25, .1, .03, 0.0)
candidate = original.copy()
used = []
worst_p, worst_r = 0., 0.
failed = None
for i in range(1,len(candidate)):
    accepted = False
    for strength in strengths:
        candidate[i,:2] = original[i,:2] + strength*(filtered[i-1]-original[i,:2])
        def err(arm):
            x = candidate[i].copy(); x[2:8] = arm
            pin.framesForwardKinematics(model,data,x)
            actual = data.oMf[fid]
            return np.r_[actual.translation-goals[i].translation,
                         pin.log3(actual.rotation.T @ goals[i].rotation)]
        seeds = [original[i,2:8], candidate[i-1,2:8]] if i > 1 else [original[i,2:8]]
        for seed in seeds:
            result = least_squares(err, np.clip(seed,lower+1e-8,upper-1e-8), bounds=(lower,upper),
                                   max_nfev=50, ftol=1e-8, xtol=1e-8, gtol=1e-8)
            e = err(result.x)
            if np.linalg.norm(e[:3]) <= .0001 and np.linalg.norm(e[3:]) <= np.deg2rad(.02):
                if i > 1 and np.max(np.abs(result.x-candidate[i-1,2:8])) > np.deg2rad(40):
                    continue
                candidate[i,2:8] = result.x
                worst_p = max(worst_p, float(np.linalg.norm(e[:3])))
                worst_r = max(worst_r, float(np.linalg.norm(e[3:])))
                accepted = True
                break
        if accepted:
            used.append(strength)
            break
    if not accepted:
        failed = i
        break
    if i % 200 == 0:
        print(f'평활 IK {i}/{len(candidate)-1}', flush=True)
if failed is not None:
    raise SystemExit(f'{failed}번 점에서 연속 IK 실패: 기존 결과를 그대로 유지합니다.')

times = [0.,max(15.,1.5*np.linalg.norm(candidate[1,:2]-candidate[0,:2])/.5)]
for i in range(2,len(candidate)):
    dx = candidate[i]-candidate[i-1]
    # 기존 IK가 만든 TCP 경로의 노드 거리를 이용한 시간 배분.
    dist = np.linalg.norm(goals[i].translation-goals[i-1].translation)
    dt = max(np.linalg.norm(dx[:2])/.5, np.max(np.abs(dx[2:8])/model.velocityLimit[2:8]), dist/.2, .02)
    times.append(times[-1]+float(dt))
old_steps = np.linalg.norm(np.diff(original[1:,:2],axis=0),axis=1)
new_steps = np.linalg.norm(np.diff(candidate[1:,:2],axis=0),axis=1)
report.update({'smoothing_strength_counts':{str(s):used.count(s) for s in strengths}, 'smoothing_sigma_points':6,
               'smoothing_seconds':time.perf_counter()-start,
               'baseline_planning_seconds':report['planning_seconds'],
               'planning_seconds':report['planning_seconds']+time.perf_counter()-start,
               'playback_seconds':times[-1],
               'prior_node_max_position_error_m':report['max_node_position_error_m'],
               'prior_node_max_orientation_error_deg':report['max_node_orientation_error_deg'],
               'smoothing_reference_max_position_error_m':worst_p,
               'smoothing_reference_max_orientation_error_deg':math.degrees(worst_r),
               'stationary_gantry_steps_before':int((old_steps<1e-6).sum()),
               'stationary_gantry_steps_after':int((new_steps<1e-6).sum()),
               'gantry_steps_total':len(new_steps),
               'collision_checked':False, 'interpolation_checked':False, 'acceleration_checked':False})
np.savez(ROOT/'follow_trajectory.npz',q=candidate,t=np.array(times),paint=paint,csv_index=indexes)
(ROOT/'follow_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)
