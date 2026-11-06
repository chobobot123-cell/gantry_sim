#!/usr/bin/env python3
"""첫점 기준 자세 방식의 전체 CSV 시험. 충돌/실제 제어 인증 없음."""
import json, time, math
from pathlib import Path
import numpy as np
import pinocchio as pin
from scipy.optimize import minimize
import plan_bumper as pb
from cell_model import Cell, DIR
out=DIR/'planner/out_whole_reference'; out.mkdir(exist_ok=True)
cell=Cell(DIR/'planner/paint_cell_collision_208.urdf',gun='off')
pb.CSV=DIR/'ES300h_front_bumper/painting_trajectory_adjusted.csv'
rows=pb.load_targets(cell)
prior=np.load(DIR/'planner/out_gun_off_231_speed/trajectory_bumper.npz')
ref=prior['X'][np.flatnonzero(prior['kind']=='target')[0],2:].copy()
q=ref.copy(); X=[]; tcp=[]; kinds=[]; rw=[]; ids=[]; paints=[]; failures=[]; iteration=[]
t0=time.perf_counter()
for row in rows:
    for k,T in enumerate(row['T']):
        pt,Rt=T[:3,3],T[:3,:3]
        def pose(qv): return cell.tcp(np.r_[0.,0.,qv])
        def res(qv):
            p,R=pose(qv); return np.r_[p[2]-pt[2],pin.log3(R.T@Rt)]
        def gb(qv):
            p,_=pose(qv); g=pt[:2]-p[:2]; return np.r_[g-cell.lower[:2],cell.upper[:2]-g]
        qprev=q.copy()
        sol=minimize(lambda v: np.sum((v-qprev)**2),qprev,method='SLSQP',bounds=list(zip(cell.lower[2:],cell.upper[2:])),constraints=[{'type':'eq','fun':res},{'type':'ineq','fun':gb}],options={'ftol':1e-10,'maxiter':100})
        q=sol.x; p,R=pose(q); g=pt[:2]-p[:2]
        err=res(q)
        if not sol.success or np.linalg.norm(err[:1])>1e-4 or np.linalg.norm(err[1:])>math.radians(.1) or np.min(gb(q))<-1e-6:
            failures.append({'row':int(row['row']),'point':k,'csv_idx':int(row['csv_idx'][k]),'reason':sol.message,'error':err.tolist()}); break
        X.append(np.r_[g,q]); tcp.append(pt); kinds.append('target'); rw.append(row['row']); ids.append(row['csv_idx'][k]); paints.append(row['paint'][k]); iteration.append(sol.nit)
    print(f"row {row['row']} points {len(X)}",flush=True)
    if failures: break
elapsed=time.perf_counter()-t0
report={'target_count':sum(len(r['T']) for r in rows),'solved':len(X),'paint_count':int(sum(paints)),'elapsed_s':elapsed,'failures':failures,'reference':'existing full-path first point (2.31 m scene); optimize previous posture sequentially','collision_checked':False}
if len(X)>1:
    X=np.array(X); tcp=np.array(tcp); rw=np.array(rw); paints=np.array(paints)
    diff=np.diff(X,axis=0); dist=np.linalg.norm(np.diff(tcp,axis=0),axis=1)
    dt=np.maximum.reduce([dist/.5,np.linalg.norm(diff[:,:2],axis=1)/.5,np.max(np.abs(diff[:,2:]),axis=1)/math.radians(120),np.full(len(diff),.025)])
    t=np.r_[0.,np.cumsum(dt)]
    report.update(duration_s=float(t[-1]),max_joint_step_deg=float(np.degrees(np.max(np.abs(diff[:,2:])))),max_xy_speed=float(np.max(np.linalg.norm(diff[:,:2],axis=1)/dt)),max_iterations=max(iteration))
    # 직접 연결은 실제 로봇 실행을 위한 안전한 전환 경로가 아니다.
    if not failures:
        np.savez(out/'trajectory_bumper.npz',X=X,t=t,tcp=tcp,kind=np.array(kinds),row=rw,csv_idx=np.array(ids),paint=paints,names=np.array(cell.names),tool_names=np.array(['gun_lift','gun_swing']),tool_values=np.array([0.,0.]))
(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
if failures: raise SystemExit(1)
