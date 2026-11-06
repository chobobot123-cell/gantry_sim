#!/usr/bin/env python3
"""근사 충돌과 IK 분기를 지키며 후보 관절 궤적을 평활화하는 실험."""
import json, math, sys, time
from pathlib import Path
import numpy as np
import pinocchio as pin
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cell_model import Cell
from plan_decoupled import Dec
import plan_chain as pc

ROOT=Path(__file__).resolve().parent
src=ROOT/'out_video_rule_speed'
out=ROOT/'out_video_smooth'
out.mkdir(exist_ok=True)
start=time.perf_counter()
val=json.loads((src/'validation.json').read_text())
with np.load(src/'trajectory_bumper.npz') as z: data={k:z[k] for k in z.files}
old=data['X']; kind=data['kind']; c=Cell(val['urdf'],gun=val['gun']); D=Dec(c)
for jn, v in val['tool_state'].items(): c.q[c.model.joints[c.model.getJointId(jn)].idx_q] = v
sig=tuple(val['metrics']['branch_signatures'][0]); D.sig=sig
goals=[]
for x in old:
 p,R=c.tcp(x); goals.append((p.copy(),R.copy()))
filtered=gaussian_filter1d(old,sigma=3,axis=0,mode='nearest')
new=old.copy(); strengths=[]; failed=[]
lo,hi=c.lower[2:],c.upper[2:]
for i in range(1,len(old)-1):
 if kind[i]=='home': continue
 for s in (1.,.5,.25,.1,0.):
  xy=old[i,:2]+s*(filtered[i,:2]-old[i,:2])
  p,R=goals[i]
  def error(q):
   pt,Rt=c.tcp(np.r_[xy,q]); return np.r_[pt-p,pin.log3(Rt.T@R)]
  seed=old[i,2:]+s*(filtered[i,2:]-old[i,2:])
  fit=least_squares(error,np.clip(seed,lo+1e-9,hi-1e-9),bounds=(lo,hi),max_nfev=50,ftol=1e-9,xtol=1e-9,gtol=1e-9)
  err=error(fit.x); x=np.r_[xy,fit.x]
  if np.linalg.norm(err[:3])>.0001 or np.linalg.norm(err[3:])>math.radians(.02): continue
  if max(np.max(abs(x[2:]-new[i-1,2:])),np.max(abs(x[2:]-old[i+1,2:])))>math.radians(12): continue
  if pc.signature(c,x)!=sig and kind[i] in ('target','mid'): continue
  if not D.free(x,.015): continue
  if kind[i]=='target' and (D.elbow_side(x)<.03 or D.singular(x)): continue
  new[i]=x; strengths.append(s); break
 else:
  failed.append(i); new[i]=old[i]
 if i%200==0: print('점별 평활 검사',i,'/',len(old),flush=True)

# 원래 구간별 시간을 유지하되 큰 관절 속도 변화에 대비해 균일하게 시간을 늘린다.
# 경로 자체도 평활화하지만 저크나 가속도 연속성은 검증하지 않는다.
times=data['t']*2.0
data['X']=new; data['t']=times
data['tcp']=np.array([c.tcp(x)[0] for x in new])
np.savez(out/'trajectory_bumper.npz',**data)
delta0=np.diff(old[1:-1],axis=0);delta1=np.diff(new[1:-1],axis=0)
summary={'method':'Gaussian target + fixed-gantry 6-axis IK with collision, branch, and elbow tests',
         'strength_counts':{str(s):strengths.count(s) for s in (1.,.5,.25,.1,0.)},'fallback_count':len(failed),
         'planning_s':time.perf_counter()-start,'time_multiplier':2.0,
         'joint_delta_change_before_rad':float(np.linalg.norm(np.diff(delta0[:,2:],axis=0),axis=1).sum()),
         'joint_delta_change_after_rad':float(np.linalg.norm(np.diff(delta1[:,2:],axis=0),axis=1).sum()),
         'cycle_s':float(times[-1])}
val['experimental_smoothing']=summary
val['metrics']['cycle_time_s']=float(times[-1])
(out/'validation.json').write_text(json.dumps(val,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
