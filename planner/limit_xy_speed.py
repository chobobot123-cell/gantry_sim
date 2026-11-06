#!/usr/bin/env python3
"""기존 경로의 시간만 전역으로 늘려 XY 합성 평균속도를 제한한다."""
import hashlib
import json
import argparse
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('src', nargs='?', default='out_corrected')
ap.add_argument('dst', nargs='?', default='out_corrected_speed')
args = ap.parse_args()
src = root/args.src
dst = root/args.dst
dst.mkdir(exist_ok=True)
npz = sorted(src.glob('trajectory_*.npz'))[0]     # 부품 이름과 무관하게 (trajectory_<부품>.npz)
with np.load(npz) as data:
    arrays = {k:data[k] for k in data.files}
q,t = arrays['X'],arrays['t']
peak = float(np.max(np.linalg.norm(np.diff(q[:,:2],axis=0),axis=1)/np.diff(t)))
factor = max(1.,peak/.5)
arrays['t'] = t*factor
target = dst/npz.name
np.savez(target,**arrays)
val = json.loads((src/'validation.json').read_text())
val['time_scaling'] = {'factor':factor, 'pre_scale_peak_xy_m_s':peak,
                       'post_scale_peak_xy_m_s':peak/factor,
                       'meaning':'global uniform time scaling; XY velocity checked per linear interpolation segment'}
val['metrics']['cycle_time_s'] = float(arrays['t'][-1])
val['trajectory_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
(dst/'validation.json').write_text(json.dumps(val,indent=2,ensure_ascii=False)+'\n')
print(f'XY 최대속도 {peak:.3f} -> {peak/factor:.3f} m/s, 시간 배율 {factor:.3f}, 사이클 {arrays["t"][-1]:.1f} s')
