#!/usr/bin/env python3
"""기존 검토 경로의 한 도장 줄만 따로 재생하기 위한 파일을 생성한다."""
import argparse
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('row', type=int, nargs='?', default=6)
ap.add_argument('--source', default='out_gun_off_231_speed')
args = ap.parse_args()
source = root/args.source/'trajectory_bumper.npz'
with np.load(source) as z:
    data = {k:z[k] for k in z.files}
target = np.flatnonzero((data['row']==args.row) & (data['kind']=='target'))
if len(target)<2:
    raise SystemExit(f'{args.row}번 줄에 목표점이 충분하지 않습니다.')
first,last = int(target[0]),int(target[-1])
if not np.all(data['row'][first:last+1]==args.row):
    raise SystemExit('줄 안에 다른 줄 또는 줄 사이 이동이 섞여 있습니다.')
node_keys = ('X','t','tcp','paint','kind','row','csv_idx','bad_edge')
out = {k:(v[first:last+1].copy() if k in node_keys else v.copy()) for k,v in data.items()}
out['t'] -= out['t'][0]
destination = root/'out_single_row'
destination.mkdir(exist_ok=True)
np.savez(destination/'trajectory_bumper.npz', **out)
summary = {'row_index':args.row,'csv_points':len(target),
           'paint_points':int(np.sum(data['paint'][target])),
           'duration_s':float(out['t'][-1]),'source':str(source),
           'first_node':first,'last_node':last,
           'note':'원래 궤적의 일부. 처음 목표 자세에서 시작하며 줄 간 이동은 포함하지 않음.'}
(destination/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
