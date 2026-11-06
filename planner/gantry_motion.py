#!/usr/bin/env python3
"""저장된 궤적으로 "실제 겐트리가 이 움직임을 할 수 있는가"를 본다.
경로(관절값)는 그대로 두고, 겐트리 속도·가속도 한계만 바꿔 시간을 다시 붙인다.

사용법: gantry_motion.py OUTDIR [--vmax 0.8] [--amax 1.0] [--rows]
  --vmax : 겐트리 축별 최고 속도 [m/s] (주면 X, Y 각각에 적용)
  --amax : 겐트리 축별 최고 가속도 [m/s^2]
출력: 사이클, 도장 속도, 줄별 X·Y 이동량과 주축/보조축 비, 방향 전환 횟수, 축별 최고 속도·가속도
"""
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plan_bumper as pb  # noqa: E402
import plan_chain as pc  # noqa: E402

TURN = .005          # [m] 이만큼 넘게 되돌아가야 방향 전환 1회로 센다 (작은 떨림 제외)


def turns(x):
    """한 축 좌표열에서 TURN 넘게 되돌아간 방향 전환 횟수와, 되돌아간 거리(전환점 사이 구간)의 최댓값."""
    pts, d, ext = [x[0]], 0, x[0]
    for v in x[1:]:
        if d == 0:
            if abs(v - ext) > TURN:
                d, ext = (1 if v > ext else -1), v
        elif (v - ext) * d > 0:
            ext = v
        elif abs(v - ext) > TURN:
            pts.append(ext)
            d, ext = -d, v
    pts.append(ext)
    legs = np.abs(np.diff(pts))                               # 전환점 사이 구간 길이
    if len(legs) > 2:                                         # 되돌아간 거리 = 양쪽이 전환점인 안쪽 구간의 최댓값
        back = float(legs[1:-1].max())
    elif len(legs) == 2:
        back = float(legs.min())
    else:
        back = 0.0
    return len(pts) - 2, back


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--vmax', type=float, default=None)
    ap.add_argument('--amax', type=float, default=None)
    ap.add_argument('--rows', action='store_true', help='줄별 표 출력')
    a = ap.parse_args()
    v = json.load(open(Path(a.out) / 'validation.json'))
    wp = v.get('workpiece', 'bumper')
    z = np.load(Path(a.out) / f'trajectory_{wp}.npz')
    X, paint, kind, row = z['X'], z['paint'], z['kind'], z['row']
    t = z['t']
    if a.vmax or a.amax:                                     # 겐트리 한계를 바꿔 시간만 다시 붙인다
        pb.set_workpiece(wp)
        cell = pb._make(v['urdf'], v['gun'])
        for jn, val in v['tool_state'].items():
            cell.q[cell.model.joints[cell.model.getJointId(jn)].idx_q] = val
        pc.GANTRY_PER_AXIS = True
        if a.vmax:
            pb.GANTRY_SPEED = a.vmax
        pc.GANTRY_AMAX = a.amax
        t = pc.time_param_smooth(cell, X, paint)
    tcp = z['tcp']
    dt = np.diff(t)
    dG = np.diff(X[:, :2], axis=0)
    vg = dG / dt[:, None]
    ag = np.diff(vg, axis=0) / ((dt[1:] + dt[:-1]) / 2)[:, None]
    L = np.linalg.norm(np.diff(tcp, axis=0), axis=1)
    seg = (paint[:-1] == 1) & (paint[1:] == 1)
    sp = L / dt
    inrow = np.isin(kind, ('target', 'mid'))
    lim = f'겐트리 {a.vmax or pb.GANTRY_SPEED} m/s' + (f', {a.amax} m/s^2' if a.amax else ', 가속도 제한 없음') if (a.vmax or a.amax) else '저장된 시간'
    print(f'== {a.out} ({wp}) [{lim}]')
    print(f'  사이클 {t[-1]:.1f} s | 도장 {L[seg].sum():.2f} m 를 {dt[seg].sum():.1f} s (평균 {L[seg].sum() / dt[seg].sum():.2f} m/s), '
          f'목표 {pb.TCP_SPEED} m/s 로 간 도장 길이 {100 * L[seg][sp[seg] >= pb.TCP_SPEED * .99].sum() / L[seg].sum():.0f} %')
    print(f'  겐트리 최고 속도 X {np.abs(vg[:, 0]).max():.2f} / Y {np.abs(vg[:, 1]).max():.2f} m/s, 최고 가속도 X {np.abs(ag[:, 0]).max():.2f} / Y {np.abs(ag[:, 1]).max():.2f} m/s^2')
    if a.vmax or a.amax:                                     # 도장 구간마다 무엇이 속도를 정했는가
        jv = np.radians(120.0)
        lims = np.c_[L / pb.TCP_SPEED, np.abs(dG).max(1) / (a.vmax or pb.GANTRY_SPEED), np.abs(np.diff(X[:, 2:], axis=0)).max(1) / jv]
        which = np.where(dt > lims.max(1) * 1.02, 3, lims.argmax(1))
        names = ('건 끝 속도', '겐트리 속도', '팔 관절 속도', '가속도(겐트리·팔)')
        tt = [dt[seg & (which == j)].sum() for j in range(4)]
        print('  도장 시간을 정한 한계: ' + ', '.join(f'{nm} {100 * x / max(sum(tt), 1e-9):.0f} %' for nm, x in zip(names, tt)))
    tg = np.flatnonzero(kind == 'target')                    # 시간 내역: 홈 -> 첫 점 / 줄 안 / 줄 사이 / 마지막 점 -> 홈
    seg_in = inrow[:-1] & inrow[1:] & (row[:-1] == row[1:])
    t_home = t[tg[0]] + (t[-1] - t[tg[-1]])
    t_in = dt[seg_in].sum()
    print(f'  시간 내역: 줄 안 {t_in:.1f} s (그중 분사 {dt[seg].sum():.1f} s) | 줄 사이 이동 {t[-1] - t_home - t_in:.1f} s | 홈 왕복 {t_home:.1f} s')
    tot = np.zeros(2); totc = 0.0; nturn = np.zeros(2, int); worst_back = np.zeros(2); out = []
    for r in sorted(set(row[inrow])):
        i = np.flatnonzero(inrow & (row == r))
        if len(i) < 2:
            continue
        g = X[i, :2]
        tr = np.abs(np.diff(g, axis=0)).sum(0)
        m = int(np.argmax(tr))
        tx, bx = turns(g[:, 0]); ty, by = turns(g[:, 1])
        nturn += (tx, ty); worst_back = np.maximum(worst_back, (bx, by))
        tot += tr; totc += tr[1 - m]
        out.append((r, len(i), tr[0], tr[1], 'XY'[m], tr[1 - m] / max(tr[m], 1e-9), np.ptp(g[:, 1 - m]), tx, ty, t[i[-1]] - t[i[0]]))
    print(f'  줄 안 겐트리 이동: X {tot[0]:.2f} m, Y {tot[1]:.2f} m | 보조축(줄마다 덜 움직인 축) 이동 합 {totc:.2f} m = 주축의 {100 * totc / max(tot.sum() - totc, 1e-9):.0f} % '
          f'| 방향 전환({TURN * 1000:.0f} mm 넘게 되돌아감) X {nturn[0]}회, Y {nturn[1]}회, 최대 되돌아감 X {worst_back[0] * 1000:.0f} / Y {worst_back[1] * 1000:.0f} mm')
    if a.rows:
        print('  줄  점수   X이동  Y이동  주축  보조/주축  보조축폭  전환X  전환Y  시간')
        for r, n, x_, y_, m, ratio, span, tx, ty, dur in out:
            print(f'  {r:2d}  {n:4d}  {x_:5.2f}  {y_:5.2f}   {m}    {100 * ratio:5.0f} %   {span * 1000:5.0f} mm  {tx:4d}  {ty:4d}  {dur:5.1f} s')


if __name__ == '__main__':
    main()
