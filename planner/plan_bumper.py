#!/usr/bin/env python3
"""범퍼 도장 경로 — 분리(decoupled) 방식: 겐트리 규칙 배치 -> 팔 6축 자코비안 IK.

  1) 겐트리: 목표 자세만으로 정해지는 '손목 중심'(W)을 기준으로, 어깨->W 거리가
     R_des 가 되도록 W 주위 방위각 phi 에 팔 베이스를 둔다. phi 기본값은 분사 방향의 뒤.
     (방위각 오프셋, R_des) 조합은 한 패스 안에서 고정 -> 분사 중 겐트리가 튀지 않는다.
  2) 팔: 겐트리를 고정한 채 6축만 DLS(감쇠 최소자승) 자코비안 반복. 직전 해로 워밍스타트.
     최적화(whole-body) 없음.
  3) 노드 충돌 여유 확인, 노드 사이 보간 정확도 확인(필요시 중점 삽입), 구간 연속 인증.
  4) 패스 사이 / 홈 이동: 직선 -> 후퇴·이동·접근 -> 우회 경유점 순으로 인증될 때까지.

사용법:  plan_bumper.py [--urdf PATH] [--out DIR] [--no-cert]
"""
import os
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

from cell_model import ARM, DIR, GANTRY, GUN, Cell

CSV = DIR / 'ES300h_front_bumper' / 'painting_trajectory.csv'
WP = 'bumper'                                        # 계획 대상 작업물 (URDF 의 <WP>_link, world_base-<WP>)
WP_FOLDER = {'bumper': 'ES300h_front_bumper', 'fender': 'ES300h_left_fender',
             'rfender': 'NX350h_right_fender', 'hood': 'ES300h_hood', 'trunk': 'ES300h_trunk'}


def set_workpiece(name):
    """계획 대상 작업물을 바꾼다 (경로 CSV 와 셀의 작업물 링크). 워커를 만들기 전에 부른다."""
    global WP, CSV
    WP, CSV = name, Path(os.environ.get('BUMPER_CSV', str(DIR / WP_FOLDER[name] / 'painting_trajectory.csv'))) if name == 'bumper' else DIR / WP_FOLDER[name] / 'painting_trajectory.csv'
    if os.environ.get('WP_CSV'):                       # 작업물 공통: 경로 CSV 를 다른 파일로 (예: 줄 순서를 바로잡은 사본)
        CSV = Path(os.environ['WP_CSV'])


POS_TOL, ANG_TOL = 1e-3, math.radians(.2)           # 판정 기준 1 mm / 0.2 deg
IK_POS, IK_ANG = 2e-5, 2e-5                          # IK 수렴 (기준보다 훨씬 엄격)
NODE_CLEAR = .015                                    # 노드 최소 여유 (계획용)
CERT_CLEAR = .005                                    # 구간 인증 하한
MAX_JOINT_STEP = math.radians(25)                    # 40 mm 스텝당 팔 관절 변화 상한 (분기 튐 방지)
SHOULDER_Z_OFF = .195                                # 팔 베이스 -> 어깨 높이
SHOULDER_R = .039                                    # 어깨 반경 오프셋
DPHI = [0, 30, -30, 60, -60, 90, -90, 120, -120]     # 방위각 오프셋 후보 [deg]
RDES = [.85, .75, .95, .65]                          # 어깨->손목 목표 거리 후보 [m]
HOME_G = None                                        # None = 겐트리 하한 모서리 (run_bumper.sh 시작 위치와 같음)
TCP_SPEED, GANTRY_SPEED = .5, 1.5                    # m/s (겐트리 = URDF velocity)
SMOOTH = 15                                          # 패스 내 겐트리 이동평균 창 [점]


# ---------------------------------------------------------------- 목표
def quat_R(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_targets(cell):
    d = np.genfromtxt(CSV, delimiter=',', names=True)
    Rw, tw = cell.T_wp[:3, :3], cell.T_wp[:3, 3]
    rows = []
    for k in np.unique(d['row_index'].astype(int)):
        s = np.flatnonzero(d['row_index'].astype(int) == k)
        T = np.zeros((len(s), 4, 4))
        for n, i in enumerate(s):
            T[n] = np.eye(4)
            T[n, :3, :3] = Rw @ quat_R(d['orientation_x'][i], d['orientation_y'][i],
                                       d['orientation_z'][i], d['orientation_w'][i])
            T[n, :3, 3] = Rw @ [d['position_x'][i], d['position_y'][i], d['position_z'][i]] + tw
        rows.append({'row': int(k), 'T': T, 'paint': d['paint'][s].astype(int), 'csv_idx': s})
    return rows


def interp_pose(A, B, u):
    T = np.eye(4)
    T[:3, 3] = (1 - u) * A[:3, 3] + u * B[:3, 3]
    T[:3, :3] = A[:3, :3] @ pin.exp3(u * pin.log3(A[:3, :3].T @ B[:3, :3]))
    return T


def pose_err(p, R, T):
    return float(np.linalg.norm(T[:3, 3] - p)), float(np.linalg.norm(pin.log3(R.T @ T[:3, :3])))


# ---------------------------------------------------------------- IK (팔 6축, 겐트리 고정)
class ArmIK:
    def __init__(self, cell):
        self.c = cell
        self.lo, self.hi = cell.lower[2:], cell.upper[2:]
        # 손목 중심(link_5 원점)의 노즐 프레임 좌표: 관절값과 무관한 상수
        x = np.zeros(8)
        pn, Rn = cell.tcp(x)
        pw, _ = cell.frame(x, 'link_5')
        self.p_wc = Rn.T @ (pw - pn)
        pb, _ = cell.frame(x, 'base_link')
        self.base_off = pb[:2] - x[:2]               # 겐트리 -> joint_1 축 XY
        self.base_z = pb[2]
        self.evals = 0

    def solve(self, g, T, q0, iters=80):
        q = np.array(q0, float)
        pt, Rt = T[:3, 3], T[:3, :3]
        for _ in range(iters):
            J, p, R = self.c.jacobian_tcp(np.r_[g, q])
            self.evals += 1
            e = np.r_[pt - p, R @ pin.log3(R.T @ Rt)]
            if np.linalg.norm(e[:3]) < IK_POS and np.linalg.norm(e[3:]) < IK_ANG:
                return q, True
            Ja = J[:, 2:]
            lam2 = 1e-6 + 1e-3 * max(0.0, .05 - np.linalg.svd(Ja, compute_uv=False)[-1])
            dq = Ja.T @ np.linalg.solve(Ja @ Ja.T + lam2 * np.eye(6), e)
            n = np.linalg.norm(dq)
            if n > .4:
                dq *= .4 / n
            q = np.clip(q + dq, self.lo, self.hi)
        return q, False

    def seeds(self, g, W):
        """패스 시작점용 시드: joint_1 을 손목 방향 +-(0, pi) 에, 팔꿈치/손목 분기 조합."""
        B = np.asarray(g) + self.base_off
        az = math.atan2(W[1] - B[1], W[0] - B[0])
        out = []
        for j1 in np.arange(-math.pi, math.pi, math.pi / 4) + az % (math.pi / 4):
            for j2, j3 in ((-.4, 1.8), (.4, -1.8), (-.9, 1.2), (.9, -1.2)):
                for j5 in (1.3, -1.3):
                    out.append(np.array([j1, j2, j3, 0.0, j5, 0.0]))
        return out


# ---------------------------------------------------------------- 겐트리 규칙
class GantryRule:
    def __init__(self, cell, ik):
        self.c, self.ik = cell, ik
        self.glo, self.ghi = cell.lower[:2], cell.upper[:2]
        self.shoulder_z = ik.base_z - SHOULDER_Z_OFF

    def wrist(self, T):
        return T[:3, 3] + T[:3, :3] @ self.ik.p_wc

    def path(self, Ts, dphi, rdes, smooth=None):
        """패스 전체 겐트리 궤적: (dphi, rdes) 고정, 방위각은 분사 방향 뒤를 기준으로 연속."""
        smooth = SMOOTH if smooth is None else smooth
        W = np.array([self.wrist(T) for T in Ts])
        s = np.array([T[:3, 2] for T in Ts])                  # 분사축(+Z)
        sh = -s[:, :2]
        nrm = np.linalg.norm(sh, axis=1)
        # 분사축이 거의 수직이면 방위각이 불안정 -> 패스 평균 방향으로 대체
        mean = sh[nrm > .3].mean(0) if np.any(nrm > .3) else np.array([1.0, 0.0])
        sh[nrm <= .3] = mean
        phi = np.unwrap(np.arctan2(sh[:, 1], sh[:, 0])) + math.radians(dphi)
        dz = self.shoulder_z - W[:, 2]
        h = np.sqrt(np.maximum(rdes ** 2 - dz ** 2, .01)) + SHOULDER_R
        B = W[:, :2] + h[:, None] * np.c_[np.cos(phi), np.sin(phi)]
        G = np.clip(B - self.ik.base_off, self.glo, self.ghi)
        if smooth > 1 and len(G) > smooth:                    # 이동평균 (끝은 짧은 창)
            k = smooth // 2
            G = np.array([G[max(0, i - k):i + k + 1].mean(0) for i in range(len(G))])
        return np.clip(G, self.glo, self.ghi)


# ---------------------------------------------------------------- 패스 추종
class Planner:
    def __init__(self, cell, log=print):
        self.c = cell
        self.ik = ArmIK(cell)
        self.rule = GantryRule(cell, self.ik)
        self.log = log
        self.stats = {'pass_attempts': 0, 'ik_calls': 0}

    def clear(self, x):
        return float(self.c.distances(x).min())

    def track(self, Ts, G, q0):
        """G 를 따라 팔만 추종. 실패 시 (None, 실패 인덱스)."""
        X, q = [], q0
        for i, (T, g) in enumerate(zip(Ts, G)):
            qn, ok = self.ik.solve(g, T, q)
            self.stats['ik_calls'] += 1
            if not ok or (i and np.abs(qn - q).max() > MAX_JOINT_STEP):
                return None, i, 'ik'
            x = np.r_[g, qn]
            if self.clear(x) < NODE_CLEAR:
                return None, i, 'collision'
            X.append(x)
            q = qn
        return np.array(X), len(Ts), 'ok'

    def start_solutions(self, g, T):
        sols = []
        for s in self.ik.seeds(g, self.rule.wrist(T)):
            q, ok = self.ik.solve(g, T, s, iters=60)
            if ok and not any(np.abs(q - p).max() < 1e-3 for p in sols):
                sols.append(q)
        scored = []
        for q in sols:
            d = self.clear(np.r_[g, q])
            if d >= NODE_CLEAR:
                # 선호: 여유 크고, 손목(joint_5)이 특이점(0)에서 멀고, 관절이 가운데
                scored.append((-min(d, .10) - .05 * min(abs(math.sin(q[4])), .5)
                               + .01 * np.abs(q).sum(), q))
        return [q for _, q in sorted(scored, key=lambda t: t[0])]

    def plan_pass(self, row, prev_x=None):
        Ts = row['T']
        best_fail = (0, None)
        cands = [(dp, r) for r in RDES for dp in DPHI]
        # 직전 패스와 같은 조합을 먼저 시도 (겐트리/분기 연속성)
        if prev_x is not None and getattr(self, 'last_cand', None) in cands:
            cands.remove(self.last_cand)
            cands.insert(0, self.last_cand)
        for dp, r in cands:
            G = self.rule.path(Ts, dp, r)
            starts = self.start_solutions(G[0], Ts[0])
            if prev_x is not None and starts:            # 직전 패스 끝 자세와 가까운 분기 우선
                starts.sort(key=lambda q: np.abs(q - prev_x[2:]).max())
            for q0 in starts[:4]:
                self.stats['pass_attempts'] += 1
                X, n, why = self.track(Ts, G, q0)
                if X is not None:
                    self.last_cand = (dp, r)
                    return X, {'dphi': dp, 'rdes': r}
                if n > best_fail[0]:
                    best_fail = (n, (dp, r, why))
        return None, {'reached': best_fail[0], 'of': len(Ts), 'last': best_fail[1]}


def shifted_urdf(urdf, dx, dy, dz):
    import re
    s = Path(urdf).read_text(encoding='utf-8')
    m = re.search(r'(<joint name="world_base-bumper".*?<origin xyz=")([^"]+)(")', s, re.S)
    v = np.array(list(map(float, m.group(2).split()))) + [dx, dy, dz]
    out = DIR / 'planner' / 'cache' / f'bumper_shift_{dx:+.2f}_{dy:+.2f}_{dz:+.2f}.urdf'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(s[:m.start(2)] + ' '.join(f'{x:.5f}' for x in v) + s[m.end(2):], encoding='utf-8')
    return str(out)


def diagnose(P, rows):
    """각 줄 첫 점: IK 해 개수 / 충돌 여유 / 가장 가까운 쌍."""
    for row in rows:
        T = row['T'][0]
        for dp, r in [(0, .85), (60, .85), (-60, .85), (90, .75)]:
            G = P.rule.path(row['T'], dp, r)
            n_ik, best = 0, (-1, None)
            for sd in P.ik.seeds(G[0], P.rule.wrist(T)):
                q, ok = P.ik.solve(G[0], T, sd, iters=60)
                if ok:
                    n_ik += 1
                    d = P.c.distances(np.r_[G[0], q])
                    if d.min() > best[0]:
                        best = (d.min(), P.c.pair_names(int(np.argmin(d))))
            print(f"row {row['row']:2d} dphi {dp:4d} r {r}: g=({G[0][0]:+.2f},{G[0][1]:+.2f}) ik {n_ik:2d}  best clear {best[0]:+.3f} {best[1]}")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- 보간 정확도 / 인증 / 이동
class Finisher:
    """패스 노드 사이 보간 검사(중점 삽입), 구간 연속 인증, 패스 간/홈 이동 계획, 시간 배정."""

    def __init__(self, P):
        self.P, self.c, self.ik = P, P.c, P.ik
        self.inserted = 0
        self.interp_max = [0.0, 0.0]
        self.cert_min = float('inf')
        self.cert_evals = 0
        self.uncertified = []

    # ---- 분사/경로 구간: 카테시안 정확도
    def _interp_ok(self, xa, xb, Ta, Tb):
        worst = [0.0, 0.0]
        for u in (.25, .5, .75):
            p, R = self.c.tcp((1 - u) * xa + u * xb)
            e = pose_err(p, R, interp_pose(Ta, Tb, u))
            worst = [max(worst[0], e[0]), max(worst[1], e[1])]
        return worst[0] <= POS_TOL and worst[1] <= ANG_TOL, worst

    def refine(self, xa, xb, Ta, Tb, depth=0):
        ok, w = self._interp_ok(xa, xb, Ta, Tb)
        if ok or depth >= 8:
            self.interp_max = [max(self.interp_max[0], w[0]), max(self.interp_max[1], w[1])]
            return [] if ok else None
        Tm = interp_pose(Ta, Tb, .5)
        gm = (xa[:2] + xb[:2]) / 2                                   # 겐트리는 선형, 팔만 IK
        qm, conv = self.ik.solve(gm, Tm, (xa[2:] + xb[2:]) / 2)
        xm = np.r_[gm, qm]
        if not conv or self.P.clear(xm) < NODE_CLEAR:
            return None
        left = self.refine(xa, xm, Ta, Tm, depth + 1)
        right = self.refine(xm, xb, Tm, Tb, depth + 1)
        if left is None or right is None:
            return None
        self.inserted += 1
        return left + [(xm, Tm)] + right

    def certify(self, xa, xb):
        ok, lo, n = self.c.certify_edge(xa, xb, CERT_CLEAR)
        self.cert_evals += n
        return ok, lo

    # ---- 이동 구간
    def retract(self, x, d, follow=False):
        """TCP 를 분사축 반대로 d 만큼 뒤로.
        follow=False: 겐트리 고정, 팔만.  follow=True: 겐트리가 TCP 수평 이동만큼 같이 빠지고
        (리밋에서 잘리면) 나머지는 팔이 푼다 -> 접힌 자세에서도 후퇴 가능."""
        p, R = self.c.tcp(x)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, p - d * R[:, 2]
        g = x[:2]
        if follow:
            g = np.clip(g + (T[:2, 3] - p[:2]), self.c.lower[:2], self.c.upper[:2])
        q, ok = self.ik.solve(g, T, x[2:])
        xr = np.r_[g, q]
        return xr if ok and self.P.clear(xr) >= NODE_CLEAR else None

    def _chain_ok(self, chain):
        los = []
        for a, b in zip(chain, chain[1:]):
            ok, lo = self.certify(a, b)
            if not ok:
                return None
            los.append(lo)
        return min(los) if los else float('inf')

    def transit(self, xa, xb):
        """인증될 때까지 후보 경로를 순서대로 시도. 불안전 폴백 없음."""
        cands = [('direct', [xa, xb])]
        ylo, yhi = self.c.lower[1], self.c.upper[1]
        for follow in (False, True):
            for d in (.10, .20, .30):
                ra, rb = self.retract(xa, d, follow), self.retract(xb, d, follow)
                if ra is None or rb is None:
                    continue
                tag = f"retract{d:.2f}{'-gantry' if follow else ''}"
                cands += [(tag, [xa, ra, rb, xb]),
                          (f'{tag}+gantry-first', [xa, ra, np.r_[rb[:2], ra[2:]], rb, xb]),
                          (f'{tag}+arm-first', [xa, ra, np.r_[ra[:2], rb[2:]], rb, xb])]
                for yd in (ylo, yhi):                               # 범퍼 옆으로 돌아가기
                    for qv in (ra[2:], rb[2:]):
                        cands.append((f'{tag}+detour_y{yd:+.2f}',
                                      [xa, ra, np.r_[ra[0], yd, qv], np.r_[rb[0], yd, qv], rb, xb]))
        for name, chain in cands:
            if any(self.P.clear(x) < NODE_CLEAR for x in chain[1:-1]):
                continue
            if not all(self.sampled_ok(u, v) for u, v in zip(chain, chain[1:])):
                continue                                             # 싼 샘플 검사로 먼저 거른다
            lo = self._chain_ok(chain)
            if lo is not None:
                self.cert_min = min(self.cert_min, lo)
                return name, chain[1:-1]
        for seed in range(6):                                        # lazy RRT: 최종 경로만 인증
            chain = self.rrt_connect(xa, xb, seed=seed)
            if chain is None:
                continue
            lo = self._chain_ok(chain)
            if lo is not None:
                self.cert_min = min(self.cert_min, lo)
                return f'rrt-connect(seed {seed}, {len(chain) - 2} via)', chain[1:-1]
        return None, []

    def sampled_ok(self, a, b, res=.05):
        """간선 위 샘플 검사 (인증 아님, 후보 거르기용)."""
        n = max(3, int(np.ceil(np.max(np.abs(b - a) / np.r_[.05, .05, np.full(6, .08)]) )))
        for u in np.linspace(0, 1, n + 2)[1:-1]:
            if self.c.distances((1 - u) * a + u * b, near=CERT_CLEAR + .002).min() < CERT_CLEAR + .002:
                return False
        return True

    def rrt_connect(self, xa, xb, iters=4000, seed=0):
        """양방향 RRT (8D). 노드는 여유 >= NODE_CLEAR, 간선은 연속 인증 통과만 허용."""
        rng = np.random.default_rng(seed)
        scale = np.r_[.30, .30, np.full(6, .6)]                      # 한 스텝 최대 (m, rad)
        lo = np.maximum(self.c.lower, np.minimum(xa, xb) - np.r_[.8, .8, np.full(6, 1.5)])
        hi = np.minimum(self.c.upper, np.maximum(xa, xb) + np.r_[.8, .8, np.full(6, 1.5)])
        lo[:2], hi[:2] = self.c.lower[:2], self.c.upper[:2]         # 겐트리는 전 행정
        trees = [([xa], [-1]), ([xb], [-1])]

        def steer(frm, to):
            d = to - frm
            k = np.max(np.abs(d) / scale)
            return to if k <= 1 else frm + d / k

        def extend(tree, target):
            V, par = tree
            A = np.array(V)
            i = int(np.argmin(np.max(np.abs(A - target) / scale, axis=1)))
            new = steer(V[i], target)
            if self.P.clear(new) < NODE_CLEAR or not self.sampled_ok(V[i], new):
                return None
            V.append(new)
            par.append(i)
            return len(V) - 1

        def path(tree, i):
            V, par = tree
            out = []
            while i >= 0:
                out.append(V[i])
                i = par[i]
            return out

        for it in range(iters):
            s = rng.uniform(lo, hi) if rng.random() > .1 else trees[1][0][0]
            a, b = trees[it % 2], trees[1 - it % 2]
            ia = extend(a, s)
            if ia is None:
                continue
            target = a[0][ia]
            while True:                                              # connect
                ib = extend(b, target)
                if ib is None:
                    break
                if np.allclose(b[0][ib], target):
                    pa, pb = path(a, ia), path(b, ib)
                    chain = pa[::-1] + pb[1:] if a is trees[0] else pb[::-1] + pa[1:]
                    return self.shortcut(chain, rng)
        return None

    def shortcut(self, chain, rng, tries=80):
        chain = list(chain)
        for _ in range(tries):
            if len(chain) <= 2:
                break
            i, j = sorted(rng.choice(len(chain), 2, replace=False))
            if j - i < 2:
                continue
            if self.sampled_ok(chain[i], chain[j]) and self.certify(chain[i], chain[j])[0]:
                chain = chain[:i + 1] + chain[j:]
        return chain

    def home_pose(self, x_first):
        """홈: 겐트리는 하한 모서리(HOME_G)부터, 안 되면 셀 중앙 쪽으로 조금씩.
        팔은 첫 점에서 후퇴한 자세 -> 몸쪽으로 접은 표준 자세 순. 충돌 없는 첫 조합."""
        arms = []
        for d in (.20, .30, .10, 0.0):
            r = self.retract(x_first, d) if d else x_first
            if r is not None:
                arms.append(r[2:])
        j1 = x_first[2]
        arms += [np.array([j1 + dj, -.3, 1.6, 0.0, 1.2, 0.0]) for dj in (0, math.pi / 2, -math.pi / 2, math.pi)]
        g0 = self.c.lower[:2] if HOME_G is None else np.asarray(HOME_G, float)
        center = (self.c.lower[:2] + self.c.upper[:2]) / 2
        for k in range(11):                                          # 모서리 -> 중앙 방향 0, 10 %, ..., 100 %
            g = g0 + (center - g0) * k / 10
            for q in arms:
                h = np.r_[g, q]
                if self.P.clear(h) >= NODE_CLEAR:
                    return h
        raise RuntimeError('no collision-free home posture')


POOL = None
_W = {}


def _make(urdf, gun='off', gantry=None):
    cell = Cell(urdf, name=WP, gun=gun, gantry=gantry)
    ng = len(cell.gantry)
    # 캐리지 박스 <-> 팔 베이스: 볼트로 고정된 조립부. 관절과 무관하게 항상 겹치는 쌍만 제외한다.
    x0 = np.r_[(cell.lower[:2] + cell.upper[:2]) / 2, cell.lower[2:ng], np.zeros(6)]
    d0 = cell.distances(x0, near=np.inf)
    base = [i for i in range(len(cell.ia))
            if {n.rsplit('_', 1)[0] for n in cell.pair_names(i)} == {'gantry_y_link', 'base_link'} and d0[i] < .005]
    cell.exclude_pairs(base, 'bolted arm mount (carriage box encloses arm base flange)')
    # 상하축 램(gantry_z_link): 캐리지를 관통해 미끄러지고 팔 베이스에 볼트 고정 -> 구조상 겹치는 쌍만 제외 (범퍼/프레임과는 검사)
    if any(cell.model.names[j] == 'gantry_z' for j in range(cell.model.njoints)):
        xs = [x0] if ng == 2 else [np.r_[x0[:2], cell.lower[2:ng], np.zeros(6)], np.r_[x0[:2], cell.upper[2:ng], np.zeros(6)]]
        ds = [cell.distances(x, near=np.inf) for x in xs]
        ram = [i for i in range(len(cell.ia))
               if 'gantry_z_link' in {n.rsplit('_', 1)[0] for n in cell.pair_names(i)}
               and {n.rsplit('_', 1)[0] for n in cell.pair_names(i)} <= {'gantry_z_link', 'gantry_y_link', 'gantry_x_link', 'base_link'}
               and min(d[i] for d in ds) < .005]
        cell.exclude_pairs(ram, 'Z-axis ram slides through carriage, bolted to arm base')
    return cell


def _winit(urdf, gun='off'):
    import warnings
    warnings.filterwarnings('ignore')
    _W['F'] = Finisher(Planner(_make(urdf, gun), log=lambda *a: None))


def _transit_local(F, xa, xb):
    F.cert_min, F.cert_evals = float('inf'), 0
    name, via = F.transit(xa, xb)
    return name, via, F.cert_min, F.cert_evals


def _w_transit(xa, xb):
    return _transit_local(_W['F'], xa, xb)


def _w_certify(chunk):
    F = _W['F']
    return [F.c.certify_edge(a, b, CERT_CLEAR) for a, b in chunk]


def time_param(cell, X):
    t = [0.0]
    for a, b in zip(X, X[1:]):
        pa, _ = cell.tcp(a)
        pb, _ = cell.tcp(b)
        dt = max(np.linalg.norm(pb - pa) / TCP_SPEED,
                 np.linalg.norm(b[:2] - a[:2]) / GANTRY_SPEED,
                 float(np.max(np.abs(b - a) / cell.vel)), 1e-3)
        t.append(t[-1] + dt)
    return np.array(t)


def run_full(a, cell, P, rows, t_load):
    F = Finisher(P)
    T = {}
    t0 = time.perf_counter()
    passes, prev = [], None
    for row in rows:
        X, info = P.plan_pass(row, prev)
        if X is None:
            raise SystemExit(f"row {row['row']} failed: {info}")
        passes.append((row, X, info))
        prev = X[-1]
    T['passes'] = time.perf_counter() - t0

    t1 = time.perf_counter()
    nodes = []          # (x, kind, paint, row, csv_idx)
    for row, X, info in passes:
        for i in range(len(X)):
            nodes_i = [(X[i], 'target', int(row['paint'][i]), row['row'], int(row['csv_idx'][i]))]
            if i + 1 < len(X):
                mids = F.refine(X[i], X[i + 1], row['T'][i], row['T'][i + 1])
                if mids is None:
                    raise SystemExit(f"row {row['row']} edge {i}: interpolation refine failed")
                paint_mid = int(row['paint'][i] and row['paint'][i + 1])
                nodes_i += [(xm, 'mid', paint_mid, row['row'], -1) for xm, _ in mids]
            nodes += [nodes_i]
    T['refine'] = time.perf_counter() - t1

    # ---- 병렬: 패스 내부 구간 인증 + 이동 구간 계획 (서로 독립)
    t2 = time.perf_counter()
    flat = [n for grp in nodes for n in grp]
    home = F.home_pose(passes[0][1][0])
    ends = [home] + [x for _, Xp, _ in passes for x in (Xp[0], Xp[-1])] + [home]
    tpairs = [(ends[2 * i], ends[2 * i + 1]) for i in range(len(passes) + 1)]
    epairs = [(flat[k][0], flat[k + 1][0]) for k in range(len(flat) - 1) if flat[k][3] == flat[k + 1][3]]
    chunks = [epairs[i:i + 24] for i in range(0, len(epairs), 24)]
    if POOL is not None:
        tr_async = [POOL.apply_async(_w_transit, (xa, xb)) for xa, xb in tpairs]   # 긴 일 먼저
        ce_async = POOL.map_async(_w_certify, chunks)
        tres = [r.get() for r in tr_async]
        cres = [x for c in ce_async.get() for x in c]
    else:
        tres = [_transit_local(F, xa, xb) for xa, xb in tpairs]
        cres = [F.certify(xa, xb) + (0,) for xa, xb in epairs]
    for ok, lo, n in cres:
        F.cert_evals += n
        if ok:
            F.cert_min = min(F.cert_min, lo)
        else:
            F.uncertified.append(('pass-edge', lo))
    seq = [(home, 'home', 0, -1, -1)]
    transits = []
    for pi, (name, via, lo, n) in enumerate(tres):
        into = passes[pi][0]['row'] if pi < len(passes) else 'home'
        if name is None:
            raise SystemExit(f'transit into row {into} not certified')
        F.cert_min, F.cert_evals = min(F.cert_min, lo), F.cert_evals + n
        transits.append({'into_row': into, 'strategy': name, 'via': len(via)})
        seq += [(v, 'transit', 0, -1, -1) for v in via]
        if pi < len(passes):
            seq += [n_ for n_ in flat if n_[3] == passes[pi][0]['row']]
    seq += [(home, 'home', 0, -1, -1)]
    T['certify+transit (parallel)'] = time.perf_counter() - t2

    X = np.array([n[0] for n in seq])
    times = time_param(cell, X)
    # 도장 구간 TCP 속도: 겐트리/관절 속도 한계 때문에 목표(0.5 m/s)보다 느려진 구간 비율
    P_tcp = np.array([cell.tcp(x)[0] for x in X])
    pflag = np.array([n[2] for n in seq])
    seg = np.flatnonzero((pflag[:-1] == 1) & (pflag[1:] == 1))
    v = np.linalg.norm(np.diff(P_tcp, axis=0), axis=1)[seg] / np.diff(times)[seg]
    L = np.linalg.norm(np.diff(P_tcp, axis=0), axis=1)[seg]

    # 검증 지표
    errs, clears = [], []
    for n in seq:
        clears.append(P.clear(n[0]))
    for row, Xp, info in passes:
        for i, x in enumerate(Xp):
            p, R = cell.tcp(x)
            errs.append(pose_err(p, R, row['T'][i]))
    errs = np.array(errs)
    total = sum(len(r['T']) for r in rows)
    return {
        'X': X, 'times': times, 'seq': seq, 'passes': passes, 'transits': transits,
        'metrics': {
            'coverage_points': f'{len(errs)}/{total}',
            'coverage_percent': 100.0 * len(errs) / total,
            'rows': f'{len(passes)}/{len(rows)}',
            'max_node_position_error_mm': float(errs[:, 0].max() * 1000),
            'max_node_orientation_error_deg': float(np.degrees(errs[:, 1].max())),
            'interp_checked_max_position_error_mm': float(F.interp_max[0] * 1000),
            'interp_checked_max_orientation_error_deg': float(np.degrees(F.interp_max[1])),
            'inserted_midpoints': F.inserted,
            'min_node_clearance_mm': float(min(clears) * 1000),
            'certified_clearance_lower_bound_mm': float(F.cert_min * 1000),
            'uncertified_edges': len(F.uncertified),
            'collision_nodes': int(sum(c < 0 for c in clears)),
            'nodes': len(X), 'cycle_time_s': float(times[-1]),
            'paint_length_m': float(L.sum()),
            'paint_length_at_target_speed_percent': float(100 * L[v >= TCP_SPEED * .99].sum() / L.sum()),
            'paint_tcp_speed_min_mps': float(v.min()),
            'max_gantry_step_in_pass_mm': float(max(np.abs(np.diff(Xp[:, :2], axis=0)).max()
                                                    for _, Xp, _ in passes if len(Xp) > 1) * 1000),
        },
        'timing_s': {'load_models': t_load, **T,
                     'planning_total (passes+refine)': T['passes'] + T['refine'],
                     'all_including_certification': sum(T.values())},
        'counts': {'ik_calls': P.stats['ik_calls'], 'pass_attempts': P.stats['pass_attempts'],
                   'distance_evals': cell.evaluations, 'certify_evals': F.cert_evals},
        'uncertified': F.uncertified, 'exclusions': cell.exclusions,
    }


def save(a, cell, res):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    X, t, seq = res['X'], res['times'], res['seq']
    tool = cell.tool
    kinds = np.array([n[1] for n in seq])
    np.savez(out / 'trajectory_bumper.npz', X=X, t=t, names=np.array(GANTRY + ARM),
             paint=np.array([n[2] for n in seq]), kind=kinds,
             row=np.array([n[3] for n in seq]), csv_idx=np.array([n[4] for n in seq]),
             tool_names=np.array(list(tool)), tool_values=np.array(list(tool.values())))
    with open(out / 'joints_bumper.csv', 'w') as f:
        f.write('t,' + ','.join(GANTRY + ARM) + ',gun_lift,gun_swing,paint,kind,row,csv_idx\n')
        for ti, x, n in zip(t, X, seq):
            f.write(f'{ti:.4f},' + ','.join(f'{v:.6f}' for v in x) +
                    f",{tool['gun_lift']},{tool['gun_swing']},{n[2]},{n[1]},{n[3]},{n[4]}\n")
    wp = cell.root.find("joint[@name='world_base-bumper']/origin")
    val = {
        'status': 'verified' if res['metrics']['uncertified_edges'] == 0 else 'NOT verified',
        'method': 'decoupled: rule-based gantry placement from wrist center + arm-only DLS Jacobian IK '
                  '(no whole-body optimization)',
        'targets': 'painting_trajectory.csv poses as given (no normal realignment), all rows, paint and non-paint',
        'criteria': {'node_position_mm': POS_TOL * 1000, 'node_orientation_deg': math.degrees(ANG_TOL),
                     'interp_check_u': [.25, .5, .75], 'node_clearance_mm': NODE_CLEAR * 1000,
                     'certified_clearance_mm': CERT_CLEAR * 1000},
        'metrics': res['metrics'], 'timing_s': res['timing_s'], 'counts': res['counts'],
        'transits': res['transits'],
        'passes': [{'row': r['row'], 'points': len(Xp), **info} for r, Xp, info in res['passes']],
        'placement': {'world_base-bumper': {'xyz': wp.get('xyz'), 'rpy': wp.get('rpy')}},
        'tool_state': cell.tool,
        'collision_exclusions': [list(e) for e in res['exclusions'] if e[2] != 'adjacent assembly'],
        'adjacent_assembly_exclusions': sum(e[2] == 'adjacent assembly' for e in res['exclusions']),
        'uncertified': res['uncertified'],
        'urdf': a.urdf, 'urdf_sha256': sha(a.urdf), 'csv_sha256': sha(CSV),
        'trajectory_sha256': sha(out / 'joints_bumper.csv'),
        'limits': 'rigid modeled geometry only; hoses, fixtures, dynamics and real TCP need physical validation',
    }
    (out / 'validation.json').write_text(json.dumps(val, indent=1, ensure_ascii=False), encoding='utf-8')
    return val


def main():
    global SMOOTH, POOL
    ap = argparse.ArgumentParser()
    ap.add_argument('--urdf', default=str(DIR / 'paint_cell_bumper.urdf'))
    ap.add_argument('--out', default=str(DIR / 'planner' / 'out'))
    ap.add_argument('--stage', default='A', choices=['A', 'full'])
    ap.add_argument('--dx', type=float, default=0.0, help='범퍼를 월드 X 로 추가 이동 [m]')
    ap.add_argument('--dy', type=float, default=0.0)
    ap.add_argument('--dz', type=float, default=0.0)
    ap.add_argument('--rows', default='', help='예: 0,5,13 (진단용)')
    ap.add_argument('--diag', action='store_true')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--gun', default='on', choices=['off', 'on'], help='건 수납(off) / 전개(on)')
    ap.add_argument('--smooth', type=int, default=SMOOTH, help='패스 내 겐트리 이동평균 창')
    a = ap.parse_args()
    SMOOTH = a.smooth
    t0 = time.perf_counter()
    if a.dx or a.dy or a.dz:
        a.urdf = shifted_urdf(a.urdf, a.dx, a.dy, a.dz)
    if a.stage == 'full' and a.workers > 1:
        import multiprocessing as mp
        POOL = mp.get_context('fork').Pool(a.workers, initializer=_winit, initargs=(a.urdf, a.gun))
    cell = _make(a.urdf, a.gun)
    t_load = time.perf_counter() - t0
    P = Planner(cell)
    rows = load_targets(cell)
    if a.rows:
        keep = {int(v) for v in a.rows.split(',')}
        rows = [r for r in rows if r['row'] in keep]
    if a.diag:
        diagnose(P, rows)
        return
    if a.stage == 'full':
        res = run_full(a, cell, P, rows, t_load)
        val = save(a, cell, res)
        print(json.dumps({k: val[k] for k in ('status', 'metrics', 'timing_s', 'counts', 'transits')},
                         indent=1, ensure_ascii=False))
        if POOL is not None:
            POOL.close()
        return
    t1 = time.perf_counter()
    ok_pts, prev = 0, None
    for row in rows:
        X, info = P.plan_pass(row, prev)
        if X is not None:
            ok_pts += len(X)
            prev = X[-1]
            print(f"row {row['row']:2d}: OK {len(X)}/{len(row['T'])}  cand {info}  "
                  f"gantry step max {np.abs(np.diff(X[:, :2], axis=0)).max() * 1000 if len(X) > 1 else 0:.0f} mm",
                  flush=True)
        else:
            print(f"row {row['row']:2d}: FAIL {info}", flush=True)
    total = sum(len(r['T']) for r in rows)
    print(f'coverage {ok_pts}/{total}  load {t_load:.2f}s  plan {time.perf_counter() - t1:.2f}s  '
          f'attempts {P.stats["pass_attempts"]}  ik {P.stats["ik_calls"]}  dist {cell.evaluations}')


if __name__ == '__main__':
    sys.exit(main())
