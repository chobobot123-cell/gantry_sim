#!/usr/bin/env python3
"""분리(decoupled) 플래너: 겐트리 2축 경로를 먼저 따로 계획하고, 그 위에서 팔 6축만 계산한다.

8축을 한 식으로 푸는 계산은 어디에도 없다 (팔 풀이는 항상 겐트리 값을 고정한 6축 IK).
  1단계 (겐트리): 줄마다 겐트리 후보 경로 (손목 기준 방위 phi 9종 x 어깨-손목 거리 R 7종 = 63개) 를 기하만으로 만들고,
                  후보마다 팔 6축 IK 를 6점 간격으로 따로 풀어 '전부 닿고, 겹치지 않고, 자세가 안 뒤집히는' 후보를 고른다.
                  시도 순서는 직전 줄 후보에서 가까운 것부터. 다 통과한 후보가 없으면 빠진 점이 적은 후보 6개까지 시도.
  2단계 (팔)    : 정해진 겐트리 경로 위에서 팔 6축만 이어서 푼다. 한 점이 거부되면 (IK 실패 / 여유 15 mm 미만 / 자세 뒤집힘)
                  겐트리를 +-12 cm 안에서만 고치고 팔을 다시 푼다. 고친 값은 뒤 점들로 천천히 풀어 준다 (점당 x0.85).
  줄 사이 이동   : 파란 직선을 팔만으로 (위치만 맞춤, 남는 자유도로 충돌 회피) -> 안 되면 돌아가는 경유점 (팔 6축, 방향 포함).
                  이동 구간의 겐트리는 기하 규칙으로 잡고, 팔이 필요로 한 보정은 그대로 이어받는다 (보정이 한쪽으로 커지면 그
                  추세를 다음 점에 미리 반영). 즉 이동 구간 겐트리 경로는 1단계 값 + 2단계 국소 보정이다.
  홈 이동        : IK 가 아니라 관절 보간. 겐트리만 움직이는 구간과 팔만 움직이는 구간으로 나눈다.
  충돌 연속 인증 / 시간 배정 / 지표는 plan_chain 과 같은 함수 (계산 방식과 무관한 검사).

속도: 노드 판정은 거리 대신 '여유를 붙인 충돌 판정' (같은 결과, 약 10배 빠름). 어려운 줄은 (후보, 이동 방식) 조합을 워커에서
동시에 풀되 순차 실행과 같은 순서로만 결과를 해석한다 -> --workers 1 과 같은 궤적이 나온다 (세움 배치에서 해시 일치 확인).

사용법: plan_decoupled.py --urdf CELL --out DIR [--phi 0] [--rdes .8] [--home-arm 180,0,120,0,-120,0] [--prior validation.json]
"""
import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):   # 작은 행렬뿐: 워커마다 BLAS 스레드가 늘면 오히려 느려짐
    os.environ.setdefault(_k, '1')
import coal
import numpy as np
import pinocchio as pin

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plan_bumper as pb  # noqa: E402
import plan_chain as pc  # noqa: E402  (형상 보조 함수와 검사 함수만 사용: 8축 풀이 함수는 쓰지 않는다)
from plan_bumper import (CERT_CLEAR, IK_POS, POS_TOL, ANG_TOL, interp_pose, load_targets, pose_err, sha)  # noqa: E402

NODE_CLEAR = .015
MAX_STEP_DEG = 8.0
SMOOTH = 15                       # 줄 안 겐트리 경로 이동평균 창
AXIS_LOCK = False                 # --axis-lock: 한 줄 안에서는 겐트리 한 축만 따라가고 다른 축은 고정 (실제 겐트리 운전 조건)
SCREEN_EVERY = 6                  # 1단계 후보 선별: 이 간격의 점만 팔 IK
SCREEN_CLEAR = .002               # 선별에서 요구하는 최소 여유 (노드 기준 15 mm 는 2단계에서 맞춤)
REPAIR_R = (.02, .04, .06, .09, .12)   # 2단계 겐트리 국소 보정 반경 [m]
REPAIR_DECAY = .85                # 보정값을 뒤 점들로 풀어 주는 비율 (점당)
HOP_DECAY = 1.0                   # 줄 사이 이동 구간에서의 같은 비율
HOP_TREND = 1.0                   # 이동 구간: 보정값이 점마다 같은 쪽으로 커지면 그 추세를 다음 점에 미리 반영하는 비율
CAND_DPHI = (0, 15, -15, 30, -30, 45, -45, 60, -60)
CAND_R = (.66, .72, .78, .84, .90, .96, 1.0)
W_ORI = .3                        # 이동(위치만) 풀이: 공구 방향을 보간 방향으로 약하게 당김
VERBOSE = False
AVOID_TARGET = .020               # 이동 구간 회피가 확보하려는 여유 (이만큼 되면 멈춤)
J3_MIN = 12.0                     # [deg] 도장 노드에서 팔꿈치가 이보다 펴지면 거부 (0 = 끔) -> 겐트리 국소 보정으로 덜 펴게
J5_MIN = 15.0                     # [deg] 도장 노드에서 손목(j5)이 특이점에 이보다 가까우면 거부 (0 = 끔)
R_PULL = None                     # 줄마다 뻗은 거리(R)를 처음 값(--rdes) 쪽으로 당기는 세기. None = 기본 (0.3, photo 는 1.0)
ENDS = 'avg'                      # 줄 양 끝의 겐트리 평활: 'avg' = 한쪽 평균(끝에서 안쪽으로 쏠림), 'sym' = 창을 줄여 대칭 평균(끝점은 규칙 값 그대로)
POSTURE = ''                      # 'photo' = 현장 사진 자세: 팔꿈치가 어깨-손목 선의 위쪽(∩)인 분기로 시작해 끝까지 유지 + 뻗은 후보 유지
CENTER_BACK = 0.0                 # 범퍼 중앙에서 +X로 캐리지 후퇴하는 추가 거리 [m], 시뮬레이션 가정
SIDE_MIN = .03                    # [m] photo: 시작 자세에서 팔꿈치가 어깨-손목 선보다 이만큼 이상 위
AVOID_ON = .030                   # 이 거리 안이면 이동 구간에서 팔만의 충돌 회피 작동


class Dec:
    """겐트리 고정 + 팔 6축 계산기."""

    def __init__(self, cell):
        self.c = cell
        self.ik = pb.ArmIK(cell)
        self.rule = pb.GantryRule(cell, self.ik)
        self.lo, self.hi = cell.lower, cell.upper
        self.sig = None                               # 자세 분기 (시작 자세에서 정하고 끝까지 유지)
        self.n_ik = 0
        self.n_repair = 0
        self._cache = None
        self.ring_pool = None                         # 주 프로세스에서만: 이동 구간 국소 보정의 12방향을 워커에 나눠 계산
        self.no_coll = False                          # True 인 동안은 충돌을 보지 않는다 (--hops straight 의 강제 직선 구간: 결과에 충돌로 표시)
        # 충돌 라이브러리가 직전 계산의 탐색 시작점을 재사용하지 않게 한다: 같은 자세면 계산 이력(어느 워커가 무엇을 먼저
        # 풀었는지)과 무관하게 같은 값 -> 워커 수와 무관하게 같은 계획
        for req in list(cell.geom_data.distanceRequests):
            req.enable_cached_gjk_guess = False
            req.gjk_initial_guess = coal.GJKInitialGuess.DefaultGuess

        def grp(n):
            if n == 'workpiece':
                return 'wp'
            if n.startswith('obstacle_'):                     # 같은 셀의 다른 작업물 = 고정 장애물
                return 'cell'
            if n.startswith(('world_base', 'gantry_')):
                return 'cell'
            if n.startswith('gun_'):
                return 'gun'
            if n.startswith(('base_link', 'link_1')):
                return 'armbase'
            return 'arm'
        # 회피 대상 쌍: 팔/건 <-> 범퍼/셀, 건 <-> 팔
        self.avoid_mask = np.array([{grp(a), grp(b)} in ({'arm', 'wp'}, {'gun', 'wp'}, {'arm', 'cell'}, {'gun', 'cell'}, {'gun', 'arm'})
                                    for a, b in (cell.pair_names(k) for k in range(len(cell.ia)))])

    # ---- 1단계: 겐트리 경로 (팔 계산 없음, 기하만) ----
    def g_path(self, Ts, phi, R):
        """손목 중심에서 방위 phi, 어깨-손목 거리 R 이 되는 자리에 팔 베이스를 두는 겐트리 경로 (줄 안 이동평균)."""
        W = np.array([self.rule.wrist(T) for T in Ts])
        dz = self.rule.shoulder_z - W[:, 2]
        h = np.sqrt(np.maximum(R ** 2 - dz ** 2, .01)) + pb.SHOULDER_R
        u = np.array([math.cos(math.radians(phi)), math.sin(math.radians(phi))])
        B = W[:, :2] + h[:, None] * u
        if CENTER_BACK and len(Ts) > 1:
            # 범퍼의 높은 중앙부에서 +X(부품에서 떨어지는 방향)로 물리고,
            # 낮은 양 끝에서 보정을 0으로 줄인다. 실제 장비 좌표는 확인 필요.
            height = np.array([T[2, 3] for T in Ts])
            span = float(height.max()-height.min())
            if span > .1:
                normalized = np.clip((height-height.min())/span, 0., 1.)
                B[:, 0] += CENTER_BACK * normalized**2 * (3 - 2*normalized)
        if AXIS_LOCK and len(B) > 1:
            # 한 축 주행: 줄에서 손목이 많이 움직이는 축(주축)만 따라가고, 다른 축(보조축)은 줄 안에서 고정한다.
            # 보조축 = 원래 규칙 자리의 줄 평균. 주축 쪽 거리를 다시 맞춰 손목까지의 수평 거리 h (= 어깨-손목 거리 R) 를 유지.
            m = int(np.argmax(np.abs(np.diff(W[:, :2], axis=0)).sum(0)))
            c = 1 - m
            bc = B[:, c].mean()
            off = bc - W[:, c]
            sgn = 1.0 if u[m] >= 0 else -1.0
            B[:, m] = W[:, m] + sgn * np.sqrt(np.maximum(h ** 2 - off ** 2, 0.0))
            B[:, c] = bc
        G = np.clip(B - self.ik.base_off, self.lo[:2], self.hi[:2])
        k = SMOOTH // 2
        if ENDS == 'sym':
            G = np.array([G[i - min(k, i, len(G) - 1 - i):i + min(k, i, len(G) - 1 - i) + 1].mean(0) for i in range(len(G))])
        else:
            G = np.array([G[max(0, i - k):i + k + 1].mean(0) for i in range(len(G))])
        return np.clip(G, self.lo[:2], self.hi[:2])

    def g_point(self, T, phi, R):
        """한 점의 겐트리 자리 (이동평균 없음)."""
        W = self.rule.wrist(T)
        dz = self.rule.shoulder_z - W[2]
        h = math.sqrt(max(R ** 2 - dz ** 2, .01)) + pb.SHOULDER_R
        B = W[:2] + h * np.array([math.cos(math.radians(phi)), math.sin(math.radians(phi))])
        return np.clip(B - self.ik.base_off, self.lo[:2], self.hi[:2])

    # ---- 검사 ----
    def clear(self, x):
        if self._cache is not None and np.array_equal(self._cache[0], x):
            return self._cache[1]
        v = float(self.c.distances(x, near=.035).min())
        self._cache = (np.array(x, float), v)
        return v

    def free(self, x, margin=NODE_CLEAR, mask=None):
        """모든 쌍(mask 를 주면 그 쌍만)이 margin 이상 떨어져 있는가. 거리를 재지 않고 '여유를 붙인 충돌 판정'만 하므로
        clear() 보다 약 10배 빠르다 (정확 거리 판정과 결과가 같음을 세 셀 2900여 자세에서 확인)."""
        c = self.c
        c.set(x)
        pin.updateGeometryPlacements(c.model, c.data, c.geom, c.geom_data)
        Rm = np.array([o.rotation for o in c.geom_data.oMg])
        tm = np.array([o.translation for o in c.geom_data.oMg])
        cc = np.einsum('nij,nj->ni', Rm, c.centers) + tm
        h = np.einsum('nij,nj->ni', abs(Rm), c.halves)
        gap = np.linalg.norm(np.maximum(abs(cc[c.ia] - cc[c.ib]) - h[c.ia] - h[c.ib], 0), axis=1)
        idx = np.flatnonzero(gap < margin if mask is None else (gap < margin) & mask)
        for k in idx[np.argsort(gap[idx])]:                   # 가까운 쌍부터: 겹치면 바로 끝
            c.geom_data.collisionRequests[int(k)].security_margin = margin
            if pin.computeCollision(c.geom, c.geom_data, int(k)):
                return False
        return True

    def masked_near(self, x):
        """회피 대상 쌍 가운데 AVOID_ON 안쪽인 것을 가까운 순서로: [(거리, 8관절 기울기)] 를 만드는 재료.
        Cell.distances + ChainSolver.pair_grad 와 같은 값이지만, 대상 쌍만 정확 거리를 재고 그 결과(최근접점/법선)를
        기울기에 그대로 쓴다 (같은 쌍 거리를 두 번 재지 않음). 반환: [(d, 쌍 번호, 법선, 최근접점1, 최근접점2)]."""
        c = self.c
        c.set(x)
        pin.computeJointJacobians(c.model, c.data, c.q)
        pin.updateGeometryPlacements(c.model, c.data, c.geom, c.geom_data)
        Rm = np.array([o.rotation for o in c.geom_data.oMg])
        tm = np.array([o.translation for o in c.geom_data.oMg])
        cc = np.einsum('nij,nj->ni', Rm, c.centers) + tm
        h = np.einsum('nij,nj->ni', abs(Rm), c.halves)
        gap = np.linalg.norm(np.maximum(abs(cc[c.ia] - cc[c.ib]) - h[c.ia] - h[c.ib], 0), axis=1)
        out = []
        for k in np.flatnonzero((gap < AVOID_ON + .005) & self.avoid_mask):
            c.geom_data.collisionRequests[int(k)].security_margin = AVOID_ON
            if not pin.computeCollision(c.geom, c.geom_data, int(k)):   # AVOID_ON 보다 먼 쌍: 정확 거리 불필요 (빠른 판정으로 거름)
                continue
            res = pin.computeDistance(c.geom, c.geom_data, int(k))
            if res.min_distance < AVOID_ON:
                out.append((float(res.min_distance), int(k), np.array(res.normal), np.array(res.getNearestPoint1()), np.array(res.getNearestPoint2())))
        out.sort(key=lambda t: (t[0], t[1]))
        return out

    def grad_of(self, item):
        """masked_near 의 한 항목 -> 그 쌍 거리의 8관절 기울기 (masked_near 직후, 같은 자세에서만 유효)."""
        c = self.c
        _, k, n, p1, p2 = item
        pr = c.geom.collisionPairs[k]
        g = np.zeros(8)
        for obj_i, p, sgn in ((pr.first, p1, -1.0), (pr.second, p2, 1.0)):
            j = c.geom.geometryObjects[obj_i].parentJoint
            if j == 0:
                continue
            Jj = pin.getJointJacobian(c.model, c.data, j, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            r = p - c.data.oMi[j].translation
            skew = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
            g += sgn * (n @ (Jj[:3] - skew @ Jj[3:])[:, c.iv])
        return g

    def reject(self, x):
        """노드 거부 사유 (없으면 None)."""
        if not self.no_coll and not self.free(x):
            return 'collision'
        if self.sig is not None and pc.signature(self.c, x) != self.sig:
            return 'branch'
        return None

    def elbow_side(self, x):
        """팔꿈치가 어깨-손목 선에서 위쪽으로 떨어진 거리 [m] (+ = 위쪽 ∩, - = 아래쪽 V). 선을 포함한 수직면 안에서 잰다."""
        c = self.c
        c.set(x)
        sh = c.data.oMi[c.model.getJointId('joint_2')].translation
        el = c.data.oMi[c.model.getJointId('joint_3')].translation
        wr = c.data.oMf[c.model.getFrameId('link_5')].translation
        u = wr - sh
        u = u / np.linalg.norm(u)
        n = np.array([0, 0, 1.0]) - u[2] * u
        nn = np.linalg.norm(n)
        return float((el - sh) @ n / nn) if nn > 1e-6 else 0.0

    def singular(self, x):
        """팔꿈치가 거의 펴졌거나 손목이 특이점에 가까운가 (도장 노드에서만 본다)."""
        return (J3_MIN > 0 and abs(math.degrees(x[4])) < J3_MIN) or (J5_MIN > 0 and abs(math.degrees(math.asin(math.sin(x[6])))) < J5_MIN)

    # ---- 팔 6축 풀이 (겐트리 g 고정) ----
    def arm(self, g, T, q0, iters=80):
        self.n_ik += 1
        return self.ik.solve(g, T, q0, iters=iters)

    def arm_pos(self, g, T, q0, iters=80):
        """위치만 맞추는 팔 6축 풀이 (이동 구간): 남는 3자유도는 직전 자세 근처 + 보간 방향 쪽."""
        self.n_ik += 1
        q = np.array(q0, float)
        pt, Rt = T[:3, 3], T[:3, :3]
        lo, hi = self.lo[2:], self.hi[2:]
        for _ in range(iters):
            J, p, R = self.c.jacobian_tcp(np.r_[g, q])
            Ja, Jo = J[:3, 2:], J[3:, 2:]
            e = pt - p
            Jp = Ja.T @ np.linalg.inv(Ja @ Ja.T + 1e-6 * np.eye(3))
            grad = .5 * (q - q0) - W_ORI * (Jo.T @ (R @ pin.log3(R.T @ Rt)))
            dq = Jp @ e - (np.eye(6) - Jp @ Ja) @ (.3 * grad)
            n = np.linalg.norm(dq)
            if n > .3:
                dq *= .3 / n
            q = np.clip(q + dq, lo, hi)
            if np.linalg.norm(e) < IK_POS and n < 2e-4:
                break
        if not self.no_coll:
            q = self.avoid_pos(g, q, pt)
        p = self.c.tcp(np.r_[g, q])[0]
        return q, bool(np.linalg.norm(pt - p) < IK_POS * 5)

    def avoid_pos(self, g, q, pt, iters=10):
        """팔 6축만의 충돌 회피 (겐트리 고정): 위치(3)를 맞추고 남는 3자유도로 가장 가까운 쌍에서 멀어진다."""
        lo, hi = self.lo[2:], self.hi[2:]
        best = -1.0
        for it in range(iters):
            x = np.r_[g, q]
            if self.free(x, AVOID_TARGET, self.avoid_mask):   # 이미 여유 충분 (아래 거리 계산과 같은 판정, 더 빠름)
                break
            J, p, _ = self.c.jacobian_tcp(x)
            Ja = J[:3, 2:]
            Jp = Ja.T @ np.linalg.inv(Ja @ Ja.T + 1e-6 * np.eye(3))
            N = np.eye(6) - Jp @ Ja
            near = self.masked_near(x)
            if not near or near[0][0] >= AVOID_TARGET:        # 필요한 여유(노드 기준 15 mm + 조금)면 충분
                break
            if it >= 3 and near[0][0] < best + .0005:         # 더 나아지지 않으면 그만
                break
            best = max(best, near[0][0])
            gn, d0 = None, None
            for item in near[:3]:                            # 기울기가 있는 가장 가까운 쌍
                d0 = item[0]
                v = N @ self.grad_of(item)[2:]
                if np.linalg.norm(v) > 1e-5:
                    gn = v
                    break
            if gn is None:
                break
            step = min(.02, AVOID_ON - d0 + .005) / np.linalg.norm(gn)
            q = np.clip(q + Jp @ (pt - p) + step * gn, lo, hi)
        for _ in range(10):                                  # 위치 다시 정확히
            J, p, _ = self.c.jacobian_tcp(np.r_[g, q])
            e = pt - p
            if np.linalg.norm(e) < IK_POS:
                break
            Ja = J[:3, 2:]
            q = np.clip(q + Ja.T @ np.linalg.solve(Ja @ Ja.T + 1e-8 * np.eye(3), e), lo, hi)
        self._cache = None
        return q

    def node(self, T, xp, g, pos_only=False, strict=False):
        """겐트리 g 에서 팔을 풀고, 거부되면 g 를 +-12 cm 안에서 고쳐 다시 푼다. 반환: (x, 사유 또는 None, 쓴 보정)."""
        g = np.clip(np.asarray(g, float), self.lo[:2], self.hi[:2])
        f = self.arm_pos if pos_only else self.arm
        q, ok = f(g, T, xp[2:])
        x = np.r_[g, q]
        def rej(x_):                                         # strict (도장 노드): 특이점 여유도 본다
            w = self.reject(x_)
            return 'singular' if w is None and strict and self.singular(x_) else w
        why = 'ik' if not ok else rej(x)
        if why is None:
            return x, None, np.zeros(2)
        first = why
        if why == 'collision':                               # 먼저: 가장 가까운 쌍에서 멀어지는 방향으로 겐트리를 조금씩 옮기고 팔만 다시 풂
            g2, q2 = g.copy(), q.copy()
            for _ in range(10):
                x2 = np.r_[g2, q2]
                d = self.c.distances(x2, near=AVOID_ON + .005)
                near = [k for k in np.argsort(d) if d[k] < AVOID_ON]
                gg = None
                for k in near[:3]:
                    _, g8 = pc.ChainSolver.pair_grad(self, x2, k)
                    if np.linalg.norm(g8[:2]) > 1e-4:
                        gg = g8[:2] / np.linalg.norm(g8[:2])
                        break
                if gg is None:
                    break
                g2 = np.clip(g2 + .015 * gg, self.lo[:2], self.hi[:2])
                q2, ok2 = f(g2, T, q2)
                if not ok2:
                    break
                self._cache = None
                if rej(np.r_[g2, q2]) is None:
                    self.n_repair += 1
                    return np.r_[g2, q2], None, g2 - g
        for r in REPAIR_R:                                   # 가까운 반경부터, 같은 반경에서는 관절 변화가 가장 작은 것
            best = None
            if _stale():
                break
            ring = [np.clip(g + r * np.array([math.cos(a), math.sin(a)]), self.lo[:2], self.hi[:2]) for a in np.radians(np.arange(0, 360, 30))]
            got = None
            if pos_only and self.ring_pool is not None:      # 위치만 풀이는 한 번이 무겁다 -> 12방향을 동시에 (계산 내용은 같음)
                self.n_ik += len(ring)
                got = self.ring_pool.map(_wring, [(T, xp, g2, self.sig) for g2 in ring], chunksize=1)
            for i, g2 in enumerate(ring):
                if got is not None:
                    x2, bad = got[i]
                    q2 = x2[2:]
                    if bad:
                        continue
                else:
                    q2, ok2 = f(g2, T, xp[2:])
                    x2 = np.r_[g2, q2]
                    if not ok2 or rej(x2) is not None:
                        continue
                c = float(np.abs(q2 - xp[2:]).max())
                if best is None or c < best[0]:
                    best = (c, x2, g2 - g)
            if best is not None:
                self.n_repair += 1
                return best[1], None, best[2]
        return x, first, np.zeros(2)

    def start(self, T, g, sig=None, all_sigs=False):
        """첫 점: 팔 시드 격자에서 풀어 여유가 가장 큰 자세 (겐트리 g 고정). sig 를 주면 그 자세 분기만.
        POSTURE == 'photo' 면 팔꿈치가 어깨-손목 선 위쪽인 자세만. all_sigs: 분기별 최선 [(분기, (점수, x))] 을 점수 순으로."""
        best, per = None, {}
        for s in self.ik.seeds(g, self.rule.wrist(T)):
            q, ok = self.arm(g, T, s)
            if not ok:
                continue
            x = np.r_[g, q]
            d = self.clear(x)
            if d < NODE_CLEAR or self.singular(x):
                continue
            if sig is not None or all_sigs or POSTURE:
                sg = pc.signature(self.c, x)
                if sig is not None and sg != tuple(sig):
                    continue
                if POSTURE == 'photo' and self.elbow_side(x) < SIDE_MIN:
                    continue
            sc = -min(d, .1) - .05 * min(abs(math.sin(q[4])), .5) + .01 * np.abs(q).sum()
            if best is None or sc < best[0]:
                best = (sc, x)
            if all_sigs and (sg not in per or sc < per[sg][0]):
                per[sg] = (sc, x)
        if all_sigs:
            return sorted(per.items(), key=lambda kv: kv[1][0])
        return None if best is None else best[1]


def fk_pose(D, x):
    p, R = D.c.tcp(x)
    return pc.pose(p, R)


def advance(D, X, C, Ts, ents, G, strict_n=0, free_n=0):
    try:
        return _advance(D, X, C, Ts, ents, G, strict_n, free_n)
    finally:
        D.no_coll = False


def _advance(D, X, C, Ts, ents, G, strict_n=0, free_n=0):
    """Ts 를 차례로 팔만 풀어 붙인다 (겐트리 = G + 보정). 관절 변화가 8도를 넘으면 그 사이를 잘게 나눈다."""
    corr = d_prev = np.zeros(2)

    def node2(T, xp, g, c, pos_only, strict):
        """보정을 더한 자리에서 먼저, 거부되면 보정 없는 원래 자리에서. 반환 (x, 사유, 원래 자리 대비 겐트리 차이)."""
        x, why, d = D.node(T, xp, g + c, pos_only, strict)
        if why is not None and np.abs(c).max() > 1e-6:
            x, why, d = D.node(T, xp, g, pos_only, strict)
            return x, why, d
        return x, why, c + d

    for i_node, (T, ent, g) in enumerate(zip(Ts, ents, G)):
        if _stale():
            return False, 'aborted'
        D.no_coll = i_node < free_n                           # 강제 직선 구간: 충돌을 보지 않고 직선 그대로
        g = np.asarray(g, float)
        pos_only = ent[1] == 'transit' and ent[3] == -1
        T_prev = fk_pose(D, X[-1])
        strict = ent[1] == 'target' or i_node < strict_n      # 도장 노드, 그리고 옆 줄 경로를 따라가는 이동 노드(같은 포즈)는 특이점 여유도 본다
        xn, why, d = node2(T, X[-1], g, corr, pos_only, strict)
        jump = np.degrees(np.abs(xn[2:] - X[-1][2:]).max()) if why is None else 999
        if why is not None or jump > MAX_STEP_DEG:
            m = int(min(16, max(2, math.ceil((jump if why is None else 24) / (MAX_STEP_DEG / 2)))))
            xp, gp, gt, xs = X[-1], X[-1][:2], g + (d if why is None else corr), []
            for k in range(1, m + 1):
                u = k / m
                Tk = T if k == m else interp_pose(T_prev, T, u)
                xk, whyk, dk = D.node(Tk, xp, gp + (gt - gp) * u, pos_only, strict)
                if whyk is not None:
                    return False, whyk if why is None else why
                if np.degrees(np.abs(xk[2:] - xp[2:]).max()) > MAX_STEP_DEG:
                    return False, f'joint jump {jump:.0f}deg'
                gt = gt + dk
                xs.append((xk, Tk))
                xp = xk
            d = xp[:2] - g
            kind = 'mid' if ent[1] == 'target' and C[-1][1] in ('target', 'mid') else 'transit'
            for xk, Tk in xs[:-1]:
                X.append(xk)
                C.append((Tk, kind, int(ent[2] and C[-1][2]), ent[3] if kind == 'mid' or ent[1] == 'transit' else -1, -1))
            xn = xs[-1][0]
        if ent[1] == 'transit':                               # d = 이번 점에서 실제로 쓴 겐트리 - 계획 자리
            corr = d * HOP_DECAY + HOP_TREND * (d - d_prev)
        else:
            corr = d * REPAIR_DECAY
        d_prev = d
        X.append(xn)
        C.append(ent)
    return True, 'ok'


def hop_gantry(D, hp, c_prev, c_new, g_start, g_end):
    """줄 사이 이동 구간의 겐트리 경로: 경유점마다 손목 기준 자리(방위/거리는 직전 줄 값 -> 다음 줄 값으로 서서히)를 잡고,
    양 끝은 실제 겐트리 위치(직전 줄 끝, 다음 줄 시작)에 맞춰 이어 붙인다. 팔 계산은 하지 않는다."""
    n = len(hp)
    if n == 0:
        return []
    u = (np.arange(n) + 1) / (n + 1)
    G = np.array([D.g_point(T, c_prev[0] + (c_new[0] - c_prev[0]) * uk, c_prev[1] + (c_new[1] - c_prev[1]) * uk) for T, uk in zip(hp, u)])
    G = G + (1 - u)[:, None] * (g_start - G[0]) + u[:, None] * (g_end - G[-1])
    k = 2
    G = np.array([G[max(0, i - k):i + k + 1].mean(0) for i in range(n)])
    return list(np.clip(G, D.lo[:2], D.hi[:2]))


def candidates(prev, init, prior=None):
    """줄의 겐트리 후보 (phi, R): 처음 준 값 주변 격자 전체를, 직전 줄 값에서 가까운 순서로 (처음 값 쪽으로 약하게 당김)."""
    c = [(init[0] + dp, R) for dp in CAND_DPHI for R in CAND_R]
    def dist(t, ref):
        return abs(t[0] - ref[0]) / 15 + abs(t[1] - ref[1]) / .06
    if POSTURE == 'photo' or R_PULL is not None:               # 뻗은 거리(R)를 처음 값 근처로 더 세게 유지
        wr = 1.0 if R_PULL is None else R_PULL
        c = sorted(dict.fromkeys(c), key=lambda t: dist(t, prev) + .3 * abs(t[0] - init[0]) / 15 + wr * abs(t[1] - init[1]) / .06)
    else:
        c = sorted(dict.fromkeys(c), key=lambda t: dist(t, prev) + .3 * dist(t, init))
    if prior is not None:
        c = [tuple(prior)] + [t for t in c if t != tuple(prior)]
    return c


def screen(D, Ts, G, q0):
    """1단계 선별: 겐트리 경로 G 위에서 일부 점만 팔 IK. 반환 (통과한 점 수, 선별 점 수, 팔 관절 변화 합)."""
    idx = sorted(set(list(range(0, len(Ts), SCREEN_EVERY)) + [len(Ts) - 1]))
    q, okn, cost = np.array(q0, float), 0, 0.0
    for j in idx:
        qn, ok = D.arm(G[j], Ts[j], q)
        x = np.r_[G[j], qn]
        # 선별 기준: 팔이 닿고, 같은 자세이고, 실제로 겹치지 않음 (좁은 여유는 2단계 국소 보정이 메운다)
        if ok and D.free(x, SCREEN_CLEAR) and not D.singular(x) and (D.sig is None or pc.signature(D.c, x) == D.sig):
            okn += 1
            cost += float(np.abs(qn - q).sum())
            q = qn
    return okn, len(idx), cost


def dp_gantry(D, Ts, q0, g_prev, init_phi, n_pos=0):
    """1단계 (2차 방법): 줄의 선별 점마다 겐트리 자리 격자(phi x R)에서 팔 6축 IK 를 따로 풀어 '닿고 충돌 없는 자리' 지도를
    만들고, 그 위에서 겐트리 이동이 가장 적은 경로를 동적계획법으로 고른다. 반환: 줄 전체 겐트리 경로 또는 (None, 막힌 점).
    Ts 앞쪽 n_pos 개는 줄 사이 이동(파란 직선) 노드: 위치만 맞추는 팔 풀이를 쓴다 (이동과 줄을 한 덩어리로 계획)."""
    idx = sorted(set(list(range(0, len(Ts), SCREEN_EVERY)) + [len(Ts) - 1] + ([n_pos] if 0 < n_pos < len(Ts) else [])))
    cells = [(init_phi + dp, R) for dp in CAND_DPHI for R in CAND_R]
    feas = []                                                 # feas[jj][c] = (g, q) 또는 None
    for jj, j in enumerate(idx):
        cur = []
        for ci, (ph, R) in enumerate(cells):
            g = D.g_point(Ts[j], ph, R)
            if jj == 0:
                seed = q0
            elif feas[jj - 1][ci] is not None:
                seed = feas[jj - 1][ci][1]
            else:                                             # 직전 점에서 가장 가까운 자리의 팔 자세를 시드로
                near = [f for f in feas[jj - 1] if f is not None]
                seed = min(near, key=lambda f: np.linalg.norm(f[0] - g))[1] if near else q0
            q, ok = (D.arm_pos if j < n_pos else D.arm)(g, Ts[j], seed)
            cur.append((g, q) if ok and D.reject(np.r_[g, q]) is None else None)
        if not any(f is not None for f in cur):
            return None, j
        feas.append(cur)
    INF = float('inf')
    cost = [[INF] * len(cells) for _ in idx]
    back = [[-1] * len(cells) for _ in idx]
    for ci, f in enumerate(feas[0]):
        if f is not None:
            cost[0][ci] = np.linalg.norm(f[0] - g_prev)        # 직전 줄 끝 겐트리에서 가까운 자리부터
    for jj in range(1, len(idx)):
        for ci, f in enumerate(feas[jj]):
            if f is None:
                continue
            for cj, fp in enumerate(feas[jj - 1]):
                if fp is None or cost[jj - 1][cj] == INF:
                    continue
                # 겐트리 이동 + 팔 관절 변화 (같은 자세가 이어지게)
                c = cost[jj - 1][cj] + np.linalg.norm(f[0] - fp[0]) + .15 * float(np.abs(f[1] - fp[1]).max())
                if c < cost[jj][ci]:
                    cost[jj][ci], back[jj][ci] = c, cj
    ci = int(np.argmin(cost[-1]))
    if cost[-1][ci] == INF:
        return None, idx[-1]
    sel = [ci]
    for jj in range(len(idx) - 1, 0, -1):
        sel.append(back[jj][sel[-1]])
    sel = sel[::-1]
    Gk = np.array([feas[jj][c][0] for jj, c in enumerate(sel)])
    j_all = np.arange(len(Ts))
    G = np.stack([np.interp(j_all, idx, Gk[:, a]) for a in (0, 1)], 1)
    k = 2                                                     # 꺾이는 곳만 살짝 둥글게 (창 5)
    G = np.array([G[max(0, i - k):i + k + 1].mean(0) for i in range(len(G))])
    return np.clip(G, D.lo[:2], D.hi[:2]), [cells[c] for c in sel]


HOPS = [('line', 0.0)] + [('around', v) for v in range(1, pc.HOP_VARIANTS)]
HOPS_CLEAN = [('line', 0.0), ('retrace', 1), ('retrace', 2)]   # --hops clean: 파란 직선, 안 되면 옆 줄 경로를 따라 (넘어가는 이동 없음)
HOPS_CURVE = [('arc', 1.5), ('retrace', 1), ('retrace', 2), ('arc', 3.0), ('line', 0.0)]   # --hops curve: 줄 사이를 곡선으로 (짧은 곳은 U 자 곡선, 범퍼에 막히는 곳은 옆 줄 경로)
ARC_MIN = .12                     # [m] U 자 곡선의 접선 세기 하한 (짧은 이동도 눈에 보이는 곡선이 되게)
HOPS_STRAIGHT = [('line', 0.0), ('force', 0.0)]              # --hops straight: 파란 직선 그대로. 충돌 없이 안 되면 충돌을 무시하고 직선으로 (결과에 충돌 구간으로 표시)
HOP_MODE = 'default'
RETRACE_MAX = .15                 # [m] 옆 줄 경로로 옮겨 탈 때 허용하는 거리 (줄 간격 5 cm)
RETRACE_BLEND = 10                # 옆 줄 경로로 옮겨 타는 옆 이동을 이만큼의 점에 걸쳐 서서히 (꺾이지 않고 부드럽게 합류)


HOPS_LANE = [('line', 0.0), ('lane', 1), ('lane', 2), ('retrace', 1), ('retrace', 2)]   # --hops lane: 파란 직선, 안 되면 옆 줄과 나란히 한 줄 간격 떨어져 가다가 줄 끝에서 짧은 직선으로 이어짐


def hop_list():
    return list({'clean': HOPS_CLEAN, 'curve': HOPS_CURVE, 'straight': HOPS_STRAIGHT, 'lane': HOPS_LANE}.get(HOP_MODE, HOPS))


def hop_arc(A, B, ta, tb, k):
    """줄 끝 A (진행 방향 ta) 에서 다음 줄 시작 B (진행 방향 tb) 로 가는 매끄러운 곡선: 양 끝에서 줄의 진행 방향과 접한다
    (3차 에르미트). 줄 끝을 지나 조금 더 나갔다가 돌아 들어오는 U 자 모양이라 건이 멈추지 않고 방향을 바꾼다."""
    pa, pb_ = A[:3, 3], B[:3, 3]
    L = float(np.linalg.norm(pb_ - pa))
    s = max(k * L, ARC_MIN)
    n = max(4, int(math.ceil((L + .5 * s) / .02)))
    out = []
    for i in range(1, n):
        u = i / n
        h00, h10, h01, h11 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u, -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
        T = interp_pose(A, B, u)
        T[:3, 3] = h00 * pa + h10 * s * ta + h01 * pb_ + h11 * s * tb
        out.append(T)
    return out


def _line_blocked(D, A, B):
    """줄 끝 A -> 다음 줄 시작 B 의 직선(건 끝이 지나는 선)이 범퍼 면을 뚫거나 1 mm 이내로 스치는가."""
    c = D.c
    if getattr(D, '_wp', None) is None:
        c.set(np.r_[(c.lower[:2] + c.upper[:2]) / 2, np.zeros(6)])
        pin.updateGeometryPlacements(c.model, c.data, c.geom, c.geom_data)
        i = next(i for i, o in enumerate(c.geom.geometryObjects) if o.name == 'workpiece')
        M = c.geom_data.oMg[i]
        D._wp = (c.geom.geometryObjects[i].geometry, coal.Transform3s(M.rotation, M.translation))
    pa, pb_ = A[:3, 3], B[:3, 3]
    L = float(np.linalg.norm(pb_ - pa))
    if L < 1e-6:
        return False
    z = (pb_ - pa) / L
    x = np.cross(z, [1.0, 0, 0] if abs(z[0]) < .9 else [0, 1.0, 0])
    x /= np.linalg.norm(x)
    R = np.c_[x, np.cross(z, x), z]                           # 캡슐의 축(z)을 직선 방향으로
    return bool(coal.collide(coal.Capsule(.001, L), coal.Transform3s(R, (pa + pb_) / 2), D._wp[0], D._wp[1],
                             coal.CollisionRequest(), coal.CollisionResult()))


def hop_seq(D, rows, r, v, zsafe, cur, cand, g_start, G):
    """r-1 번 줄 끝 -> r 번 줄 시작의 이동 방식 v: (경유 포즈들, 위치만 풀이 여부 태그, 겐트리 경로). 불가능한 방식이면 None.
    line    : 파란 직선 (위치만 맞춤)
    around  : 물러남 -> 직선 -> 접근 / 위로 넘어감 (포즈 6D)
    retrace : 옆 줄 경로를 도장할 때와 똑같이 따라 이동 (분사만 끔). 1 = 다음 줄 경로를 거꾸로, 2 = 방금 칠한 줄 경로를 거꾸로."""
    Ts, prev = rows[r]['T'], rows[r - 1]['T']
    A, B = prev[-1], Ts[0]
    if v[0] == 'line':
        if _line_blocked(D, A, B):                            # 직선 자체가 범퍼 면을 뚫으면 풀어 볼 필요가 없다 (실패를 기다리는 시간만 든다)
            return None
        hp = pc.hop_line(D, A, B, 0.0)[0]
        return hp, -1, hop_gantry(D, hp, cur, cand, g_start, G[0])
    if v[0] == 'force':                                       # 같은 파란 직선, 충돌 판정 없이 (표시는 -5)
        hp = pc.hop_line(D, A, B, 0.0)[0]
        return hp, -5, hop_gantry(D, hp, cur, cand, g_start, G[0])
    if v[0] == 'arc':                                         # 줄 진행 방향에 접하는 곡선 (위치만 맞춤, 파란 직선과 같은 풀이)
        if len(prev) < 2 or len(Ts) < 2:
            return None
        ta, tb = prev[-1][:3, 3] - prev[-2][:3, 3], Ts[1][:3, 3] - Ts[0][:3, 3]
        hp = hop_arc(A, B, ta / np.linalg.norm(ta), tb / np.linalg.norm(tb), v[1])
        return hp, -1, hop_gantry(D, hp, cur, cand, g_start, G[0])
    if v[0] == 'around':
        hp = pc.hop_poses(A, B, zsafe, v[1])
        return hp, -2, hop_gantry(D, hp, cur, cand, g_start, G[0])
    def shifted(T, off):                                       # 같은 방향, 위치만 옮긴 포즈
        T2 = np.array(T, float)
        T2[:3, 3] = T2[:3, 3] + off
        return T2

    if v[0] == 'lane':
        # 옆 줄과 나란히, 줄 간격만큼 옆으로 떨어진 채로 끝까지 가고 (짧은 줄을 옆 줄 모양대로 늘인 길), 줄 끝에서 다른 줄 사이
        # 이동처럼 짧은 직선으로 옆 줄에 이어진다. 1 = 다음 줄 모양을 따라 그 시작점 옆까지, 2 = 방금 칠한 줄 끝에서 먼저 옆으로
        # 짧게 옮긴 뒤 그 줄 모양을 거꾸로 따라 다음 줄 시작점까지.
        ref = Ts if v[1] == 1 else prev
        Pr = np.array([T[:3, 3] for T in ref])
        nrm = np.linalg.svd(Pr - Pr.mean(0))[2][-1]           # 줄이 놓인 평면의 법선 = 줄 간격 방향
        def split(off):                                        # 옆(줄 간격) 성분은 끝까지 유지, 나머지는 서서히 0 으로
            lat = (off @ nrm) * nrm
            return lat, off - lat

        def sm(t):
            t = min(1.0, max(0.0, t))
            return t * t * (3 - 2 * t)

        if v[1] == 1:
            d = np.array([np.linalg.norm(T[:3, 3] - A[:3, 3]) for T in Ts])
            k = int(np.argmin(d))
            if k < 2 or d[k] > RETRACE_MAX:
                return None
            lat, rest = split(A[:3, 3] - Ts[k][:3, 3])
            idx = list(range(k - 1, -1, -1))                  # 다음 줄 시작점 바로 옆까지
            nb = min(RETRACE_BLEND, len(idx))
            hp = [shifted(Ts[i], lat + (1 - sm((n + 1) / (nb + 1))) * rest) for n, i in enumerate(idx)]
            Gd = np.array([G[i] for i in idx])                # 그 점들을 칠할 때와 같은 겐트리 자리
            u = (np.arange(len(hp)) + 1) / len(hp)
            return hp, -2, list(np.clip(Gd + (1 - u)[:, None] * (g_start - Gd[0]), D.lo[:2], D.hi[:2]))
        d = np.array([np.linalg.norm(T[:3, 3] - B[:3, 3]) for T in prev])
        m = int(np.argmin(d))
        if m >= len(prev) - 2 or d[m] > RETRACE_MAX:
            return None
        lat, rest = split(B[:3, 3] - prev[m][:3, 3])
        idx = list(range(len(prev) - 1, m, -1))               # 줄 끝점 바로 옆에서 시작 (첫 이동 = 짧은 직선)
        nb = min(RETRACE_BLEND, len(idx))
        hp = [shifted(prev[i], lat + sm((n - (len(idx) - 1 - nb)) / (nb + 1)) * rest) for n, i in enumerate(idx)]
        Gp = D.g_path(prev, cur[0], cur[1])
        Gd = np.array([Gp[i] for i in idx])
        d0, d1 = g_start - Gp[-1], G[0] - Gd[-1]
        u = (np.arange(len(hp)) + 1) / (len(hp) + 1)
        return hp, -2, list(np.clip(Gd + (1 - u)[:, None] * d0 + u[:, None] * d1, D.lo[:2], D.hi[:2]))

    def smooth(t):                                             # 0 -> 1 매끄럽게
        t = min(1.0, max(0.0, t))
        return t * t * (3 - 2 * t)

    if v[1] == 1:                                             # 다음 줄 경로를 거꾸로 따라 시작점까지. 줄 간격만큼의 옆 이동은 처음 RETRACE_BLEND 점에 걸쳐 서서히
        d = np.array([np.linalg.norm(T[:3, 3] - A[:3, 3]) for T in Ts])
        k = int(np.argmin(d))
        if k < 2 or d[k] > RETRACE_MAX:
            return None
        off = A[:3, 3] - Ts[k][:3, 3]
        idx = list(range(k - 1, 0, -1))
        nb = min(RETRACE_BLEND, len(idx))
        hp = [shifted(Ts[i], (1 - smooth((n + 1) / (nb + 1))) * off) for n, i in enumerate(idx)]
        Gd = np.array([G[i] for i in idx])                    # 그 점들을 칠할 때와 같은 겐트리 자리
        d0 = g_start - Gd[0]
        u = (np.arange(len(hp)) + 1) / len(hp)
        return hp, -2, list(np.clip(Gd + (1 - u)[:, None] * d0, D.lo[:2], D.hi[:2]))
    d = np.array([np.linalg.norm(T[:3, 3] - B[:3, 3]) for T in prev])    # 방금 칠한 줄을 거꾸로 따라 B 근처까지, 끝의 RETRACE_BLEND 점에 걸쳐 서서히 B 쪽 줄로
    m = int(np.argmin(d))
    if m >= len(prev) - 2 or d[m] > RETRACE_MAX:
        return None
    off = B[:3, 3] - prev[m][:3, 3]
    idx = list(range(len(prev) - 2, m, -1))
    if not idx:
        return None
    nb = min(RETRACE_BLEND, len(idx))
    hp = [shifted(prev[i], smooth((n - (len(idx) - 1 - nb)) / (nb + 1)) * off) for n, i in enumerate(idx)]
    Gp = D.g_path(prev, cur[0], cur[1])
    Gd = np.array([Gp[i] for i in idx])
    d0, d1 = g_start - Gp[-1], G[0] - Gd[-1]
    u = (np.arange(len(hp)) + 1) / (len(hp) + 1)
    return hp, -2, list(np.clip(Gd + (1 - u)[:, None] * d0 + u[:, None] * d1, D.lo[:2], D.hi[:2]))
MAX_PARTIAL = 6                   # 선별에서 일부 점이 빠진 후보 중 시도하는 수 (빠진 점이 적은 순서)
_W = {}


SKIP = set()                      # --skip: 계획에서 빼는 원본 CSV 점 번호 (닿지 않는 점). CSV 파일은 그대로 두고 계획만 건너뛴다
ROW_ONLY = None


def targets(cell):
    """원본 CSV 의 줄별 목표. SKIP 에 든 점은 뺀다 (줄 끝의 닿지 않는 점을 건너뛰는 용도)."""
    rows = load_targets(cell)
    if ROW_ONLY is not None:
        rows = [r for r in rows if r['row'] == ROW_ONLY]
        if not rows:
            raise ValueError(f'CSV에 {ROW_ONLY}번 줄이 없습니다.')
    if not SKIP:
        return rows
    out = []
    for r in rows:
        keep = np.array([int(i) not in SKIP for i in r['csv_idx']])
        if keep.any():
            out.append({'row': r['row'], 'T': r['T'][keep], 'paint': r['paint'][keep], 'csv_idx': r['csv_idx'][keep]})
    return out


def _winit(urdf, gun, gen):
    """워커: 셀과 계산기를 한 번 만들어 둔다 (줄 시도 병렬 + 간선 인증 겸용). gen = 현재 유효한 작업 묶음 번호 (공유)."""
    warnings.filterwarnings('ignore')
    cell = pb._make(urdf, gun)
    pb._W['F'] = pb.Finisher(pb.Planner(cell, log=lambda *a: None))
    rows = targets(cell)
    _W.update(D=Dec(cell), rows=rows, zsafe=max(T[2, 3] for r in rows for T in r['T']) + .10, gen=gen, mine=-1)


def _stale():
    """워커 안에서: 이 작업이 이미 필요 없어졌는가 (앞 순위 시도가 성공해 묶음 번호가 넘어감)."""
    return _W.get('gen') is not None and _W['gen'].value != _W['mine']


def row_ents(row):
    return [(T, 'target', int(row['paint'][i]), row['row'], int(row['csv_idx'][i])) for i, T in enumerate(row['T'])]


def try_hop_row(D, rows, zsafe, r, ph, R, v, x_last, c_last, cur):
    """직전 줄 끝 상태(x_last)에서 이동 방식 v 로 r 번 줄에 들어가 줄 전체를 팔만 풀어 본다 (겐트리 후보 = (ph, R))."""
    row = rows[r]
    Ts = row['T']
    G = D.g_path(Ts, ph, R)
    hs = hop_seq(D, rows, r, v, zsafe, cur, (ph, R), x_last[:2], G)
    if hs is None:
        return False, 'not applicable', None, 0, 0
    hp, tag, Gh = hs
    n = len(hp)
    X2, C2 = [x_last], [c_last]
    ok, why = advance(D, X2, C2, hp + list(Ts), [(T, 'transit', 0, tag, -1) for T in hp] + row_ents(row), list(Gh) + list(G),
                      strict_n=n if v[0] in ('retrace', 'lane') else 0, free_n=n if v[0] == 'force' else 0)
    return ok, why, (X2[1:], C2[1:]) if ok else None, len(X2) - 1, n


def _wtask(a):
    gid, r, ph, R, v, x_last, c_last, sig, cur = a
    _W['mine'] = gid
    if _stale():
        return False, 'aborted', None, 0, 0, 0, 0, 0.0
    D = _W['D']
    D.sig, D._cache = sig, None
    n0, r0, t0 = D.n_ik, D.n_repair, time.perf_counter()
    out = try_hop_row(D, _W['rows'], _W['zsafe'], r, ph, R, v, x_last, c_last, cur)
    return out + (D.n_ik - n0, D.n_repair - r0, time.perf_counter() - t0)


def _wring(a):
    """이동 구간 국소 보정의 한 방향: 겐트리 자리 g2 에서 위치만 맞추는 팔 풀이 + 거부 판정."""
    T, xp, g2, sig = a
    D = _W['D']
    D.sig, D._cache = sig, None
    _W['mine'] = _W['gen'].value                              # 이 계산은 중간에 그만두지 않는다
    q2, ok2 = D.arm_pos(g2, T, xp[2:])
    x2 = np.r_[g2, q2]
    return x2, (not ok2) or D.reject(x2) is not None


def _wscreen(a):
    r, ph, R, q0, sig = a
    D = _W['D']
    D.sig, D._cache = sig, None
    _W['mine'] = _W['gen'].value
    n0 = D.n_ik
    Ts = _W['rows'][r]['T']
    return screen(D, Ts, D.g_path(Ts, ph, R), q0) + (D.n_ik - n0,)


def plan(D, rows, zsafe, phi, rdes, prior=None, log=None, pool=None, gen=None, workers=1, sig=None, r0=0, partial=None, no_dp=False):
    """줄 단위 계획. 줄마다: 1단계 겐트리 후보 선별 -> 2단계 팔 풀이.
    결정 규칙 (순차 실행 기준): 선별을 통과한 후보를 가까운 순서로, 후보 안에서는 이동 방식을 파란 직선부터 차례로 시도해 처음
    성공한 것을 쓴다. 이동은 됐는데 줄 안에서 막히면 그 후보의 남은 이동 방식은 건너뛴다. 통과 후보가 다 실패하면 선별에서 빠진
    점이 적은 후보 6개를 같은 식으로. pool 이 있으면 여러 조합을 워커에서 동시에 풀되, 결과는 위 순서로만 해석하므로
    어느 워커가 먼저 끝나든 순차 실행과 같은 계획이 나온다."""
    log = [] if log is None else log
    X = C = None
    init = (phi, min(CAND_R, key=lambda v: abs(v - rdes)))
    cur = init

    def accept(nodes, entry):
        nonlocal X, C
        X += nodes[0]
        C += nodes[1]
        log.append(entry)

    def run_tasks(r, order, hops, seed):
        """order = 후보 (우선순위 순), seed = 이미 푼 결과 {(후보 번호, 이동 방식 번호): 결과}.
        반환 (성공한 (후보, 이동 방식, 노드) 또는 None, 마지막 실패 기록)."""
        res = dict(seed)

        def skipped(ci, vi):                                   # 앞선 이동 방식이 '이동은 됐고 줄 안에서 막힘' 이면 건너뜀
            return any((ci, u) in res and not res[(ci, u)][0] and res[(ci, u)][3] > res[(ci, u)][4] + 2 for u in range(vi))

        def walk():
            """우선순위 순서로 결과를 읽는다: ('ok', key) / ('fail', None) / ('wait', None), 그리고 읽은 key 들."""
            seen = []
            for ci in range(len(order)):
                for vi in range(len(hops)):
                    k = (ci, vi)
                    if k not in res:
                        return 'wait', None, seen
                    seen.append(k)
                    if res[k][0]:
                        return 'ok', k, seen
                    if res[k][3] > res[k][4] + 2:
                        break
            return 'fail', None, seen

        keys = [(ci, vi) for ci in range(len(order)) for vi in range(len(hops))]
        if pool is None:
            while True:
                st, k, seen = walk()
                if st != 'wait':
                    break
                k = next(k for k in keys if k not in res and not skipped(*k))
                c, v = order[k[0]], hops[k[1]]
                t0 = time.perf_counter()
                res[k] = try_hop_row(D, rows, zsafe, r, c[0], c[1], v, X[-1], C[-1], cur) + (0, 0, time.perf_counter() - t0)
        else:
            gen.value += 1
            gid = gen.value
            pending = {}
            while True:
                for k in [k for k, ar in pending.items() if ar.ready()]:
                    res[k] = pending.pop(k).get()
                st, k, seen = walk()
                if st != 'wait':
                    break
                for k in keys:                                 # 빈 워커에 다음 순위 조합을 채운다
                    if len(pending) >= workers:
                        break
                    if k in res or k in pending or skipped(*k):
                        continue
                    c, v = order[k[0]], hops[k[1]]
                    pending[k] = pool.apply_async(_wtask, ((gid, r + r0, c[0], c[1], v, X[-1], C[-1], D.sig, cur),))   # r0: 워커는 전체 줄 목록 기준
                time.sleep(.002)
            gen.value += 1                                     # 아직 돌고 있는 뒤 순위 시도는 스스로 그만둔다
        last = None
        for kk in seen:
            ok, why, nodes, reached, n, nik, nrep, dt = res[kk]
            c, v = order[kk[0]], hops[kk[1]]
            if kk not in seed:
                D.n_ik += nik
                D.n_repair += nrep
            if VERBOSE:
                print(f"    try {c} {v}: {why}, reached {reached} (hop {n}) {dt:.1f}s", file=sys.stderr, flush=True)
            if not ok:
                last = {'into_row': rows[r]['row'], 'phi': c[0], 'rdes': c[1], 'variant': v, 'reached': reached, 'why': why}
        return ((order[k[0]], hops[k[1]], res[k][2]) if st == 'ok' else None), last

    def dp_attempt(r, row, pri):
        """2차 방법: 줄 사이 이동 + 줄을 한 덩어리로, 점마다 닿는 겐트리 자리 지도 위에서 경로를 고른 뒤 팔만 푼다."""
        nonlocal X, C
        Ts = row['T']
        A = rows[r - 1]['T'][-1]
        last = None
        for v in hop_list():
            if v[0] in ('retrace', 'arc', 'force', 'lane'):
                continue
            hp = pc.hop_line(D, A, Ts[0], 0.0)[0] if v[0] == 'line' else pc.hop_poses(A, Ts[0], zsafe, v[1])
            n = len(hp)
            seq = hp + list(Ts)
            G, info = dp_gantry(D, seq, X[-1][2:], X[-1][:2], init[0], n_pos=n if v[0] == 'line' else 0)
            if G is None:
                last = {'into_row': row['row'], 'phi': 'dp', 'rdes': 'dp', 'variant': v, 'why': f'no reachable gantry cell at node {info} of {len(seq)}'}
                continue
            tag = -1 if v[0] == 'line' else -2
            X2, C2 = list(X), list(C)
            ok, why = advance(D, X2, C2, seq, [(T, 'transit', 0, tag, -1) for T in hp] + row_ents(row), list(G))
            if ok:
                X, C = X2, C2
                log.append({'into_row': row['row'], 'hop_variant': v, 'phi': 'dp', 'rdes': 'dp', 'screen_miss': 0,
                            'dp_cells': [list(c) for c in info]})
                return None
            last = {'into_row': row['row'], 'phi': 'dp', 'rdes': 'dp', 'variant': v, 'reached': len(X2) - len(X), 'why': why}
        return last

    for r, row in enumerate(rows):
        Ts = row['T']
        ents = row_ents(row)
        pri = (prior or {}).get(row['row'])
        t_row = time.perf_counter()
        dp_first = pri is not None and pri.get('phi') == 'dp'    # 레퍼런스에서 이 줄을 지도+DP 로 풀었으면 그것부터
        cands = candidates(cur, init, None if pri is None or dp_first else (pri['phi'], pri['rdes']))
        if r == 0:                                             # 첫 줄: 시작 자세를 정하고 줄을 푼다 (이동 없음)
            done, last = False, None
            for ph, R in cands:
                G = D.g_path(Ts, ph, R)
                x0 = D.start(Ts[0], G[0], sig)
                if x0 is None:
                    continue
                okn, n, cost = screen(D, Ts, G, x0[2:])
                if okn < n:
                    continue
                sig_saved, D.sig = D.sig, pc.signature(D.c, x0)
                X2, C2 = [x0], [ents[0]]
                ok, why = advance(D, X2, C2, list(Ts[1:]), ents[1:], list(G[1:]))
                if ok:
                    X, C, done, cur = X2, C2, True, (ph, R)
                    log.append({'row': row['row'], 'phi': ph, 'rdes': R, 'screen_miss': 0})
                    break
                D.sig = sig_saved
                last = {'row': row['row'], 'phi': ph, 'rdes': R, 'why': why}
            if VERBOSE:
                print(f"[row {row['row']}] {'ok' if done else 'FAIL'} {time.perf_counter() - t_row:.1f}s -> {log[-1] if done else last}", file=sys.stderr, flush=True)
            if not done:
                return None, None, log + [last or {'fail': 'no start'}]
            stat = (D.n_ik, D.n_repair)
            continue
        if dp_first and dp_attempt(r, row, pri) is None:
            continue
        hops = hop_list()
        if pri is not None and pri.get('hop') is not None:
            hops = [tuple(pri['hop'])] + [v for v in hops if tuple(v) != tuple(pri['hop'])]
        scr = {}                                               # 후보 -> (빠진 점 수, 팔 관절 변화 합)

        def do_screen(cs):
            """1단계: 겐트리 후보 선별 (후보마다 팔 6축 IK 를 일부 점에서 따로 풂)."""
            if pool is not None and len(cs) > 2:
                res = pool.map(_wscreen, [(r + r0, c[0], c[1], X[-1][2:], D.sig) for c in cs], chunksize=1)
                for c, (okn, n, cost, nik) in zip(cs, res):
                    scr[c] = (n - okn, cost)
                    D.n_ik += nik
            else:
                for c in cs:
                    okn, n, cost = screen(D, Ts, D.g_path(Ts, c[0], c[1]), X[-1][2:])
                    scr[c] = (n - okn, cost)

        got, last, first = None, None, None
        for c in cands[:3]:                                    # 쉬운 줄: 가까운 후보부터 하나씩, 처음 통과한 후보를 바로 시도
            do_screen([c])
            if scr[c][0] == 0:
                t0 = time.perf_counter()
                out = try_hop_row(D, rows, zsafe, r, c[0], c[1], hops[0], X[-1], C[-1], cur) + (0, 0, time.perf_counter() - t0)
                if out[0]:
                    got = (c, hops[0], out[2])
                    if VERBOSE:
                        print(f"    try {c} {hops[0]}: ok, reached {out[3]} (hop {out[4]}) {out[7]:.1f}s", file=sys.stderr, flush=True)
                first = (c, out)
                break
        if got is None:                                        # 어려운 줄: 후보 전체를 선별하고 조합을 동시에 시도
            do_screen([c for c in cands if c not in scr])
            order = [c for c in cands if scr[c][0] == 0] + sorted((c for c in cands if scr[c][0] > 0), key=lambda c: scr[c])[:MAX_PARTIAL]
            got, last = run_tasks(r, order, hops, {} if first is None else {(order.index(first[0]), 0): first[1]})
        if got is not None:
            c, v, nodes = got
            accept(nodes, {'into_row': row['row'], 'hop_variant': v, 'phi': c[0], 'rdes': c[1], 'screen_miss': scr[c][0]})
            cur = c
        else:                                                  # 2차: 점마다 닿는 겐트리 자리 지도 + 동적계획법
            last2 = {'into_row': row['row'], 'why': 'skipped (posture switch follows)'} if no_dp else dp_attempt(r, row, pri)
            if last2 is not None:
                if VERBOSE:
                    print(f"[row {row['row']}] FAIL {time.perf_counter() - t_row:.1f}s -> {last2} / {last}", file=sys.stderr, flush=True)
                if partial is not None:                        # 여기까지 푼 줄들 (자세 전환용)
                    partial.update(X=list(X), C=list(C), log=list(log), sig=D.sig, rows_done=r, n_ik=stat[0], n_repair=stat[1])
                return None, None, log + [last, last2]
        stat = (D.n_ik, D.n_repair)
        if VERBOSE:
            n_full = sum(1 for c in scr if scr[c][0] == 0)
            print(f"[row {row['row']}] ok {time.perf_counter() - t_row:.1f}s screened {len(scr)} full {n_full} -> {log[-1]}", file=sys.stderr, flush=True)
    return np.array(X), C, log


def refine(D, X, C):
    """간선 중간(u=0.1..0.9)이 경로에서 1 mm / 0.2 deg 를 넘으면 중점을 넣는다 (겐트리 = 양 끝 평균, 팔만 풂)."""
    outX, outC, ins, worst = [X[0]], [C[0]], 0, [0.0, 0.0]

    def rec(xa, xb, Ta, Tb, depth, pos_only):
        nonlocal ins
        w = [0.0, 0.0]
        for u in pc.U_CHECK:
            p, R = D.c.tcp((1 - u) * xa + u * xb)
            e = pose_err(p, R, interp_pose(Ta, Tb, u))
            w = [max(w[0], e[0]), max(w[1], 0.0 if pos_only else e[1])]
        good = w[0] <= POS_TOL and w[1] <= ANG_TOL
        if good or depth >= 8:
            worst[0], worst[1] = max(worst[0], w[0]), max(worst[1], w[1])
            return [] if good else None
        Tm = interp_pose(Ta, Tb, .5)
        xm, why, _ = D.node(Tm, xa, (xa[:2] + xb[:2]) / 2, pos_only)
        if why is not None:
            return None
        L, Rr = rec(xa, xm, Ta, Tm, depth + 1, pos_only), rec(xm, xb, Tm, Tb, depth + 1, pos_only)
        if L is None or Rr is None:
            return None
        ins += 1
        return L + [(xm, Tm)] + Rr

    for k in range(len(X) - 1):
        ka, kb = C[k][1], C[k + 1][1]
        line_hop = 'transit' in (ka, kb) and -2 not in (C[k][3], C[k + 1][3])
        pos_only = line_hop and kb != 'target'                # 줄로 들어가는 마지막 구간은 방향까지 맞춰 이어 준다
        Ta = fk_pose(D, X[k]) if ka == 'transit' and C[k][3] in (-1, -5) else C[k][0]       # 위치만 푼 노드는 실제 자세가 기준
        Tb = fk_pose(D, X[k + 1]) if kb == 'transit' and C[k + 1][3] in (-1, -5) else C[k + 1][0]
        forced = (ka == 'transit' and C[k][3] == -5) or (kb == 'transit' and C[k + 1][3] == -5)
        D.no_coll = forced                                    # 강제 직선 구간은 충돌을 보지 않는다
        if forced and kb == 'target':
            pos_only = True
        mids = rec(X[k], X[k + 1], Ta, Tb, 0, pos_only)
        D.no_coll = False
        if mids is None:
            return None, None, {'fail_edge': k, 'kind': ka, 'row': C[k][3]}
        kind = 'mid' if ka in ('target', 'mid') and kb in ('target', 'mid') else 'transit'
        paint = int(C[k][2] and C[k + 1][2])
        row = C[k][3] if kind == 'mid' else (-5 if forced else -2 if -2 in (C[k][3], C[k + 1][3]) else -1)
        for xm, Tm in mids:
            outX.append(xm)
            outC.append((Tm, kind, paint, row, -1))
        outX.append(X[k + 1])
        outC.append(C[k + 1])
    return np.array(outX), outC, {'inserted': ins, 'interp_max_mm': worst[0] * 1000, 'interp_max_deg': math.degrees(worst[1])}


def arm_change(D, g, qa, qb):
    """겐트리 g 에서 팔 자세 qa -> qb (팔만 움직임). 한 번에 안 되면 관절 묶음(1축 / 2,3축 / 손목)을 차례로 움직인다.
    반환: 중간 자세 목록 (없으면 []), 불가능하면 None."""
    def ok(u, v):
        return D.c.certify_edge(np.r_[g, u], np.r_[g, v], CERT_CLEAR)[0]
    if ok(qa, qb):
        return []
    groups = ([0], [1, 2], [3, 4, 5])
    for order in ((0, 1, 2), (2, 1, 0), (1, 0, 2), (0, 2, 1), (1, 2, 0), (2, 0, 1)):
        q, via, good = np.array(qa, float), [], True
        for gi in order[:-1]:
            qn = q.copy()
            qn[groups[gi]] = qb[groups[gi]]
            if not D.free(np.r_[g, qn]) or not ok(q, qn):
                good = False
                break
            via.append(qn)
            q = qn
        if good and ok(q, qb):
            return via
    return None


def link(D, a, b):
    """홈 <-> 접근 자세: 관절 보간. 겐트리만 움직이는 구간과 팔만 움직이는 구간으로 나눈 후보 중 인증되는 첫 경로."""
    ga, gb, qa, qb = a[:2], b[:2], a[2:], b[2:]

    def gantry_ok(g0, g1, q):
        return np.abs(g1 - g0).max() < 1e-9 or D.c.certify_edge(np.r_[g0, q], np.r_[g1, q], CERT_CLEAR)[0]
    # 팔 자세를 바꿀 겐트리 자리 후보: 양 끝, 그리고 셀 안 격자 (도착 쪽에서 가까운 순)
    xs = np.linspace(D.lo[0] + .3, D.hi[0] - .3, 9)
    ys = np.linspace(D.lo[1] + .3, D.hi[1] - .3, 7)
    grid = sorted((np.array([x, y]) for x in xs for y in ys), key=lambda g: np.linalg.norm(g - gb) + np.linalg.norm(g - ga))
    for gs in [gb, ga] + grid:
        # a --(겐트리만)--> gs --(팔만)--> gs --(겐트리만)--> b.  직선으로 안 되면 x 먼저 / y 먼저 꺾어서
        for leg_in in ([gs], [np.array([gs[0], ga[1]]), gs], [np.array([ga[0], gs[1]]), gs]):
            pts = [ga] + leg_in
            if any(not D.free(np.r_[p, qa]) for p in leg_in) or not all(gantry_ok(u, v, qa) for u, v in zip(pts, pts[1:])):
                continue
            for leg_out in ([gb], [np.array([gb[0], gs[1]]), gb], [np.array([gs[0], gb[1]]), gb]):
                pts2 = [gs] + leg_out
                if any(not D.free(np.r_[p, qb]) for p in pts2[:-1]) or not all(gantry_ok(u, v, qb) for u, v in zip(pts2, pts2[1:])):
                    continue
                via = arm_change(D, gs, qa, qb)
                if via is None:
                    break                                     # 이 자리에서는 팔 자세를 못 바꿈 -> 다른 자리
                ch = [np.r_[p, qa] for p in pts] + [np.r_[gs, q] for q in via] + [np.r_[p, qb] for p in pts2]
                ch = [x for k, x in enumerate(ch) if k == 0 or np.abs(x - ch[k - 1]).max() > 1e-9]
                return ch[1:-1]
            else:
                continue
            break
    return None


RETRACT_D = (.25, .18, .12, .08, .05)


def retract(D, x, T, dist):
    """점 x (포즈 T) 에서 분사축 반대로 dist 만큼 물러남: 겐트리 고정, 팔만. 반환: 가까운 쪽부터의 노드들 (실패 None)."""
    P0, P1 = pc.pose(T[:3, 3], T[:3, :3]), pc.pose(T[:3, 3] - dist * T[:3, 2], T[:3, :3])
    xs, xp = [], x
    for Tk in pc.line(P0, P1) + [P1]:
        xp, why, _ = D.node(Tk, xp, xp[:2])
        if why is not None:
            return None
        xs.append(xp)
    return xs


def densify(nodes):
    """이웃 노드 사이 팔 관절 변화가 8도를 넘지 않게 관절 직선 위에 점을 더 넣는다 (경로는 그대로)."""
    out = [nodes[0]]
    for u, v in zip(nodes, nodes[1:]):
        n = max(1, int(math.ceil(np.degrees(np.abs(v[2:] - u[2:]).max()) / MAX_STEP_DEG)))
        out += [u + (v - u) * i / n for i in range(1, n + 1)]
    return out


def connect(D, xa, Ta, sig_a, xb, Tb, sig_b):
    """자세 전환 이동: xa 에서 분사축 반대로 물러남(팔만) -> 관절 보간 이동(겐트리만 / 팔만 구간) -> xb 로 접근(팔만).
    반환: xa 와 xb 사이 노드들 (없으면 None)."""
    D.sig = sig_a
    rets = [r for r in (retract(D, xa, Ta, d) for d in RETRACT_D) if r is not None]
    D.sig = sig_b
    apps = [r for r in (retract(D, xb, Tb, d) for d in RETRACT_D) if r is not None]
    for ret in rets:
        for app in apps:
            v = link(D, ret[-1], app[-1])
            if v is not None:
                return densify([xa] + ret + v + app[::-1] + [xb])[1:-1]
    return None


def add_home(D, X, C, home_arm, sigs=None):
    home = np.r_[D.hi[:2], np.radians(home_arm)]
    if not D.free(home):
        return None, None, 'home pose collides'
    if sigs is not None:
        D.sig = sigs[0]
    apps = [r for r in (retract(D, X[0], C[0][0], d) for d in RETRACT_D) if r is not None]
    if sigs is not None:
        D.sig = sigs[1]
    rets = [r for r in (retract(D, X[-1], C[-1][0], d) for d in RETRACT_D) if r is not None]
    vin = next(((app, v) for app in apps for v in [link(D, home, app[-1])] if v is not None), None)
    vout = next(((ret, v) for ret in rets for v in [link(D, ret[-1], home)] if v is not None), None)
    if vin is None or vout is None:
        return None, None, f"home link failed ({'in' if vin is None else ''} {'out' if vout is None else ''})"
    (app, v_in), (ret, v_out) = vin, vout
    H, Tn = (None, 'home', 0, -1, -1), (None, 'transit', 0, -1, -1)
    nX = [home] + v_in + app[::-1] + list(X) + ret + v_out + [home]
    nC = [H] + [Tn] * (len(v_in) + len(app)) + list(C) + [Tn] * (len(ret) + len(v_out)) + [H]
    return np.array(nX), nC, 'ok'


def plan_with_switch(D, rows, zsafe, a, pool, gen, tried, parts):
    """사진 자세(∩) 한 분기로 끝까지 못 갈 때: 막힌 줄 앞까지는 ∩ 로 푼 것을 그대로 쓰고, 막힌 줄부터는 다른 분기로 풀어
    그 사이를 자세 전환 이동(물러남 -> 관절 보간 -> 접근)으로 잇는다. 전환은 한 번.
    반환: (X, C, 정보, 기록, 시작 분기, 끝 분기, 전환 기록) 또는 (None, 사유)."""
    global POSTURE
    best = max(tried, key=lambda t: t.get('fail_idx', -1))     # 가장 멀리 간 분기 (같으면 먼저 시도한 것)
    k = best.get('fail_idx', -1)
    if k <= 0:
        return None, 'photo posture fails at the first row'
    kw = dict(pool=pool, gen=gen, workers=a.workers)
    part = parts[best['sig']]
    sig1, log1 = part['sig'], part['log']
    D.sig = sig1
    X1, C1, r1 = refine(D, np.array(part['X']), part['C'])
    if X1 is None:
        return None, f'part 1 refine failed {r1}'
    D.n_ik, D.n_repair = part['n_ik'], part['n_repair']        # 통계는 실제로 쓴 계획 기준
    failed = {tuple(t['sig']) for t in tried if t.get('fail_idx') == k}
    order = [sg for sg in sorted(((s0, s1, s2) for s0 in (1, -1) for s1 in (1, -1) for s2 in (1, -1)),
                                 key=lambda sg: sum(u != v for u, v in zip(sg, sig1))) if sg not in failed and sg != tuple(sig1)]
    keep, POSTURE = POSTURE, ''                               # 막힌 줄부터는 자세 조건 없이
    try:
        for sg in order:
            D.sig = None
            X2, C2, log2 = plan(D, rows[k:], zsafe, a.phi, a.rdes, None, sig=sg, r0=k, **kw)
            if VERBOSE:
                print(f"[switch before row {rows[k]['row']} -> branch {sg}] {'ok' if X2 is not None else 'FAIL'}", file=sys.stderr, flush=True)
            if X2 is None:
                continue
            sig2 = D.sig
            X2, C2, r2 = refine(D, X2, C2)
            if X2 is None:
                continue
            con = connect(D, X1[-1], C1[-1][0], sig1, X2[0], C2[0][0], sig2)
            if con is not None:
                break
            if VERBOSE:
                print(f"    posture-change move not found for branch {sg}", file=sys.stderr, flush=True)
        else:
            return None, f"no other branch can do rows from {rows[k]['row']} with a connecting move"
    finally:
        POSTURE = keep
    Tn = (None, 'transit', 0, -1, -1)
    X = np.array(list(X1) + con + list(X2))
    C = list(C1) + [Tn] * len(con) + list(C2)
    log2[0] = {**log2[0], 'into_row': log2[0]['row'], 'hop_variant': ('switch', 0.0)}
    info = {'inserted': r1['inserted'] + r2['inserted'], 'interp_max_mm': max(r1['interp_max_mm'], r2['interp_max_mm']),
            'interp_max_deg': max(r1['interp_max_deg'], r2['interp_max_deg'])}
    dq = np.degrees(X2[0][2:] - X1[-1][2:])
    sw = [{'before_row': rows[k]['row'], 'from': list(sig1), 'to': list(sig2), 'nodes': len(con),
           'arm_joint_change_deg': [round(float(v), 1) for v in dq], 'gantry_move_m': round(float(np.linalg.norm(X2[0][:2] - X1[-1][:2])), 3),
           'photo_rows': [r['row'] for r in rows[:k]], 'photo_points': int(sum(len(r['T']) for r in rows[:k]))}]
    return X, C, info, log1 + log2, sig1, sig2, sw


def run(a):
    t0 = time.perf_counter()
    import multiprocessing as mp
    ctx = mp.get_context('fork')
    gen = ctx.RawValue('i', 0)
    pool = ctx.Pool(a.workers, initializer=_winit, initargs=(a.urdf, a.gun, gen)) if a.workers > 1 else None   # 줄 시도 + 간선 인증
    cell = pb._make(a.urdf, a.gun)
    rows = targets(cell)
    D = Dec(cell)
    zsafe = max(T[2, 3] for r in rows for T in r['T']) + .10
    t_load = time.perf_counter() - t0
    prior = None
    if a.prior:
        try:
            prior = {int(k): v for k, v in json.loads(Path(a.prior).read_text())['row_hints'].items()}
        except Exception as e:
            print(f'[prior ignored: {e}]', file=sys.stderr)
    base = {'method': 'decoupled (gantry path first, then arm 6-axis)', 'gun': a.gun, 'phi_w_deg': a.phi, 'rdes': a.rdes,
            'urdf': str(Path(a.urdf).resolve()), 'home_arm': a.home_arm, 'transit': 'line',
            'j3_min_deg': J3_MIN, 'j5_min_deg': J5_MIN, 'posture': POSTURE, 'ends': ENDS, 'hops': HOP_MODE, 'phi_max_deg': a.phi_max,
            'center_back_m': CENTER_BACK}
    t1 = time.perf_counter()
    D.ring_pool = pool
    # 시작 자세 분기: 지정(--sig) 또는 photo 면 첫 점에서 가능한 위쪽(∩) 분기를 점수 순으로 차례로 (끝까지 못 가면 다음 분기)
    sig_list = [None]
    if a.sig:
        sig_list = [tuple(int(v) for v in a.sig.split(','))]
    elif POSTURE == 'photo':
        init = (a.phi, min(CAND_R, key=lambda v: abs(v - a.rdes)))
        R0, opts = rows[0]['T'], {}
        for ph, R in candidates(init, init)[:6]:
            for sg, (sc, _) in D.start(R0[0], D.g_path(R0, ph, R)[0], all_sigs=True):
                opts[sg] = min(opts.get(sg, 9e9), sc)
        sig_list = sorted(opts, key=opts.get) or [None]
    tried, parts = [], {}
    for sg in sig_list:
        D.sig = None
        part = {}
        X, C, log = plan(D, rows, zsafe, a.phi, a.rdes, prior, pool=pool, gen=gen, workers=a.workers, sig=sg, partial=part,
                         no_dp=POSTURE == 'photo' and a.switch)       # 전환 모드: 막히면 느린 2차 방법 대신 자세 전환으로
        tried.append({'sig': sg, 'ok': X is not None})
        if X is None:                                          # 어느 줄에서 막혔는가
            fr = next((l.get('into_row', l.get('row')) for l in reversed(log) if l and 'why' in l and l.get('into_row', l.get('row')) is not None), None)
            tried[-1].update(fail_row=fr, fail_idx=part.get('rows_done', -1))
            parts[sg] = part
            if POSTURE == 'photo' and a.switch and part.get('rows_done', 0) > 0:
                break                                          # 전환 모드: 첫 위쪽 분기가 막히면 바로 전환
        if VERBOSE:
            print(f"[start branch {sg}] {'ok' if X is not None else 'FAIL'}", file=sys.stderr, flush=True)
        if X is not None:
            break
    base['start_branches_tried'] = tried
    sigs, t_ref = None, 0.0
    if X is None and POSTURE == 'photo' and a.switch and sig_list != [None]:
        # 사진 자세 한 분기로는 끝까지 못 감 -> 막힌 줄부터 다른 분기 (자세 전환 1회, 사용자가 켠 경우만)
        res = plan_with_switch(D, rows, zsafe, a, pool, gen, tried, parts)
        if res[0] is None:
            D.ring_pool = None
            return {**base, 'status': 'fail', 'stage': 'chain', 'why': res[1], 'log': log[-3:], 'timing_s': {'load': t_load, 'chain': time.perf_counter() - t1}}
        X, C, rinfo, log, s_in, s_out, base['posture_switches'] = res
        sigs = (s_in, s_out)
    D.ring_pool = None
    t_chain = time.perf_counter() - t1
    if X is None:
        return {**base, 'status': 'fail', 'stage': 'chain', 'log': log[-3:], 'timing_s': {'load': t_load, 'chain': t_chain}}
    if sigs is None:
        t2 = time.perf_counter()
        X, C, rinfo = refine(D, X, C)
        t_ref = time.perf_counter() - t2
        if X is None:
            return {**base, 'status': 'fail', 'stage': 'refine', 'info': rinfo, 'timing_s': {'load': t_load, 'chain': t_chain}}
    t3 = time.perf_counter()
    Xh, Ch, why = add_home(D, X, C, [float(v) for v in a.home_arm.split(',')], sigs)
    t_home = time.perf_counter() - t3
    if Xh is None:
        return {**base, 'status': 'fail', 'stage': 'home', 'why': why, 'timing_s': {'load': t_load, 'chain': t_chain}}
    t4 = time.perf_counter()
    if a.certify:                                              # 전 구간 충돌 보장 (5 mm): 계획 시간이 늘어 기본은 끔. 노드마다 15 mm 판정은 그대로
        bad, lo = pc.certify_all(cell, Xh, a.workers, a.urdf, a.gun, pool)
    else:
        bad, lo = [], None
    if pool is not None:
        pool.close()
    t_cert = time.perf_counter() - t4
    bad_edge = np.zeros(len(Xh), bool)                         # bad_edge[k]: k -> k+1 간선이 충돌 인증을 통과하지 못함
    bad_edge[list(bad)] = True
    forced = [k for k in range(len(Xh) - 1) if -5 in (Ch[k][3], Ch[k + 1][3]) and 'transit' in (Ch[k][1], Ch[k + 1][1])]
    if forced:                                                 # 강제 직선 구간 (충돌을 보지 않고 파란 직선 그대로 간 곳)
        base['forced_straight_links'] = {'edges': len(forced), 'colliding_edges': int(bad_edge[forced].sum()) if a.certify else None,
                                         'colliding_edges_elsewhere': int(bad_edge.sum() - bad_edge[forced].sum()) if a.certify else None,
                                         'into_rows': [h.get('into_row') for h in log if h.get('hop_variant') and h['hop_variant'][0] == 'force']}
    m = pc.metrics(cell, Xh, Ch, rows)
    pf = np.array([c[2] for c in Ch])
    times = pc.time_param_smooth(cell, Xh, pf)
    P = np.array([cell.tcp(x)[0] for x in Xh])
    m.update(rinfo)
    m.update(pc.motion_stats(cell, Xh, Ch))
    m.update({'certified_lower_bound_mm': lo * 1000 if a.certify else None, 'certified_clearance_lower_bound_mm': lo * 1000 if a.certify else None,
              'uncertified_edges': len(bad) if a.certify else None, 'edges_certified': bool(a.certify),
              'cycle_time_s': float(times[-1]), 'hop_variants': [h.get('hop_variant') for h in log],
              'arm_ik_solves': D.n_ik, 'gantry_local_repairs': D.n_repair})
    total = sum(len(r['T']) for r in rows)
    ok = not bad and m['coverage'] == f'{total}/{total}'
    if SKIP:                                                   # 건너뛴 점이 있으면 원본 CSV 전체 대비로 적는다
        n_csv = total + len(SKIP)
        m['coverage_of_planned'] = m['coverage']
        m['coverage'] = f'{total}/{n_csv}'
        base.update(skipped_csv_idx=sorted(int(i) for i in SKIP), points_in_csv=n_csv)
    row_hints = {}
    for l in log:
        r_ = l.get('row', l.get('into_row'))
        if r_ is not None and 'why' not in l:
            row_hints[int(r_)] = {'phi': l['phi'], 'rdes': l['rdes'], 'hop': l.get('hop_variant')}
    word = 'verified' if a.certify else 'planned (edges not certified)'      # 전 구간 보장을 안 했으면 '검증' 이라고 하지 않는다
    res = {**base, 'status': (word if not SKIP else f'{word} except {len(SKIP)} skipped points') if ok else 'NOT verified', 'metrics': m, 'row_hints': row_hints, 'used_prior': bool(prior),
           'timing_s': {'load': t_load, 'chain': t_chain, 'refine': t_ref, 'home': t_home, 'certify(parallel)': t_cert,
                        'planning_total': t_chain + t_ref + t_home + t_cert}}
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        tool = cell.tool
        np.savez(out / f'trajectory_{pb.WP}.npz', X=Xh, t=times, names=np.array(pb.GANTRY + pb.ARM), tcp=P, paint=pf,
                 kind=np.array([c[1] for c in Ch]), row=np.array([c[3] for c in Ch]), csv_idx=np.array([c[4] for c in Ch]),
                 tool_names=np.array(list(tool)), tool_values=np.array(list(tool.values())), bad_edge=bad_edge)
        with open(out / f'joints_{pb.WP}.csv', 'w') as f:
            f.write('t,' + ','.join(pb.GANTRY + pb.ARM) + ',gun_lift,gun_swing,paint,kind,row,csv_idx\n')
            for ti, x, c in zip(times, Xh, Ch):
                f.write(f'{ti:.4f},' + ','.join(f'{v:.6f}' for v in x) + f",{tool['gun_lift']},{tool['gun_swing']},{c[2]},{c[1]},{c[3]},{c[4]}\n")
        res.update({'tool_state': tool, 'urdf_sha256': sha(a.urdf), 'csv_sha256': sha(pb.CSV), 'csv': str(Path(pb.CSV).resolve()), 'workpiece': pb.WP, 'axis_lock': AXIS_LOCK, 'gantry_vmax': pb.GANTRY_SPEED, 'gantry_per_axis': pc.GANTRY_PER_AXIS, 'gantry_amax': pc.GANTRY_AMAX, 'tcp_speed': pb.TCP_SPEED,
                    'collision_exclusions': [list(e) for e in cell.exclusions if e[2] != 'adjacent assembly'],
                    'trajectory_sha256': sha(out / f'joints_{pb.WP}.csv')})
        (out / 'validation.json').write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8')
    return res


def main():
    global VERBOSE, J3_MIN, J5_MIN, POSTURE, ENDS, HOP_MODE, CAND_DPHI, R_PULL, SKIP, SMOOTH, AXIS_LOCK, CENTER_BACK, ROW_ONLY
    ap = argparse.ArgumentParser()
    ap.add_argument('--urdf', required=True)
    ap.add_argument('--out', default='')
    ap.add_argument('--wp', default='bumper', choices=sorted(pb.WP_FOLDER), help='계획 대상 작업물 (셀에 다른 작업물이 같이 있으면 그것은 고정 장애물로 본다)')
    ap.add_argument('--row-only', type=int, default=None, help='CSV에서 이 줄만 계획 (0부터)')
    ap.add_argument('--gun', default='on', choices=['on', 'off'])
    ap.add_argument('--phi', type=float, default=0.0, help='겐트리 방위 (손목 기준, 월드 deg). 0 = 겐트리가 +X(홈) 쪽')
    ap.add_argument('--rdes', type=float, default=.80, help='어깨-손목 거리 시작값 [m]')
    ap.add_argument('--home-arm', default='180,0,120,0,-120,0')
    ap.add_argument('--prior', default='', help='이전 계획 validation.json (줄별 겐트리 후보를 먼저 시도)')
    ap.add_argument('--posture', default='', choices=['', 'photo'], help="photo = 현장 사진 자세: 팔꿈치가 어깨-손목 선 위쪽(∩)인 분기 + 뻗은 거리 유지")
    ap.add_argument('--hops', default='default', choices=['default', 'clean', 'curve', 'straight', 'lane'], help="clean = 줄 사이 이동을 파란 직선으로, 안 되면 옆 줄 경로를 따라 (물러나서 돌아가거나 위로 넘어가는 이동 금지)")
    ap.add_argument('--phi-max', type=float, default=60.0, help='겐트리 방위 후보 범위 +- [deg] (15도 간격, 기본 60)')
    ap.add_argument('--r-pull', type=float, default=None, help='줄마다 겐트리 거리 후보를 --rdes 쪽으로 당기는 세기 (기본 0.3; 2 이상이면 줄마다 --rdes 부터 시도 = 팔을 더 뻗음)')
    ap.add_argument('--certify', action='store_true', help='전 구간 충돌 보장 (모든 간선이 5 mm 이상임을 증명, 계획 시간 +2 s 정도). 기본은 끔: 노드마다 15 mm 판정만')
    ap.add_argument('--axis-lock', action='store_true', help='한 줄 안에서는 겐트리 한 축만 따라가고 다른 축은 고정 (줄 사이에서만 옮김)')
    ap.add_argument('--center-back', type=float, default=0.0, help='높은 중앙부에서 +X로 추가 후퇴 [m], 영상 참고용 가정')
    ap.add_argument('--gantry-vmax', type=float, default=None, help='겐트리 축별 최고 속도 [m/s] (주면 X, Y 각각. 기본: 예전 값 1.5 m/s, XY 합성)')
    ap.add_argument('--gantry-amax', type=float, default=None, help='겐트리 축별 최고 가속도 [m/s^2] (기본: 보지 않음)')
    ap.add_argument('--tcp-speed', type=float, default=None, help='건 끝 목표 도장 속도 [m/s] (기본 0.5)')
    ap.add_argument('--smooth', type=int, default=SMOOTH, help='줄 안 겐트리 경로 이동평균 창 [점] (기본 15). 줄이 창보다 짧으면 겐트리가 거의 멈추고 팔만 움직인다 -> 짧은 줄이 많으면 줄인다')
    ap.add_argument('--ends', default='avg', choices=['avg', 'sym'], help="줄 양 끝 겐트리 평활: sym = 끝점을 규칙 값 그대로 (아치 끝이 빠듯한 셀용)")
    ap.add_argument('--switch', action='store_true', help='photo: 한 자세로 끝까지 못 가면 막힌 줄 앞에서 자세를 한 번 바꾼다 (줄 사이, 물러난 뒤 관절 보간)')
    ap.add_argument('--skip', default='', help="닿지 않는 점 목록 json ({'points': [{'csv_idx', 'status'}]} 에서 status != ok, 또는 번호 목록): 그 점은 건너뛰고 계획한다 (CSV 는 그대로)")
    ap.add_argument('--sig', default='', help="시작 자세 분기 강제 '어깨,팔꿈치,손목' (예: -1,1,-1)")
    ap.add_argument('--j3-min', type=float, default=J3_MIN, help='도장 노드 팔꿈치 최소 굽힘 [deg] (0 = 제한 없음)')
    ap.add_argument('--j5-min', type=float, default=J5_MIN, help='도장 노드 손목 특이점 최소 여유 [deg] (0 = 제한 없음)')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--verbose', action='store_true', help='줄마다 선별/시도 횟수와 시간을 stderr 에 출력')
    a = ap.parse_args()
    VERBOSE, J3_MIN, J5_MIN, POSTURE, ENDS, HOP_MODE = a.verbose, a.j3_min, a.j5_min, a.posture, a.ends, a.hops
    R_PULL = a.r_pull
    SMOOTH = a.smooth
    AXIS_LOCK = a.axis_lock
    CENTER_BACK = a.center_back
    ROW_ONLY = a.row_only
    if a.gantry_vmax:
        pb.GANTRY_SPEED, pc.GANTRY_PER_AXIS = a.gantry_vmax, True
    if a.gantry_amax:
        pc.GANTRY_AMAX = a.gantry_amax
    if a.tcp_speed:
        pb.TCP_SPEED = a.tcp_speed
    pb.set_workpiece(a.wp)
    if a.skip:
        sk = json.loads(Path(a.skip).read_text())
        SKIP = {int(q['csv_idx']) for q in sk['points'] if q.get('status') != 'ok'} if isinstance(sk, dict) else {int(i) for i in sk}
    if a.phi_max != 60.0:
        CAND_DPHI = (0,) + tuple(sgn * d for d in range(15, int(a.phi_max) + 1, 15) for sgn in (1, -1))
    print(json.dumps(run(a), indent=1, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
