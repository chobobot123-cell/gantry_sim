#!/usr/bin/env python3
"""연속 사슬 경로 계획 — 전체 경로(줄 + 줄 사이 이동 + 홈)를 하나의 워밍스타트 사슬로 푼다.

요구사항 (2026-10-04)
  * CSV 경로를 표시된 그대로 (직선 구간 포함) 따라간다: 노드 1 mm / 0.2 deg, 노드 사이 u=0.1..0.9 도 같은 기준
  * 로봇 자세가 도중에 바뀌지 않는다: 줄 안에서도, 줄 -> 다음 줄 이동에서도
    -> 처음 한 번만 시드하고 끝까지 직전 해로 이어서 푼다 (줄마다 재시드 없음).
    -> 줄 사이 이동도 카테시안 경로(짧으면 직선, 길면 후퇴-들어올림-이동-접근)를 같은 IK 로 푼다. RRT 없음.
    -> 팔 분기 서명(어깨 앞/뒤, 팔꿈치, 손목)이 전 구간 일정한지, 노드당 관절 변화가 작은지 보고한다.
  * 계획 시간 20 s 이내 (배치 고정 시 1회 계획 기준)

방법
  decoupled : 겐트리 = 규칙(손목중심 W 에서 월드 고정 방위각 phi_w, 어깨->W 거리 R_des, 줄 단위 평활), 팔 6축만 DLS
  wholebody : 8축 resolved-rate. 8열 자코비안 DLS
              + 영공간: 겐트리 -> 규칙 위치(약), 전체 -> 직전 자세(연속성), 가까운 충돌쌍에서 멀어지기

사용법: plan_chain.py --urdf X --gun on --method wholebody [--phi 180] [--rdes .85] [--out DIR]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plan_bumper as pb  # noqa: E402
from plan_bumper import (ArmIK, GantryRule, interp_pose, load_targets, pose_err, sha,  # noqa: E402
                         time_param, CSV, POS_TOL, ANG_TOL, CERT_CLEAR)

NODE_CLEAR = .015
AVOID_ON = .030                 # 이 거리 안이면 영공간 회피 작동
STEP = .04                      # 이동 구간 카테시안 분할 [m]
SHORT_HOP = .20                 # 이 이하면 직선, 초과면 들어올린 경로
LIFT = .12                      # 후퇴량 기본값 [m]
MAX_STEP_DEG = 8.0              # 노드당 팔 관절 변화 상한 (자세 연속성). 넘으면 구간을 잘게 나눠 이어 푼다
SMOOTH = 15                     # 줄 내 겐트리 힌트 이동평균 창
GUIDE_DEG = 0                   # >0 이면 줄 겐트리 안내 곡선을 이 차수 다항식으로 (0 = 이동평균 SMOOTH)
W_GANTRY = 1.0                  # 영공간: 겐트리를 힌트 곡선으로 당기는 세기 (크면 겐트리가 매끄러운 힌트를 더 따라감)
U_CHECK = np.linspace(.1, .9, 9)
HOP_VARIANTS = 5
TRANSIT_MODE = 'line'           # 'line': 줄 끝 -> 다음 줄 시작 직선(파란 선)을 위치로 따라감, 공구 방향은 자유
                                # 'free': 파란 선 대신 최적 이동 = 관절공간 직선 보간 (필요하면 후퇴 -> 관절 이동 -> 접근)
JOINT_HOP_DEG = 4.0
ALLOW_FLIP = False              # 시험용: 줄 사이 관절공간 이동에서 팔 자세(분기) 뒤집기 허용 (기본 금지)
ROW_FAIL_NEXT_HINT = 2           # 이동 뒤 줄 안에서 이만큼 실패하면 이동 방식 대신 겐트리 힌트를 바꾼다 (탐색 시간)             # 관절공간 이동: 노드당 팔 관절 변화
LINE_OFFSETS = [0.0, .03, .05, .08]   # 직선이 범퍼에 이보다 가까우면 그 구간만 바깥으로 비킴 (0 = 정확히 직선)
W_ORI_TRANSIT = .3              # 이동 중 공구 방향: 보간 방향으로 약하게 당김 (영공간)
J5_MIN = math.radians(25.0)     # 손목 특이점(j5=0) 회피: 시작 부호를 유지하며 이만큼 떨어지게
J3_MIN = math.radians(15.0)     # 팔꿈치 특이점(j3=0, 팔이 다 펴짐) 회피: 시작 부호 유지


def pose(p, R):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


ANG_STEP = math.radians(3.0)    # 이동 구간: 자세가 크게 도는 곳은 3 deg 마다 노드


def line(A, B, step=STEP):
    ang = np.linalg.norm(pin.log3(A[:3, :3].T @ B[:3, :3]))
    n = max(1, int(math.ceil(np.linalg.norm(B[:3, 3] - A[:3, 3]) / step)), int(math.ceil(ang / ANG_STEP)))
    return [interp_pose(A, B, k / n) for k in range(1, n)]


def hop_poses(A, B, zsafe, variant):
    """줄 끝 A -> 다음 줄 시작 B 사이 카테시안 경유 포즈 (A, B 제외).
    0 직선 | 1,2 후퇴(분사축 반대) -> 직선 -> 접근 | 3,4 후퇴 -> 위로 넘어감 -> 접근"""
    if variant == 0:
        via = [A, B]
    else:
        lift = LIFT * (1 if variant in (1, 3) else 2)
        A1 = pose(A[:3, 3] - lift * A[:3, 2], A[:3, :3])
        B1 = pose(B[:3, 3] - lift * B[:3, 2], B[:3, :3])
        if variant <= 2:
            via = [A, A1, B1, B]
        else:
            zt = max(zsafe, A1[2, 3], B1[2, 3]) + .05 * (variant - 3)
            A2 = pose(np.r_[A1[:2, 3], zt], A[:3, :3])
            B2 = pose(np.r_[B1[:2, 3], zt], B[:3, :3])
            via = [A, A1, A2, B2, B1, B]
    out = []
    for a, b in zip(via, via[1:]):
        out += line(a, b)
        if b is not via[-1]:
            out.append(b)
    return out


def hop_line(S, A, B, D):
    """줄 끝 A -> 다음 줄 시작 B 직선(파란 선). D>0 이면 범퍼에 D 보다 가까운 점만 바깥으로 비킨다.
    반환: (경유 포즈들, 직선에서 벗어난 최대 거리 [m]). 방향은 보간값(약한 목표)이다."""
    pa, pb_ = A[:3, 3], B[:3, 3]
    poses = line(A, B, step=.02)
    if D > 0 and poses:
        P = np.array([T[:3, 3] for T in poses])
        d, i = S.bumper_kd().query(P)
        Q = S.bumper_kd().data[i]
        off = np.zeros_like(P)
        for k in range(len(P)):
            if d[k] < D:
                v = P[k] - Q[k]
                off[k] = (D - d[k]) * v / max(np.linalg.norm(v), 1e-9)
        if np.any(off):                                   # 비킴을 매끄럽게 (양 끝은 고정)
            w = 5
            sm = np.array([off[max(0, k - w):k + w + 1].mean(0) for k in range(len(off))])
            ramp = np.minimum(1.0, np.minimum(np.arange(1, len(P) + 1), np.arange(len(P), 0, -1)) / 3.0)
            off = np.maximum(np.linalg.norm(off, axis=1), np.linalg.norm(sm, axis=1))[:, None] * \
                np.nan_to_num(sm / np.maximum(np.linalg.norm(sm, axis=1, keepdims=True), 1e-9)) * ramp[:, None]
            for T, o in zip(poses, off):
                T[:3, 3] = T[:3, 3] + o
    dev = 0.0
    u = (pb_ - pa) / max(np.linalg.norm(pb_ - pa), 1e-9)
    for T in poses:
        v = T[:3, 3] - pa
        dev = max(dev, float(np.linalg.norm(v - np.dot(v, u) * u)))
    return poses, dev


def signature(cell, x):
    """6R 분기 서명 (표준): 어깨 = joint_2 축에 수직인 전방 방향에 대한 손목중심의 쪽,
    팔꿈치 = sign(j3) (0 = 팔이 다 펴진 특이점), 손목 = sign(sin j5) (0 = 손목 특이점).
    이 셋이 전 구간 같으면 특이점을 지나지 않은 같은 자세 가족이다."""
    cell.set(x)
    B = cell.data.oMf[cell.model.getFrameId('base_link')].translation
    W = cell.data.oMf[cell.model.getFrameId('link_5')].translation
    a2 = cell.data.oMf[cell.model.getFrameId('link_2')].rotation[:, 2]
    fwd = np.cross([0.0, 0.0, 1.0], a2)
    s = float(np.dot((W - B)[:2], fwd[:2]))
    return (int(np.sign(s)) if abs(s) > 1e-3 else 0, int(np.sign(x[4])), int(np.sign(math.sin(x[6]))))


GANTRY_RULE = 'adaptive'        # 분리 방식 겐트리 규칙: adaptive(손목 기준 방위/거리, 높이차 반영) | fender(TCP_xy + 고정 오프셋 + LPF)
FENDER_OFFSET, FENDER_ALPHA = .45, .25
LOCAL_SEARCH = True             # 분리 방식: IK 실패 시 겐트리 +-80 mm 3x3 재탐색
W_LARM = 0.0                    # 위팔 수평(ㄱ자) 유지 세기
W_ELBOW = 2.0                   # 팔꿈치 위 유지 세기 (ELBOW='up' 일 때)
ELBOW = ''                      # '' 자유 | 'up' 팔꿈치가 손목보다 위 | 'down' 아래 (시작 자세에서 정하고 사슬 내내 유지)


def elbow_up(cell, x):
    cell.set(x)
    return bool(cell.data.oMi[cell.model.getJointId('joint_3')].translation[2] > cell.data.oMf[cell.model.getFrameId('link_5')].translation[2])


class ChainSolver:
    def __init__(self, cell, method, phi, rdes):
        self.c = cell
        self.method = method
        self.ik = ArmIK(cell)
        self.rule = GantryRule(cell, self.ik)
        self.phi = math.radians(phi)
        self.rdes = rdes
        self.lo, self.hi = cell.lower, cell.upper
        self.stats = {'ik_iters': 0, 'avoid_iters': 0}
        self._cache = None
        # 회피 대상 쌍: 로봇(팔/건) <-> 범퍼/프레임/겐트리, 건 <-> 팔(link_2..). 팔 내부 구조쌍은 검사만 하고 회피는 안 함.
        def grp(n):
            if n == 'workpiece':
                return 'wp'
            if n.startswith(('world_base', 'gantry_')):
                return 'cell'
            if n.startswith('gun_'):
                return 'gun'
            if n.startswith('base_link') or n.startswith('link_1'):
                return 'armbase'
            return 'arm'
        self.avoid_mask = np.array([
            {grp(a), grp(b)} in ({'arm', 'wp'}, {'gun', 'wp'}, {'arm', 'cell'}, {'gun', 'cell'}, {'gun', 'arm'})
            for a, b in (cell.pair_names(k) for k in range(len(cell.ia)))])
        self.j5_sign = 0.0          # 0 이면 손목 회피 비활성 (시작 해가 정해지면 설정)
        self.j3_sign = 0.0
        self.keep_wrist = True
        self.hop_free = True        # 이동 노드에서 공구 방향 자유 (파란 직선 모드)

    def g_ref(self, T):
        W = self.rule.wrist(T)
        dz = self.rule.shoulder_z - W[2]
        h = math.sqrt(max(self.rdes ** 2 - dz ** 2, .01)) + pb.SHOULDER_R
        B = W[:2] + h * np.array([math.cos(self.phi), math.sin(self.phi)])
        return np.clip(B - self.ik.base_off, self.lo[:2], self.hi[:2])

    def g_row(self, Ts):
        if GANTRY_RULE == 'fender':
            # 펜더 규칙 기반 (~/gantry METHOD.md 4절): 겐트리 = TCP_xy + 고정 오프셋 -> 행정 클램프 -> 1차 LPF(a=0.25) -> 클램프
            d = FENDER_OFFSET * np.array([math.cos(self.phi), math.sin(self.phi)])
            G, g = [], None
            for T in Ts:
                raw = np.clip(T[:2, 3] + d - self.ik.base_off, self.lo[:2], self.hi[:2])
                g = raw if g is None else FENDER_ALPHA * raw + (1 - FENDER_ALPHA) * g
                G.append(np.clip(g, self.lo[:2], self.hi[:2]))
            return np.array(G)
        G = np.array([self.g_ref(T) for T in Ts])
        if GUIDE_DEG and len(G) > GUIDE_DEG + 2:
            # 매끄러운 안내 곡선: TCP 호 길이에 대한 저차 다항식 맞춤 (분사 방향 변화로 생기는 손목 흔들림 제거)
            P = np.array([T[:3, 3] for T in Ts])
            u = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
            u = u / max(u[-1], 1e-9)
            G = np.stack([np.polyval(np.polyfit(u, G[:, a], GUIDE_DEG), u) for a in (0, 1)], 1)
            return np.clip(G, self.lo[:2], self.hi[:2])
        k = SMOOTH // 2
        G = np.array([G[max(0, i - k):i + k + 1].mean(0) for i in range(len(G))])
        return np.clip(G, self.lo[:2], self.hi[:2])

    def clear(self, x):
        """최소 거리. AVOID_ON 보다 먼 쌍은 AABB 하한으로 대체 (판정 불변: 하한 >= 임계값)."""
        if self._cache is not None and np.array_equal(self._cache[0], x):
            return self._cache[1]
        d = self.c.distances(x, near=AVOID_ON + .005)
        v = float(d.min())
        self._cache = (x.copy(), v, d)
        return v

    def pair_grad(self, x, k):
        """쌍 k 거리와 8관절에 대한 해석 기울기: n . (J2(p2) - J1(p1)), 최근접점/법선 이용."""
        c = self.c
        c.set(x)
        pin.computeJointJacobians(c.model, c.data, c.q)
        pin.updateGeometryPlacements(c.model, c.data, c.geom, c.geom_data)
        res = pin.computeDistance(c.geom, c.geom_data, int(k))
        pr = c.geom.collisionPairs[int(k)]
        n = np.asarray(res.normal)
        g = np.zeros(8)
        for obj_i, p, sgn in ((pr.first, res.getNearestPoint1(), -1.0), (pr.second, res.getNearestPoint2(), 1.0)):
            j = c.geom.geometryObjects[obj_i].parentJoint
            if j == 0:
                continue
            Jj = pin.getJointJacobian(c.model, c.data, j, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            r = np.asarray(p) - c.data.oMi[j].translation
            skew = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
            Jlin = Jj[:3] - skew @ Jj[3:]
            g += sgn * (n @ Jlin[:, c.iv])
        return float(res.min_distance), g

    def elbow_grad(self, x, margin=.10):
        """f = (손목 z - 팔꿈치 z + margin) 가 양수일 때만 f 를 줄이는 기울기 (팔꿈치가 손목보다 margin 이상 위면 0)."""
        c = self.c
        c.q[c.iq] = x
        pin.computeJointJacobians(c.model, c.data, c.q)
        pin.updateFramePlacements(c.model, c.data)
        je = c.model.getJointId('joint_3')
        fw = c.model.getFrameId('link_5')
        f = c.data.oMf[fw].translation[2] - c.data.oMi[je].translation[2] + margin
        if f <= 0:
            return np.zeros(len(x))
        Je = pin.getJointJacobian(c.model, c.data, je, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[2, c.iv]
        Jw = pin.getFrameJacobian(c.model, c.data, fw, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[2, c.iv]
        return f * (Jw - Je)

    def larm_grad(self, x):
        """f = (팔꿈치 z - 어깨 z)^2 / 2 -> 위팔(어깨->팔꿈치)이 수평이 되게."""
        c = self.c
        c.q[c.iq] = x
        pin.computeJointJacobians(c.model, c.data, c.q)
        js, je = c.model.getJointId('joint_2'), c.model.getJointId('joint_3')
        dz = c.data.oMi[je].translation[2] - c.data.oMi[js].translation[2]
        Je = pin.getJointJacobian(c.model, c.data, je, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[2, c.iv]
        Js = pin.getJointJacobian(c.model, c.data, js, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[2, c.iv]
        return dz * (Je - Js)

    def start(self, T, g):
        best = None
        for s in self.ik.seeds(g, self.rule.wrist(T)):
            q, ok = self.ik.solve(g, T, s, iters=60)
            if not ok:
                continue
            d = self.clear(np.r_[g, q])
            if d < NODE_CLEAR:
                continue
            if ELBOW and elbow_up(self.c, np.r_[g, q]) != (ELBOW == 'up'):
                continue                                   # 팔꿈치 방향 지정 (up: 팔꿈치가 손목보다 위, 현장 사진 자세)
            sc = -min(d, .1) - .05 * min(abs(math.sin(q[4])), .5) + .01 * np.abs(q).sum()
            if best is None or sc < best[0]:
                best = (sc, np.r_[g, q])
        return None if best is None else best[1]

    def solve(self, T, x_prev, g_hint, iters=80):
        if self.method == 'decoupled':
            q, ok = self.ik.solve(g_hint, T, x_prev[2:], iters=iters)
            if not ok and LOCAL_SEARCH:
                for dx in (-.08, 0, .08):
                    for dy in (-.08, 0, .08):
                        if dx == 0 and dy == 0:
                            continue
                        g2 = np.clip(np.asarray(g_hint) + [dx, dy], self.lo[:2], self.hi[:2])
                        q, ok = self.ik.solve(g2, T, x_prev[2:], iters=iters)
                        if ok:
                            return np.r_[g2, q], ok
            return np.r_[g_hint, q], ok
        x = x_prev.copy()
        pt, Rt = T[:3, 3], T[:3, :3]
        ok = False
        for it in range(iters):
            J, p, R = self.c.jacobian_tcp(x)
            self.stats['ik_iters'] += 1
            e = np.r_[pt - p, R @ pin.log3(R.T @ Rt)]
            lam2 = 1e-6 + 1e-3 * max(0.0, .05 - np.linalg.svd(J, compute_uv=False)[-1])
            Jp = J.T @ np.linalg.inv(J @ J.T + lam2 * np.eye(6))
            grad = np.r_[W_GANTRY * (x[:2] - g_hint), np.zeros(6)] + .5 * (x - x_prev)
            if self.j5_sign and self.j5_sign * x[6] < J5_MIN:
                # 손목 분기 유지: j5 를 시작 부호 쪽 J5_MIN 으로 당긴다 (영공간 = 겐트리 여유자유도로 해결)
                grad[6] += 4.0 * (x[6] - self.j5_sign * J5_MIN)
            if self.j3_sign and self.j3_sign * x[4] < J3_MIN:
                grad[4] += 4.0 * (x[4] - self.j3_sign * J3_MIN)
            if ELBOW == 'up':
                grad += W_ELBOW * self.elbow_grad(x)          # 영공간: 팔꿈치를 손목보다 위로 (영상/사진 자세)
            if W_LARM:
                grad += W_LARM * self.larm_grad(x)            # 영공간: 위팔을 수평으로 (영상의 ㄱ자 자세)
            dx = Jp @ e - (np.eye(8) - Jp @ J) @ (.3 * grad)
            n = np.linalg.norm(dx)
            if n > .3:
                dx *= .3 / n
            x = np.clip(x + dx, self.lo, self.hi)
            if np.linalg.norm(e[:3]) < pb.IK_POS and np.linalg.norm(e[3:]) < pb.IK_ANG and np.linalg.norm(dx) < 1e-4:
                ok = True
                break
        if not ok:
            J, p, R = self.c.jacobian_tcp(x)
            e = np.r_[pt - p, R @ pin.log3(R.T @ Rt)]
            ok = np.linalg.norm(e[:3]) < pb.IK_POS and np.linalg.norm(e[3:]) < pb.IK_ANG
        if ok:
            x = self.avoid(x, T)
            x, ok = self.polish(x, T)
        return x, ok

    def polish(self, x, T, pos_only=False, iters=30):
        """회피(영공간 밀기) 뒤 작업 오차를 다시 0 으로: 작업 공간 투영만 반복. 못 맞추면 실패."""
        pt, Rt = T[:3, 3], T[:3, :3]
        for _ in range(iters):
            J, p, R = self.c.jacobian_tcp(x)
            if pos_only:
                e = pt - p
                if np.linalg.norm(e) < pb.IK_POS:
                    return x, True
                J = J[:3]
            else:
                e = np.r_[pt - p, R @ pin.log3(R.T @ Rt)]
                if np.linalg.norm(e[:3]) < pb.IK_POS and np.linalg.norm(e[3:]) < pb.IK_ANG:
                    return x, True
            Jp = J.T @ np.linalg.inv(J @ J.T + 1e-8 * np.eye(len(e)))
            x = np.clip(x + Jp @ e, self.lo, self.hi)
        return x, False

    def bumper_kd(self):
        if getattr(self, '_bkd', None) is None:
            import trimesh
            from scipy.spatial import cKDTree
            from cell_model import read_ply, resolve
            mesh = self.c.root.find(f"link[@name='{self.c.name}_link']/visual/geometry/mesh")
            V, F = read_ply(resolve(mesh.get('filename')))
            m = trimesh.Trimesh(V, F, process=False)
            P = np.vstack([V, m.sample(200000)])
            P = (self.c.T_wp[:3, :3] @ P.T).T + self.c.T_wp[:3, 3]
            self._bkd = cKDTree(P)
        return self._bkd

    def solve_pos(self, T, x_prev, g_hint, iters=80):
        """이동 구간: TCP 위치만 정확히 (3D 작업), 공구 방향은 영공간에서 보간 방향으로 약하게 당김."""
        x = x_prev.copy()
        pt, Rt = T[:3, 3], T[:3, :3]
        ok = False
        for it in range(iters):
            J, p, R = self.c.jacobian_tcp(x)
            self.stats['ik_iters'] += 1
            ep = pt - p
            eo = R @ pin.log3(R.T @ Rt)
            Jt = J[:3]
            Jp = Jt.T @ np.linalg.inv(Jt @ Jt.T + 1e-6 * np.eye(3))
            grad = np.r_[W_GANTRY * (x[:2] - g_hint), np.zeros(6)] + .5 * (x - x_prev) - W_ORI_TRANSIT * (J[3:].T @ eo)
            if self.j5_sign and self.j5_sign * x[6] < J5_MIN:
                grad[6] += 4.0 * (x[6] - self.j5_sign * J5_MIN)
            if self.j3_sign and self.j3_sign * x[4] < J3_MIN:
                grad[4] += 4.0 * (x[4] - self.j3_sign * J3_MIN)
            dx = Jp @ ep - (np.eye(8) - Jp @ Jt) @ (.3 * grad)
            n = np.linalg.norm(dx)
            if n > .3:
                dx *= .3 / n
            x = np.clip(x + dx, self.lo, self.hi)
            if np.linalg.norm(ep) < pb.IK_POS and np.linalg.norm(dx) < 2e-4:
                ok = True
                break
        if not ok:
            _, p, _ = self.c.jacobian_tcp(x)
            ok = np.linalg.norm(pt - p) < pb.IK_POS
        if ok:
            x = self.avoid(x, T, pos_only=True)
            x, ok = self.polish(x, T, pos_only=True)
        return x, ok

    def avoid(self, x, T, iters=25, pos_only=False):
        """영공간 충돌 회피: 가장 가까운 쌍 거리를 키우는 방향으로, 작업(TCP)은 유지."""
        pt, Rt = T[:3, 3], T[:3, :3]
        for _ in range(iters):
            self.clear(x)
            d = self._cache[2]
            near = [k for k in np.argsort(d) if d[k] < AVOID_ON and self.avoid_mask[k]]
            if not near:
                return x
            self.stats['avoid_iters'] += 1
            d0, g = None, None
            for k in near[:3]:                       # 기울기가 있는 가장 가까운 쌍
                d0, g = self.pair_grad(x, k)
                if np.linalg.norm(g) > 1e-4:
                    break
            if g is None or np.linalg.norm(g) <= 1e-4:
                return x
            J, p, R = self.c.jacobian_tcp(x)
            if pos_only:
                J = J[:3]
                e = pt - p
                Jp = J.T @ np.linalg.inv(J @ J.T + 1e-6 * np.eye(3))
            else:
                e = np.r_[pt - p, R @ pin.log3(R.T @ Rt)]
                Jp = J.T @ np.linalg.inv(J @ J.T + 1e-6 * np.eye(6))
            N = np.eye(8) - Jp @ J
            gn = N @ g
            if np.linalg.norm(gn) < 1e-6:
                return x
            step = min(.02, AVOID_ON - d0 + .005) / max(np.linalg.norm(gn), 1e-6)
            x = np.clip(x + Jp @ e + step * gn, self.lo, self.hi)
        return x


def hop_joint(S, X, C, A, B, gB, lift):
    """최적 이동 (TRANSIT_MODE 'free'): 줄 끝 A 에서 lift 만큼 후퇴 -> 다음 줄 접근 자세까지 관절공간 직선 보간 -> 접근.
    관절공간 직선이라 자세가 고르게 조금씩 바뀌고 이동이 가장 짧다. 성공하면 X/C 에 붙이고 (True, 노드수)."""
    def fk_pose(x):
        p, R = S.c.tcp(x)
        return pose(p, R)
    lift_in = lift
    lift = max(lift, 0.0)
    if lift > 0:
        A1 = pose(A[:3, 3] - lift * A[:3, 2], A[:3, :3])
        hp = line(A, A1) + [A1]
        ok, why = advance(S, X, C, hp, [(T, 'transit', 0, -2, -1) for T in hp], [X[-1][:2]] * len(hp))
        if not ok:
            return False, f'retract {why}'
    B1 = pose(B[:3, 3] - lift * B[:3, 2], B[:3, :3])
    global ELBOW
    elbow_saved = ELBOW
    if lift_in < 0:                                       # 자세 바꾸기 변형: 팔꿈치 방향 조건 없이 다음 줄 시작 자세를 찾음
        ELBOW = ''
        xb, ok = None, False
        xs = S.start(B1, gB)
        if xs is not None:
            xb, ok = xs, True
    else:
        xb, ok = S.solve(B1, X[-1], gB)
    ELBOW = elbow_saved
    if not ok or S.clear(xb) < NODE_CLEAR:
        xs = S.start(B1, gB)
        if xs is None and ALLOW_FLIP:                     # 시험용: 겐트리 위치를 넓게 바꿔 아무 분기나
            ELBOW = ''
            for phi in range(0, 360, 30):
                for r in (.6, .7, .8, .9, 1.0):
                    S.phi, S.rdes = math.radians(phi), r
                    xs = S.start(B1, S.g_ref(B1))
                    if xs is not None:
                        break
                if xs is not None:
                    break
            ELBOW = elbow_saved
        if xs is None:
            return False, 'approach ik'
        xb = xs
    if lift_in < 0:                                      # 'link' 변형: 인증되는 연결 경로(정해진 후보 -> RRT)
        mids = home_link(S, X[-1], xb)
        if mids is None:
            return False, 'link failed'
        ch = [X[-1]] + mids + [xb]
        nodes = []
        for u, v in zip(ch, ch[1:]):
            m = int(max(1, math.ceil(np.degrees(np.abs(v[2:] - u[2:]).max()) / JOINT_HOP_DEG),
                        math.ceil(np.linalg.norm(v[:2] - u[:2]) / STEP)))
            nodes += [u + (v - u) * k / m for k in range(1, m + 1)]
        S.j5_sign = float(np.sign(math.sin(xb[6])) or 1.0)
        S.j3_sign = float(np.sign(xb[4]) or 1.0)
        S.flips = getattr(S, 'flips', 0) + 1
        for xk in nodes:
            X.append(xk)
            C.append((fk_pose(xk), 'transit', 0, -3, -1))
        return True, len(nodes)
    if signature(S.c, xb) != signature(S.c, X[-1]):
        if not ALLOW_FLIP:
            return False, 'branch change'
        S.j5_sign = float(np.sign(math.sin(xb[6])) or 1.0)   # 시험용: 이동 중 천천히 자세 뒤집기 허용 -> 새 분기 유지
        S.j3_sign = float(np.sign(xb[4]) or 1.0)
        S.flips = getattr(S, 'flips', 0) + 1
    xa = X[-1]
    pa, pb_ = S.c.tcp(xa)[0], S.c.tcp(xb)[0]
    n = int(max(1, math.ceil(np.degrees(np.abs(xb[2:] - xa[2:]).max()) / JOINT_HOP_DEG),
                math.ceil(np.linalg.norm(xb[:2] - xa[:2]) / STEP), math.ceil(np.linalg.norm(pb_ - pa) / STEP)))
    nodes = [xa + (xb - xa) * k / n for k in range(1, n + 1)]
    for xk in nodes:
        if S.clear(xk) < NODE_CLEAR:
            return False, 'joint hop collision'
    for xk in nodes:
        X.append(xk)
        C.append((fk_pose(xk), 'transit', 0, -3, -1))
    if lift > 0:
        hp = line(B1, B)
        ok, why = advance(S, X, C, hp, [(T, 'transit', 0, -2, -1) for T in hp], [gB] * len(hp))
        if not ok:
            return False, f'approach {why}'
    return True, n


def advance(S, X, C, Ts, entries, G):
    """Ts 를 차례로 풀어 X/C 에 붙인다. 실패하면 (False, 이유).
    관절 변화가 MAX_STEP_DEG 를 넘으면 그 구간을 카테시안으로 잘게 나눠 이어 푼다
    (경로가 급하게 도는 곳 -> 같은 분기로 연속 추종). 나눈 노드는 'mid'/'transit' 로 들어간다."""
    def fk_pose(x):
        p, R = S.c.tcp(x)
        return pose(p, R)

    def sol(T, xp, g, ent):
        if ent[1] == 'transit' and ent[3] == -1 and TRANSIT_MODE in ('line', 'free') and S.method == 'wholebody':
            return S.solve_pos(T, xp, g)
        return S.solve(T, xp, g)
    for T, ent, g in zip(Ts, entries, G):
        T_prev = fk_pose(X[-1])
        xn, ok = sol(T, X[-1], g, ent)
        jump = np.degrees(np.abs(xn[2:] - X[-1][2:]).max()) if ok else 999
        if not ok or jump > MAX_STEP_DEG:
            m = int(min(16, max(2, math.ceil((jump if ok else 24) / (MAX_STEP_DEG / 2)))))
            sub_ok, xs = True, []
            xp, gp = X[-1], X[-1][:2]
            for k in range(1, m + 1):
                u = k / m
                Tk = T if k == m else interp_pose(T_prev, T, u)
                xk, okk = sol(Tk, xp, gp + (np.asarray(g) - gp) * u, ent)
                if not okk or np.degrees(np.abs(xk[2:] - xp[2:]).max()) > MAX_STEP_DEG or S.clear(xk) < NODE_CLEAR:
                    sub_ok = False
                    break
                xs.append((xk, Tk))
                xp = xk
            if not sub_ok:
                if not ok:
                    return False, 'ik'
                return False, f'joint jump {jump:.0f}deg (subdiv failed)'
            kind = 'mid' if ent[1] == 'target' and C[-1][1] in ('target', 'mid') else 'transit'
            for xk, Tk in xs[:-1]:
                X.append(xk)
                C.append((Tk, kind, int(ent[2] and C[-1][2]), ent[3] if kind == 'mid' else (ent[3] if ent[1] == 'transit' else -1), -1))
            xn = xs[-1][0]
        if S.clear(xn) < NODE_CLEAR:
            d = S.c.distances(xn)
            j = int(np.argmin(d))
            return False, f'collision {S.c.pair_names(j)} {d[j] * 1000:.0f}mm'
        X.append(xn)
        C.append(ent)
    return True, 'ok'


HINT_DPHI = [0, 22.5, -22.5, 45, -45, 67.5, -67.5, 90, -90]
HINT_R = [0, -.05, .05, -.10, .10, -.15]


def hint_candidates(phi, r):
    """직전 줄의 (phi, R) 에서 가까운 순서. 줄마다 겐트리 힌트만 바꾸고 팔은 이어서 푼다."""
    c = [(phi + dp, min(max(r + dr, .6), 1.0)) for dp in HINT_DPHI for dr in HINT_R]
    return sorted(dict.fromkeys(c), key=lambda t: abs(((t[0] - phi + 180) % 360) - 180) / 22.5 + abs(t[1] - r) / .05)


def solve_chain(S, rows, zsafe, max_hints=20, prior=None):
    log = []
    phi0, r0 = math.degrees(S.phi), S.rdes
    G0 = S.g_row(rows[0]['T'])
    x0 = S.start(rows[0]['T'][0], G0[0])
    if x0 is None:                                          # 줄 평균 안내가 첫 점에 안 맞으면: 첫 점 기준 안내 -> 주변 힌트
        T0 = rows[0]['T'][0]
        phi_s, r_s = S.phi, S.rdes
        for phi, rd in [(math.degrees(phi_s), r_s)] + hint_candidates(math.degrees(phi_s), r_s)[:max_hints]:
            S.phi, S.rdes = math.radians(phi), rd
            x0 = S.start(T0, S.g_ref(T0))
            if x0 is not None:
                break
        S.phi, S.rdes = phi_s, r_s
    if x0 is None:
        return None, None, [{'fail': 'no start'}]
    X = [x0]
    C = [(rows[0]['T'][0], 'target', int(rows[0]['paint'][0]), rows[0]['row'], int(rows[0]['csv_idx'][0]))]
    cur = (phi0, r0)
    if S.method == 'wholebody' and S.keep_wrist:
        S.j5_sign = float(np.sign(math.sin(x0[6])) or 1.0)
        S.j3_sign = float(np.sign(x0[4]) or 1.0)
    for r, row in enumerate(rows):
        ents = [(T, 'target', int(row['paint'][i]), row['row'], int(row['csv_idx'][i])) for i, T in enumerate(row['T'])]
        A = rows[r - 1]['T'][-1] if r else None
        done, last = False, None
        cands = hint_candidates(*cur)[:max_hints]
        pri = (prior or {}).get(row['row'])
        if pri is not None:                                # 레퍼런스(이전 계획) 힌트를 먼저
            cands = [(pri['phi'], pri['rdes'])] + [c for c in cands if c != (pri['phi'], pri['rdes'])]
        for phi, rd in cands:
            S.phi, S.rdes = math.radians(phi), rd
            Gr = S.g_row(row['T'])
            if r == 0:
                X2, C2 = list(X), list(C)
                ok, why = advance(S, X2, C2, row['T'][1:], ents[1:], Gr[1:])
                if ok:
                    X, C, done = X2, C2, True
                    log.append({'row': row['row'], 'phi': phi, 'rdes': rd})
                    break
                last = {'row': row['row'], 'phi': phi, 'rdes': rd, 'why': why}
                continue
            # 파란 직선을 정확히 따라감. 직선이 범퍼를 관통해 불가능할 때만 바깥으로 돌아가는 경로(후퇴/넘어감).
            if TRANSIT_MODE == 'line':
                # (범퍼 근처만 3~8 cm 비킨 직선도 시험했지만 관통 3곳에서 모두 실패하고 시간만 3배 -> 쓰지 않음)
                variants = [('line', 0.0)] + [('around', v) for v in range(1, HOP_VARIANTS)]
                if ALLOW_FLIP:
                    variants.append(('joint', -1.0))         # 마지막 수단: 줄 사이 이동 중 팔 자세 바꾸기 (인증 연결 경로)
            elif TRANSIT_MODE == 'free':
                # 관절공간 이동 먼저. 다음 줄 시작이 다른 분기로만 풀리면 직선을 따라 이어 푼 것(같은 분기)으로
                variants = [('joint', 0.0), ('line', 0.0), ('joint', LIFT), ('joint', 2 * LIFT)] + [('around', v) for v in range(1, HOP_VARIANTS)]
                if ALLOW_FLIP:
                    variants.append(('joint', -1.0))         # 시험용 마지막 수단: 자세 뒤집기 + 인증 연결 경로
            else:
                variants = [('around', v) for v in range(HOP_VARIANTS)]
            if pri is not None and pri.get('hop') is not None:
                hv = tuple(pri['hop'])
                variants = [hv] + [v for v in variants if tuple(v) != hv]
            row_fail = 0                                   # 이동은 됐는데 줄 안에서 실패한 횟수 -> 2번이면 힌트 문제로 보고 다음 힌트
            for v in variants:
                if row_fail >= ROW_FAIL_NEXT_HINT:
                    break
                if v[0] == 'joint':
                    X2, C2 = list(X), list(C)
                    signs = (S.j5_sign, S.j3_sign, getattr(S, 'flips', 0), S.phi, S.rdes)
                    ok, why = hop_joint(S, X2, C2, A, row['T'][0], Gr[0], v[1])
                    S.phi, S.rdes = signs[3], signs[4]
                    if ok:
                        n = why
                        ok, why = advance(S, X2, C2, list(row['T']), ents, list(Gr))
                        row_fail += not ok
                    if ok:
                        X, C, done = X2, C2, True
                        log.append({'into_row': row['row'], 'hop_variant': v, 'hop_nodes': n, 'phi': phi, 'rdes': rd,
                                    'line_deviation_mm': None,
                                    'hop_length_mm': round(float(np.linalg.norm(row['T'][0][:3, 3] - A[:3, 3])) * 1000)})
                        break
                    S.j5_sign, S.j3_sign, S.flips = signs[:3]
                    last = {'into_row': row['row'], 'phi': phi, 'rdes': rd, 'variant': v, 'why': why}
                    continue
                if v[0] == 'line':
                    hp, dev = hop_line(S, A, row['T'][0], v[1])
                else:
                    hp, dev = hop_poses(A, row['T'][0], zsafe, v[1]), None
                n = len(hp)
                gA, gB = X[-1][:2], Gr[0]
                Gh = [gA + (gB - gA) * (k + 1) / (n + 1) for k in range(n)]
                X2, C2 = list(X), list(C)
                tag = -1 if v[0] == 'line' else -2
                ok, why = advance(S, X2, C2, hp + list(row['T']),
                                  [(T, 'transit', 0, tag, -1) for T in hp] + ents, Gh + list(Gr))
                if ok:
                    X, C, done = X2, C2, True
                    log.append({'into_row': row['row'], 'hop_variant': v, 'hop_nodes': n, 'phi': phi, 'rdes': rd,
                                'line_deviation_mm': None if dev is None else round(dev * 1000, 1),
                                'hop_length_mm': round(float(np.linalg.norm(row['T'][0][:3, 3] - A[:3, 3])) * 1000)})
                    break
                row_fail += len(X2) - len(X) > n
                last = {'into_row': row['row'], 'phi': phi, 'rdes': rd, 'variant': v,
                        'reached_in_segment': len(X2) - len(X), 'why': why}
            if done:
                break
        if not done:
            return None, None, log + [last]
        cur = (phi, rd)
    return np.array(X), C, log


GANTRY_SMOOTH_WIN = (61, 41, 25, 15)   # 줄 안 겐트리 곡선 이동평균 창 (큰 것부터 시도)


def smooth_gantry(S, X, chain):
    """줄마다 겐트리 궤적을 매끄럽게 (왕복/떨림 제거): 이동평균 + 양 끝 고정 보정 -> 겐트리를 그 곡선에 고정하고
    팔 6축만 이어서 다시 푼다 (분리 방식). 충돌/관절 급변/분기 변화가 있으면 더 작은 창, 다 안 되면 원래 줄 유지."""
    X = np.array(X)
    rows = sorted({c[3] for c in chain if c[1] in ('target', 'mid') and c[3] >= 0})
    info = {}
    for r in rows:
        idx = [k for k, c in enumerate(chain) if c[1] in ('target', 'mid') and c[3] == r]
        if len(idx) < 5 or idx != list(range(idx[0], idx[-1] + 1)):
            continue
        G = X[idx, :2]
        sig0 = signature(S.c, X[idx[0]])
        done = None
        for w in GANTRY_SMOOTH_WIN:
            k = w // 2
            Gs = G.copy()
            for _ in range(2):
                Gs = np.array([Gs[max(0, i - k):i + k + 1].mean(0) for i in range(len(Gs))])
            u = np.linspace(0, 1, len(G))[:, None]
            Gs = Gs + (1 - u) * (G[0] - Gs[0]) + u * (G[-1] - Gs[-1])       # 양 끝은 원래 값 (줄 사이 이동과 이어짐)
            Gs = np.clip(Gs, S.lo[:2], S.hi[:2])
            xs, xp, ok = [], X[idx[0] - 1] if idx[0] > 0 else X[idx[0]], True
            for j, kk in enumerate(idx):
                q, okq = S.ik.solve(Gs[j], chain[kk][0], (X[kk] if j == 0 else xp)[2:], iters=60)
                xn = np.r_[Gs[j], q]
                if (not okq or S.clear(xn) < NODE_CLEAR
                        or np.degrees(np.abs(xn[2:] - xp[2:]).max()) > MAX_STEP_DEG or signature(S.c, xn) != sig0):
                    ok = False
                    break
                xs.append(xn)
                xp = xn
            if ok:
                done = (w, xs)
                break
        if done is not None:
            X[idx] = np.array(done[1])
        info[int(r)] = None if done is None else done[0]
    return X, info


def refine(S, X, chain):
    """모든 간선 u=0.1..0.9 카테시안 검사, 넘으면 중점 삽입 (재귀).
    도장 줄 간선: 위치 1 mm + 방향 0.2 deg.  이동(파란 직선) 간선: 위치만 1 mm (방향은 자유)."""
    outX, outC, worst, ins = [X[0]], [chain[0]], [0.0, 0.0], 0
    free = TRANSIT_MODE in ('line', 'free') and S.method == 'wholebody'
    S.hop_free = True

    def rec(xa, xb, Ta, Tb, depth, transit):
        nonlocal ins
        w = [0.0, 0.0]
        for u in U_CHECK:
            p, R = S.c.tcp((1 - u) * xa + u * xb)
            e = pose_err(p, R, interp_pose(Ta, Tb, u))
            w = [max(w[0], e[0]), max(w[1], 0.0 if transit else e[1])]
        good = w[0] <= POS_TOL and w[1] <= ANG_TOL
        if good or depth >= 8:
            worst[0], worst[1] = max(worst[0], w[0]), max(worst[1], w[1])
            return [] if good else None
        Tm = interp_pose(Ta, Tb, .5)
        gm = (xa[:2] + xb[:2]) / 2
        solve = S.solve_pos if transit else S.solve
        # 왼쪽 끝에서 이어 푼다 (연속 경로 추종). 실패하면 평균 시드로 한 번 더.
        xm, ok = solve(Tm, xa, gm)
        if not ok or S.clear(xm) < NODE_CLEAR:
            xm, ok = solve(Tm, (xa + xb) / 2, gm)
        if not ok or S.clear(xm) < NODE_CLEAR:
            return None
        L = rec(xa, xm, Ta, Tm, depth + 1, transit)
        Rr = rec(xm, xb, Tm, Tb, depth + 1, transit)
        if L is None or Rr is None:
            return None
        ins += 1
        return L + [(xm, Tm)] + Rr

    for k in range(len(X) - 1):
        ka, kb = chain[k][1], chain[k + 1][1]
        transit = free and 'transit' in (ka, kb) and -2 not in (chain[k][3], chain[k + 1][3])
        if transit and kb == 'target' and ELBOW:             # 줄로 들어가는 이동 구간(-> 도장점)은 방향까지 보간 (위치만 풀면 도장점 자세와 안 이어져 수렴 실패)
            transit = False
        Ta, Tb = chain[k][0], chain[k + 1][0]
        if S.method == 'wholebody' and S.keep_wrist:          # 구간마다 그 구간의 자세 분기 유지 (중간에 자세를 바꾼 경우 대비)
            S.j3_sign = float(np.sign(X[k][4]) or 1.0)
            S.j5_sign = float(np.sign(math.sin(X[k][6])) or 1.0)
        if not transit and ELBOW:                           # 위치만 푼 이동 노드는 실제 자세(FK)를 기준으로 (저장된 목표 방향은 안 맞춰져 있음)
            if ka == 'transit':
                Ta = pose(*S.c.tcp(X[k]))
            if kb == 'transit':
                Tb = pose(*S.c.tcp(X[k + 1]))
        if -3 in (chain[k][3], chain[k + 1][3]):          # 관절공간 이동: 카테시안 경로 요구 없음 (충돌은 인증에서)
            mids = []
        else:
            mids = rec(X[k], X[k + 1], Ta, Tb, 0, transit)
        if mids is None:
            return None, None, {'fail_edge': k, 'kind': ka, 'row': chain[k][3]}
        kind = 'mid' if ka in ('target', 'mid') and kb in ('target', 'mid') else 'transit'
        paint = int(chain[k][2] and chain[k + 1][2])
        row = chain[k][3] if kind == 'mid' else (-2 if -2 in (chain[k][3], chain[k + 1][3]) else
                                                 -3 if -3 in (chain[k][3], chain[k + 1][3]) else -1)
        for xm, Tm in mids:
            outX.append(xm)
            outC.append((Tm, kind, paint, row, -1))
        outX.append(X[k + 1])
        outC.append(chain[k + 1])
    return np.array(outX), outC, {'inserted': ins, 'interp_max_mm': worst[0] * 1000,
                                  'interp_max_deg': math.degrees(worst[1])}


def stow_posture(S, g):
    """홈(겐트리 g)에서 쓸 팔 접은 자세: 충돌 없고 여유가 가장 큰 것."""
    best = None
    for j1 in np.radians([0, 45, 90, 135, 180, -45, -90, -135]):
        for j2, j3 in ((-.6, 2.2), (.6, -2.2), (-.3, 1.8), (.3, -1.8), (-1.0, 2.4), (1.0, -2.4)):
            for j5 in (1.2, -1.2, .6, -.6):
                x = np.r_[g, j1, j2, j3, 0.0, j5, 0.0]
                d = S.clear(x)
                if d >= NODE_CLEAR and (best is None or d > best[0]):
                    best = (d, x)
    return None if best is None else best[1]


def home_link(S, a, b):
    """홈 <-> 접근 자세 연결 (도장 경로가 아니므로 자세 변화 허용). 인증되는 첫 경로."""
    cands = [[a, b], [a, np.r_[b[:2], a[2:]], b], [a, np.r_[a[:2], b[2:]], b]]
    mid_g = np.r_[a[0], b[1]]
    cands += [[a, np.r_[mid_g, a[2:]], np.r_[b[:2], a[2:]], b], [a, np.r_[b[0], a[1], a[2:]], np.r_[b[:2], a[2:]], b]]
    # 통로(y = a.y 또는 b.y)를 따라 겐트리를 옮기다가 중간 지점에서 팔 자세를 바꾸는 경로
    # (홈 모서리나 범퍼 옆에서는 자세 변화가 막히는 경우)
    for yl in dict.fromkeys((float(a[1]), float(b[1]))):
        for u in (.5, .25, .75, 0.0, 1.0):
            xs = a[0] + (b[0] - a[0]) * u
            ch = [a, np.r_[a[0], yl, a[2:]], np.r_[xs, yl, a[2:]], np.r_[xs, yl, b[2:]], np.r_[b[0], yl, b[2:]], b]
            cands.append([x for k, x in enumerate(ch) if k == 0 or np.abs(x - ch[k - 1]).max() > 1e-9])
    for ch in cands:
        if any(S.clear(x) < NODE_CLEAR for x in ch[1:-1]):
            continue
        if all(S.c.certify_edge(u, v, CERT_CLEAR)[0] for u, v in zip(ch, ch[1:])):
            return ch[1:-1]
    # 정해진 경로가 모두 막히면 인증되는 RRT (도장 경로 밖이라 허용). 시드 고정 -> 재현 가능
    if getattr(S, '_fin', None) is None:
        S._fin = pb.Finisher(pb.Planner(S.c, log=lambda *a: None))
    for seed in range(4):
        ch = S._fin.rrt_connect(a, b, seed=seed)
        if ch is not None and S._fin._chain_ok(ch) is not None:
            return ch[1:-1]
    return None


def add_home(S, X, chain, home_g=None, home_arm='ready'):
    """홈 1곳: 겐트리 = 행정 상한 모서리(기본, run_*.sh 슬라이더 모드 시작 위치와 같음) 또는 지정 좌표.
    팔 = 'ready': 첫 도장점 준비 자세(첫 점에서 후퇴한 접근 자세의 팔 관절 그대로 -> 출발 시 겐트리만 움직임),
         'stow' : 접은 자세. ready 가 홈 위치에서 충돌하면 stow 로 대신한다.
    시작: 홈 -> 첫 점에서 분사축 반대로 후퇴한 접근 자세 -> 첫 점.  끝: 마지막 점 -> 후퇴 -> 같은 홈."""
    if home_g in (None, 'upper'):
        g = S.hi[:2].copy()
    elif home_g == 'lower':
        g = S.lo[:2].copy()
    else:
        g = np.array([float(v) for v in str(home_g).split(',')]) if isinstance(home_g, str) else np.asarray(home_g, float)
    stow = stow_posture(S, g)

    def retract(x, T, dist):
        """첫/마지막 점에서 분사축 반대로 dist 만큼 카테시안 직선 후퇴 (같은 IK 로 이어서, 자세 유지)."""
        if S.method == 'wholebody' and S.keep_wrist:
            S.j3_sign = float(np.sign(x[4]) or 1.0)
            S.j5_sign = float(np.sign(math.sin(x[6])) or 1.0)
        P0 = pose(T[:3, 3], T[:3, :3])
        P1 = pose(T[:3, 3] - dist * T[:3, 2], T[:3, :3])
        xs, xp = [], x
        for Tk in line(P0, P1) + [P1]:
            xp, ok = S.solve(Tk, xp, xp[:2])
            if not ok or S.clear(xp) < NODE_CLEAR:
                return None
            xs.append(xp)
        return xs
    # 접근/후퇴를 멀리 -> 홈 이동이 범퍼에서 떨어진다. 행정 끝 근처라 멀리 못 빠지면 짧게 (시작/끝 따로)
    DISTS = (.25, .18, LIFT, .08, .05)
    apps = [r for r in (retract(X[0], chain[0][0], d) for d in DISTS) if r is not None]
    rets = [r for r in (retract(X[-1], chain[-1][0], d) for d in DISTS) if r is not None]
    if not apps or not rets:
        return None, None
    homes = []
    if isinstance(home_arm, str) and ',' in home_arm:     # 지정 자세 'j1,...,j6' [deg]
        homes.append(np.r_[g, np.radians([float(v) for v in home_arm.split(',')])])
        stow = None                                       # 지정 자세가 안 되면 실패로 알림 (임의로 바꾸지 않음)
    elif home_arm == 'ready':
        homes += [np.r_[g, app[-1][2:]] for app in apps]
    homes.append(stow)
    found = None
    for home in homes:
        if home is None or S.clear(home) < NODE_CLEAR:
            continue
        v_in = next(((app, v) for app in apps for v in [home_link(S, home, app[-1])] if v is not None), None)
        if v_in is None:
            continue
        v_out = next(((ret, v) for ret in rets for v in [home_link(S, ret[-1], home)] if v is not None), None)
        if v_out is not None:
            found = home, v_in, v_out
            break
    if found is None:
        return None, None
    home, (app, v_in), (ret, v_out) = found
    H = (None, 'home', 0, -1, -1)
    Tn = (None, 'transit', 0, -1, -1)
    nX = [home] + v_in + app[::-1] + list(X) + ret + v_out + [home]
    nC = [H] + [Tn] * len(v_in) + [Tn] * len(app) + list(chain) + [Tn] * len(ret) + [Tn] * len(v_out) + [H]
    return np.array(nX), nC


REV_MIN = .03                    # 겐트리 왕복으로 셀 최소 되돌아감 [m] (눈에 보이는 왕복)


def back_travel(v, th=REV_MIN):
    """1차원 궤적 v: 진행 방향에서 th 이상 되돌아간 횟수와, 한 번에 되돌아간 최대 거리."""
    n, worst, d, ext = 0, 0.0, 0, float(v[0])
    for x in map(float, v[1:]):
        if d == 0:                                         # 아직 방향 없음: th 이상 움직이면 방향 결정
            if abs(x - ext) > th:
                d, ext = (1 if x > ext else -1), x
            continue
        if (x - ext) * d >= 0:                             # 진행 방향으로 더 감
            ext = x
            continue
        b = abs(x - ext)                                   # 되돌아감
        worst = max(worst, b)
        if b > th:
            n += 1
            d, ext = -d, x
    return n, worst


def motion_stats(cell, X, chain):
    """도장 줄 안 겐트리 왕복(REV_MIN 이상 되돌아감) 횟수 / 최대 되돌아감, 겐트리 이동 거리, 팔 뻗음(어깨->손목, 최대 1.076 m)."""
    rev, back, trav, reach, per_row = 0, 0.0, 0.0, [], {}
    rows = sorted({c[3] for c in chain if c[1] in ('target', 'mid') and c[3] >= 0})
    for r in rows:
        idx = [k for k, c in enumerate(chain) if c[1] in ('target', 'mid') and c[3] == r]
        G = X[idx, :2]
        trav += float(np.linalg.norm(np.diff(G, axis=0), axis=1).sum())
        rr = [back_travel(G[:, a]) for a in (0, 1)]
        rev += sum(n for n, _ in rr)
        back = max(back, max(w for _, w in rr))
        per_row[int(r)] = (sum(n for n, _ in rr), round(max(w for _, w in rr), 3))
        for k in idx[::3]:
            cell.set(X[k])
            sh = cell.data.oMf[cell.model.getFrameId('link_2')].translation
            wr = cell.data.oMf[cell.model.getFrameId('link_5')].translation
            reach.append(float(np.linalg.norm(wr - sh)))
    up = []
    for r in rows:
        idx = [k for k, c in enumerate(chain) if c[1] in ('target', 'mid') and c[3] == r]
        up += [elbow_up(cell, X[k]) for k in idx[::3]]
    return {'elbow_above_wrist_percent': float(100 * np.mean(up)), 'gantry_reversals_in_rows': int(rev), 'gantry_back_travel_max_m': back, 'gantry_travel_in_rows_m': trav,
            'gantry_reversals_per_row': per_row,
            'arm_reach_max_m': max(reach), 'arm_reach_mean_m': float(np.mean(reach)),
            'arm_reach_over_1m_percent': float(100 * np.mean(np.array(reach) > 1.0))}


def certify_all(cell, X, workers, urdf, gun, pool=None):
    pairs = [(X[k], X[k + 1]) for k in range(len(X) - 1)]
    if pool is not None:
        chunks = [pairs[i:i + 24] for i in range(0, len(pairs), 24)]
        res = [r for c in pool.map(pb._w_certify, chunks) for r in c]
    elif workers > 1:
        import multiprocessing as mp
        chunks = [pairs[i:i + 24] for i in range(0, len(pairs), 24)]
        with mp.get_context('fork').Pool(workers, initializer=pb._winit, initargs=(urdf, gun)) as pool:
            res = [r for c in pool.map(pb._w_certify, chunks) for r in c]
    else:
        res = [cell.certify_edge(a, b, CERT_CLEAR) for a, b in pairs]
    bad = [k for k, r in enumerate(res) if not r[0]]
    lo = min((r[1] for r in res if r[0]), default=float('nan'))
    return bad, lo


def metrics(cell, X, chain, rows):
    kinds0 = [c[1] for c in chain]
    f0 = next(i for i, k in enumerate(kinds0) if k == 'target')
    l0 = len(kinds0) - 1 - next(i for i, k in enumerate(reversed(kinds0)) if k == 'target')
    sig = [signature(cell, x) for x in X[f0:l0 + 1]]        # 첫 도장점 ~ 마지막 도장점
    sigs = sorted(set(sig))
    steps = np.degrees(np.abs(np.diff(X[:, 2:], axis=0)))
    kinds = [c[1] for c in chain]
    in_row = np.array([kinds[k] in ('target', 'mid') and kinds[k + 1] in ('target', 'mid') for k in range(len(X) - 1)])
    tr = np.array([kinds[k] == 'transit' or kinds[k + 1] == 'transit' for k in range(len(X) - 1)])
    # 홈 <-> 접근 구간은 도장 경로 밖이라 자세 변화 허용 -> 이동 지표에서 제외
    first_t = next(i for i, k in enumerate(kinds) if k == 'target')
    last_t = len(kinds) - 1 - next(i for i, k in enumerate(reversed(kinds)) if k == 'target')
    homeseg = np.array([k < first_t or k >= last_t for k in range(len(X) - 1)])
    tr &= ~homeseg
    errs = []
    for x, c in zip(X, chain):
        if c[1] == 'target':
            p, R = cell.tcp(x)
            errs.append(pose_err(p, R, c[0]))
    errs = np.array(errs)
    R_s = []
    for x, c in zip(X, chain):
        if c[1] in ('target', 'mid'):
            cell.set(x)
            s = cell.data.oMf[cell.model.getFrameId('link_2')].translation
            w = cell.data.oMf[cell.model.getFrameId('link_5')].translation
            R_s.append(np.linalg.norm(w - s))
    clears = [float(cell.distances(x).min()) for x in X]
    # 줄 사이 이동: 실제 TCP 가 파란 직선(줄 끝 -> 다음 줄 시작)에서 벗어난 최대 거리
    line_dev = []
    tgt = [i for i, k in enumerate(kinds) if k == 'target']
    for a_i, b_i in zip(tgt, tgt[1:]):
        if b_i - a_i > 1 and any(kinds[j] == 'transit' for j in range(a_i + 1, b_i)) and chain[a_i][3] != chain[b_i][3]:
            pa, _ = cell.tcp(X[a_i])
            pb_, _ = cell.tcp(X[b_i])
            u = (pb_ - pa) / max(np.linalg.norm(pb_ - pa), 1e-9)
            dmax = 0.0
            for j in range(a_i + 1, b_i):
                v = cell.tcp(X[j])[0] - pa
                dmax = max(dmax, float(np.linalg.norm(v - np.dot(v, u) * u)))
            line_dev.append((chain[b_i][3], round(float(np.linalg.norm(pb_ - pa)) * 1000), round(dmax * 1000, 1)))
    total = sum(len(r['T']) for r in rows)
    gstep = np.linalg.norm(np.diff(X[:, :2], axis=0), axis=1)
    return {
        'coverage': f'{len(errs)}/{total}',
        'branch_signatures': [list(s) for s in sigs],
        'branch_constant': len(sigs) == 1,
        'max_joint_step_in_row_deg': float(steps[in_row].max()) if in_row.any() else 0.0,
        'max_joint_step_in_transit_deg': float(steps[tr].max()) if tr.any() else 0.0,
        'max_j4_j6_step_deg': float(steps[~homeseg][:, [3, 5]].max()),
        'max_gantry_step_in_row_mm': float(gstep[in_row].max() * 1000) if in_row.any() else 0.0,
        'max_node_pos_err_mm': float(errs[:, 0].max() * 1000),
        'max_node_ang_err_deg': float(np.degrees(errs[:, 1].max())),
        'min_node_clearance_mm': float(min(clears) * 1000),
        'R_min': float(min(R_s)), 'R_med': float(np.median(R_s)),
        'j5_min_deg': float(np.degrees(np.arcsin(min(abs(math.sin(x[6])) for x in X[f0:l0 + 1])))),
        'wrist_sign_changes': int(sum(1 for a, b in zip(X[f0:l0], X[f0 + 1:l0 + 1]) if np.sign(math.sin(a[6])) != np.sign(math.sin(b[6])))),
        'elbow_sign_changes': int(sum(1 for a, b in zip(X[f0:l0], X[f0 + 1:l0 + 1]) if np.sign(a[4]) != np.sign(b[4]))),
        'j3_min_deg': float(np.degrees(np.abs(X[f0:l0 + 1, 4]).min())),
        'home_gantry': [float(v) for v in X[0][:2]],
        'hop_line_deviation_mm': line_dev,            # (다음 줄, 직선 길이 mm, 직선에서 최대 이탈 mm)
        'max_hop_line_deviation_mm': max((d for _, _, d in line_dev), default=0.0),
        'arm_joint_travel_rad': float(np.abs(np.diff(X[:, 2:], axis=0)).sum()),
        'nodes': len(X),
    }


PAINT_JOINT_VMAX = math.radians(120.0)    # 도장 중 팔 관절 속도 상한 (급격한 관절 변화 금지)
MOVE_JOINT_VMAX = math.radians(120.0)    # 이동 중 상한
JOINT_AMAX = math.radians(600.0)         # 팔 관절 가속도 상한
GANTRY_PER_AXIS = False                  # True: 겐트리 속도 상한을 X, Y 축별로 (실제 장비처럼). False: XY 합성 속도 (예전 결과)
GANTRY_AMAX = None                       # [m/s^2] 겐트리 축별 가속도 상한. None = 보지 않음 (예전 결과)


def _accel_max(X, t):
    v = np.diff(X[:, 2:], axis=0) / np.diff(t)[:, None]
    a = np.abs(np.diff(v, axis=0)) / ((t[2:] - t[:-2]) / 2)[:, None]
    return float(a.max()) if len(a) else 0.0


def time_param_limits(cell, X, paint):
    """속도·가속도 한계를 지키는 가장 빠른 시간 배분 (겐트리 가속도 한계를 줄 때 쓴다).
    경로 변수 s = 노드 번호(구간마다 1), b = (ds/dt)^2 를 노드마다 정한다. 축 j 의
      속도   : |q'_j| sqrt(b) <= v_j            (겐트리 GANTRY_SPEED, 팔 120 deg/s, 건 끝 TCP_SPEED)
      가속도 : |q'_j a + q''_j b| <= a_j,  a = (b_{k+1} - b_k) / 2   (겐트리 GANTRY_AMAX, 팔 600 deg/s^2)
    q' = 구간 이동량, q'' = 이웃 구간 이동량의 차. b 에 대해 선형이라, 뒤에서 앞으로 "멈출 수 있는 최대 b" 를 구하고
    앞에서 뒤로 그 안에서 가장 크게 고르면 한계를 지키는 가장 빠른 속도 곡선이 된다 (처음과 끝은 정지).
    긴 구간(예: 홈 이동 5 m 한 구간)은 겐트리 2 cm / 관절 1 도 이하 조각으로 나눠 계산하고 원래 구간 시간으로 합친다."""
    X0 = np.asarray(X, float)
    nk = np.maximum(1, np.ceil(np.maximum(np.abs(np.diff(X0[:, :2], axis=0)).max(1) / .02,
                                          np.abs(np.diff(X0[:, 2:], axis=0)).max(1) / math.radians(1.0)))).astype(int)
    owner = np.repeat(np.arange(len(nk)), nk)                  # 조각 -> 원래 구간 번호
    X = np.vstack([X0[k] + np.outer(np.arange(m) / m, X0[k + 1] - X0[k]) for k, m in enumerate(nk)] + [X0[-1:]])
    paint = np.r_[np.repeat(np.asarray(paint)[:-1], nk), np.asarray(paint)[-1]]
    P = np.array([cell.tcp(x)[0] for x in X])
    D = np.diff(X, axis=0)                                     # 8축 구간 이동량 = q' (구간마다 일정)
    n = len(D)
    seg_paint = (paint[:-1] == 1) & (paint[1:] == 1)
    jv = np.where(seg_paint[:, None], np.minimum(cell.vel[2:], PAINT_JOINT_VMAX), np.minimum(cell.vel[2:], MOVE_JOINT_VMAX))
    vmax = np.c_[np.full((n, 2), pb.GANTRY_SPEED), jv]
    amax = np.r_[GANTRY_AMAX, GANTRY_AMAX, np.full(6, JOINT_AMAX)]
    lt = np.linalg.norm(np.diff(P, axis=0), axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        cap_seg = np.minimum(np.where(lt > 1e-12, (pb.TCP_SPEED / lt) ** 2, np.inf),
                             np.min(np.where(np.abs(D) > 1e-12, (vmax / np.abs(D)) ** 2, np.inf), axis=1))
    cap_seg = np.minimum(cap_seg, 1e6)
    cap = np.r_[0.0, np.minimum(cap_seg[:-1], cap_seg[1:]), 0.0]    # 노드 k 의 b 상한 (양옆 구간의 속도 한계), 처음·끝 정지
    # 노드 k (0..n-1) 의 가속도 식: c1 * b_k + c2 * b_{k+1} in [-amax, amax],  q' = D_k, q'' = D_k - D_{k-1}
    Dp = np.vstack([np.zeros(8), D[:-1]])
    c1 = (D - Dp) - D / 2
    c2 = D / 2

    def y_range(k, x):
        """노드 k 에서 b_k = x 일 때 가능한 b_{k+1} 범위 (가속도 식만)."""
        lo, hi = 0.0, np.inf
        for j in range(8):
            a2, r = c2[k, j], amax[j]
            base = c1[k, j] * x
            if abs(a2) < 1e-15:
                if abs(base) > r * 1.0001:
                    return 1.0, 0.0
                continue
            l_, h_ = (-r - base) / a2, (r - base) / a2
            if a2 < 0:
                l_, h_ = h_, l_
            lo, hi = max(lo, l_), min(hi, h_)
        return lo, hi
    K = np.zeros(n + 1)                                        # 뒤에서 앞으로: 노드 k 에서 끝까지 한계를 지키며 갈 수 있는 최대 b
    for k in range(n - 1, -1, -1):
        def feas(x):
            lo, hi = y_range(k, x)
            return max(lo, 0.0) <= min(hi, K[k + 1]) + 1e-12
        if feas(cap[k]):
            K[k] = cap[k]
        else:
            lo, hi = 0.0, cap[k]
            for _ in range(50):
                mid = (lo + hi) / 2
                if feas(mid):
                    lo = mid
                else:
                    hi = mid
            K[k] = lo
    b = np.zeros(n + 1)                                        # 앞에서 뒤로: 가능한 범위에서 가장 크게
    for k in range(n):
        lo, hi = y_range(k, b[k])
        b[k + 1] = max(0.0, min(hi, K[k + 1]))
        if b[k + 1] < lo - 1e-12:                              # 수치 오차 안전장치
            b[k + 1] = max(0.0, min(K[k + 1], lo))
    sb = np.sqrt(np.maximum(b, 0.0))
    dtf = 2.0 / np.maximum(sb[:-1] + sb[1:], 1e-9)
    return np.bincount(owner, weights=dtf, minlength=len(nk))  # 원래 구간 시간 (time_param_smooth 가 마무리)


def time_param_smooth(cell, X, paint):
    """구간 시간 = max(TCP 0.5 m/s, 겐트리 1.5 m/s, 관절 속도 상한), 그 뒤 관절 가속도 상한을 만족할 때까지
    앞/뒤로 반복해 느린 쪽으로 맞춘다. 경로(관절값)는 그대로, 시간만 늘린다.
    겐트리 가속도 한계(GANTRY_AMAX)를 주면 겐트리 축별 가속도도 같은 방식으로 지키고, 다른 출발점(time_param_limits:
    한계를 지키는 속도 곡선)에서도 한 번 더 맞춰 더 짧은 쪽을 쓴다 (둘 다 한계를 지킨다)."""
    P = np.array([cell.tcp(x)[0] for x in X])
    dq = np.abs(np.diff(X[:, 2:], axis=0))
    seg_paint = (paint[:-1] == 1) & (paint[1:] == 1)
    vmax = np.where(seg_paint[:, None], np.minimum(cell.vel[2:], PAINT_JOINT_VMAX),
                    np.minimum(cell.vel[2:], MOVE_JOINT_VMAX))
    dG = np.diff(X[:, :2], axis=0)
    g_len = np.abs(dG).max(1) if GANTRY_PER_AXIS else np.linalg.norm(dG, axis=1)
    dt0 = np.maximum.reduce([np.linalg.norm(np.diff(P, axis=0), axis=1) / pb.TCP_SPEED,
                             g_len / pb.GANTRY_SPEED,
                             (dq / vmax).max(1), np.full(len(dq), 1e-3)])
    sd = np.diff(X[:, 2:], axis=0)

    def settle(dt):
        dt = np.array(dt, float)
        for _ in range(400 if GANTRY_AMAX else 200):
            changed = False
            for order in (range(len(dt) - 1), reversed(range(len(dt) - 1))):
                for k in order:
                    va, vb = sd[k] / dt[k], sd[k + 1] / dt[k + 1]
                    h = (dt[k] + dt[k + 1]) / 2
                    f = np.abs(vb - va).max() / h / JOINT_AMAX
                    if GANTRY_AMAX:                            # 겐트리 축별 가속도도 같은 방식으로
                        ga, gb = dG[k] / dt[k], dG[k + 1] / dt[k + 1]
                        fg = np.abs(gb - ga).max() / h / GANTRY_AMAX
                        if fg > f:
                            f, va, vb = fg, ga, gb
                    if f > 1.0001:
                        f = math.sqrt(f)                        # 빠른 쪽 구간을 늘린다
                        if np.abs(va).max() >= np.abs(vb).max():
                            dt[k] *= f
                        else:
                            dt[k + 1] *= f
                        changed = True
            if not changed:
                break
        return dt
    dt = settle(dt0)
    if GANTRY_AMAX:
        dt2 = settle(np.maximum(dt0, time_param_limits(cell, X, paint)))
        if dt2.sum() < dt.sum():
            dt = dt2
    return np.r_[0.0, np.cumsum(dt)]


def auto_phi(rows):
    """겐트리 힌트 방위각 = 도장 분사 방향 수평성분 평균의 반대 (팔 베이스가 노즐 뒤에 서도록)."""
    v = np.zeros(2)
    for r in rows:
        for T, p in zip(r['T'], r['paint']):
            if p:
                v -= T[:2, 2]
    return math.degrees(math.atan2(v[1], v[0])) if np.linalg.norm(v) > 1e-6 else 180.0


def run(a):
    global TRANSIT_MODE
    TRANSIT_MODE = getattr(a, 'transit', 'line')
    global GUIDE_DEG
    GUIDE_DEG = getattr(a, 'guide_deg', 0)
    global ALLOW_FLIP, ELBOW
    ALLOW_FLIP = getattr(a, 'allow_flip', False)
    ELBOW = getattr(a, 'elbow', '')
    global W_ELBOW, W_LARM, GANTRY_RULE
    W_ELBOW = getattr(a, 'w_elbow', 2.0)
    W_LARM = getattr(a, 'w_larm', 0.0)
    GANTRY_RULE = getattr(a, 'gantry_rule', 'adaptive')
    global W_GANTRY
    W_GANTRY = getattr(a, 'w_gantry', 1.0)
    t0 = time.perf_counter()
    pool = None
    if a.workers > 1:                      # 인증용 워커를 먼저 띄워 모델 로딩을 사슬 계산과 겹친다
        import multiprocessing as mp
        pool = mp.get_context('fork').Pool(a.workers, initializer=pb._winit, initargs=(a.urdf, a.gun))
    cell = pb._make(a.urdf, a.gun)
    rows = load_targets(cell)
    skipped = []
    if getattr(a, 'skip', ''):                          # reach_points.py 결과: 닿지 않는 점은 빼고 계획 (시험용)
        bad = {p['csv_idx'] for p in json.loads(Path(a.skip).read_text())['points'] if p['status'] != 'ok'}
        skipped = sorted(bad)
        nr = []
        for r in rows:
            keep = np.array([int(i) not in bad for i in r['csv_idx']])
            if keep.any():
                nr.append({'row': r['row'], 'T': r['T'][keep], 'paint': r['paint'][keep], 'csv_idx': r['csv_idx'][keep]})
        rows = nr
    t_load = time.perf_counter() - t0
    phi = auto_phi(rows) if str(a.phi) == 'auto' else float(a.phi)
    a.phi = phi
    S = ChainSolver(cell, a.method, phi, a.rdes)
    zsafe = max(T[2, 3] for r in rows for T in r['T']) + .10
    t1 = time.perf_counter()
    prior = None
    if getattr(a, 'prior', ''):
        try:
            prior = {int(k): v for k, v in json.loads(Path(a.prior).read_text())['row_hints'].items()}
        except Exception as e:                              # 레퍼런스가 없거나 형식이 다르면 일반 탐색
            print(f'[prior ignored: {e}]', file=sys.stderr)
    X, C, log = solve_chain(S, rows, zsafe, prior=prior)
    t_chain = time.perf_counter() - t1
    base = {'method': a.method, 'gun': a.gun, 'phi_w_deg': a.phi, 'rdes': a.rdes, 'urdf': a.urdf, 'home_arm': a.home_arm,
            'transit': a.transit, 'guide_deg': a.guide_deg, 'w_gantry': a.w_gantry,
            'skipped_csv_idx': skipped, 'elbow': getattr(a, 'elbow', '')}
    if X is None:
        return {**base, 'status': 'fail', 'stage': 'chain', 'log': log[-3:],
                'timing_s': {'load': t_load, 'chain': t_chain}}
    t2 = time.perf_counter()
    gs_info = None
    if getattr(a, 'smooth_gantry', False):
        X, gs_info = smooth_gantry(S, X, C)
    X, C, rinfo = refine(S, X, C)
    t_ref = time.perf_counter() - t2
    if X is None:
        return {**base, 'status': 'fail', 'stage': 'refine', 'info': rinfo}
    Xh, Ch = add_home(S, X, C, home_g=a.home, home_arm=a.home_arm)
    if Xh is None:
        return {**base, 'status': 'fail', 'stage': 'home'}
    t3 = time.perf_counter()
    bad, lo = certify_all(cell, Xh, a.workers, a.urdf, a.gun, pool)
    if pool is not None:
        pool.close()
    t_cert = time.perf_counter() - t3
    m = metrics(cell, Xh, Ch, rows)
    pf = np.array([c[2] for c in Ch])
    times = time_param_smooth(cell, Xh, pf)
    P = np.array([cell.tcp(x)[0] for x in Xh])
    seg = np.flatnonzero((pf[:-1] == 1) & (pf[1:] == 1))
    L = np.linalg.norm(np.diff(P, axis=0), axis=1)[seg]
    v = L / np.diff(times)[seg]
    m.update(rinfo)
    m.update(motion_stats(cell, Xh, Ch))
    m['gantry_smooth_window'] = gs_info
    m.update({'certified_lower_bound_mm': lo * 1000, 'uncertified_edges': len(bad),
              'cycle_time_s': float(times[-1]),
              'paint_at_target_speed_percent': float(100 * L[v >= pb.TCP_SPEED * .99].sum() / L.sum()),
              'paint_tcp_speed_min_mps': float(v.min()),
              'paint_joint_speed_max_dps': float(np.degrees((np.abs(np.diff(Xh[:, 2:], axis=0))[seg] / np.diff(times)[seg][:, None]).max())),
              'joint_accel_max_dps2': float(np.degrees(_accel_max(Xh, times))),
              'hop_variants': [h.get('hop_variant') for h in log]})
    total = sum(len(r['T']) for r in rows)
    ok = not bad and m['coverage'] == f'{total}/{total}'
    row_hints = {}
    for l in log:
        r_ = l.get('row', l.get('into_row'))
        if r_ is not None and 'why' not in l:
            row_hints[int(r_)] = {'phi': l['phi'], 'rdes': l['rdes'], 'hop': l.get('hop_variant')}
    res = {**base, 'status': 'verified' if ok else 'NOT verified', 'metrics': m, 'row_hints': row_hints,
           'used_prior': bool(prior),
           'timing_s': {'load': t_load, 'chain': t_chain, 'refine': t_ref, 'certify(parallel)': t_cert,
                        'planning_total': t_chain + t_ref + t_cert}, 'stats': S.stats}
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        tool = cell.tool
        np.savez(out / 'trajectory_bumper.npz', X=Xh, t=times, names=np.array(pb.GANTRY + pb.ARM), tcp=P,
                 paint=pf, kind=np.array([c[1] for c in Ch]), row=np.array([c[3] for c in Ch]),
                 csv_idx=np.array([c[4] for c in Ch]),
                 tool_names=np.array(list(tool)), tool_values=np.array(list(tool.values())))
        with open(out / 'joints_bumper.csv', 'w') as f:
            f.write('t,' + ','.join(pb.GANTRY + pb.ARM) + ',gun_lift,gun_swing,paint,kind,row,csv_idx\n')
            for ti, x, c in zip(times, Xh, Ch):
                f.write(f'{ti:.4f},' + ','.join(f'{v:.6f}' for v in x) +
                        f",{tool['gun_lift']},{tool['gun_swing']},{c[2]},{c[1]},{c[3]},{c[4]}\n")
        res.update({'tool_state': tool, 'urdf_sha256': sha(a.urdf), 'csv_sha256': sha(CSV),
                    'collision_exclusions': [list(e) for e in cell.exclusions if e[2] != 'adjacent assembly'],
                    'trajectory_sha256': sha(out / 'joints_bumper.csv'),
                    'metrics': {**m, 'certified_clearance_lower_bound_mm': lo * 1000}})
        (out / 'validation.json').write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--urdf', required=True)
    ap.add_argument('--gun', default='on', choices=['on', 'off'])
    ap.add_argument('--method', default='wholebody', choices=['decoupled', 'wholebody'])
    ap.add_argument('--phi', default='180', help="겐트리 힌트 월드 방위각 [deg] 또는 'auto' (분사 방향 뒤)")
    ap.add_argument('--rdes', type=float, default=.85)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--out', default='')
    ap.add_argument('--home', default='upper', help="겐트리 홈: 'upper'(상한 모서리, 기본) | 'lower' | 'x,y'")
    ap.add_argument('--transit', default='line', choices=['line', 'free'],
                    help="줄 사이 이동: 'line'(파란 직선 그대로, 기본) | 'free'(최적 이동: 관절공간 보간)")
    ap.add_argument('--guide-deg', type=int, default=0, help='겐트리 안내 곡선 다항식 차수 (0 = 이동평균, 2~3 권장: 겐트리 왕복 제거)')
    ap.add_argument('--w-gantry', type=float, default=1.0, help='겐트리를 안내 곡선으로 당기는 세기 (기본 1)')
    ap.add_argument('--gantry-rule', default='adaptive', choices=['adaptive', 'fender'], help='분리 방식 겐트리 규칙')
    ap.add_argument('--w-larm', type=float, default=0.0, help='위팔 수평(ㄱ자) 유지 세기')
    ap.add_argument('--w-elbow', type=float, default=2.0, help='팔꿈치 위 유지 세기 (--elbow up 일 때)')
    ap.add_argument('--elbow', default='', choices=['', 'up', 'down'], help="팔꿈치 방향: up = 손목보다 위(현장 사진처럼), down = 아래")
    ap.add_argument('--smooth-gantry', action='store_true', help='줄 안 겐트리 궤적 매끄럽게 (왕복/떨림 제거) 후 팔만 다시 풂')
    ap.add_argument('--home-arm', default='ready',
                    help="홈 팔 자세: 'ready'(첫 도장점 준비 자세, 기본) | 'stow'(접은 자세) | 'j1,j2,j3,j4,j5,j6'(deg 지정)")
    ap.add_argument('--allow-flip', action='store_true', help='시험용: 줄 사이 이동에서 팔 자세 뒤집기 허용 (기본 금지)')
    ap.add_argument('--skip', default='', help='reach_points.py 결과 json: 닿지 않는 점을 빼고 계획 (시험용)')
    ap.add_argument('--prior', default='', help='레퍼런스: 이전 계획의 validation.json (줄별 힌트/이동 방식을 먼저 시도)')
    a = ap.parse_args()
    print(json.dumps(run(a), indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
