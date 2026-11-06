"""셀 기구학 + 충돌 모델 (pinocchio + coal).

~/gantry/cell_collision.py 를 이 프로젝트용으로 옮긴 것. 차이점:
  * 경로/의존성 자립 (DIR, read_ply, TOOL_STATE 내장)
  * 볼록포 정점을 .npz 로 캐시해 로딩 시간을 계획 시간에서 뺀다
  * 셀 조립품 내부의 영구 겹침 쌍을 명시적으로 제외하고 이유를 남긴다

형상: 팔 충돌 메쉬 · 툴 비주얼 메쉬는 볼록포(보수적), 셀은 URDF 박스,
범퍼는 삼각형 그대로(볼록포는 범퍼 안쪽을 메워 버린다). 단위 m.
"""
import hashlib
import os
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import coal
import numpy as np
import pinocchio as pin
import trimesh
from scipy.spatial import ConvexHull

DIR = Path(__file__).resolve().parent.parent
CACHE = DIR / 'planner' / 'cache'
ARM = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
GANTRY = ['gantry_x', 'gantry_y']
# 건 상태. off = 수납(lift 0), on = 전개(도장 자세, swing 은 lift 에 캠 연동 x17.3773)
GUN = {'off': {'gun_lift': 0.0, 'gun_swing': 0.0},
       'on': {'gun_lift': 0.0736, 'gun_swing': 1.27897}}
TOOL_STATE = {'bumper': GUN['on'], 'fender': GUN['on'], 'rfender': GUN['on'], 'hood': GUN['on'], 'trunk': GUN['on']}


def R_rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def origin_T(el):
    T = np.eye(4)
    if el is not None:
        T[:3, :3] = R_rpy(*map(float, el.get('rpy', '0 0 0').split()))
        T[:3, 3] = list(map(float, el.get('xyz', '0 0 0').split()))
    return T


def read_ply(path):
    """PLY (binary_little_endian / ascii) -> (정점, 삼각형)."""
    f = open(path, 'rb')
    hdr = []
    while True:
        line = f.readline().decode(errors='replace').strip()
        hdr.append(line)
        if line == 'end_header':
            break
    nv = nf = 0
    vprops, fmt = [], 'ascii'
    cur = None
    for line in hdr:
        t = line.split()
        if t[:1] == ['format']:
            fmt = t[1]
        elif t[:1] == ['element']:
            cur = t[1]
            if cur == 'vertex':
                nv = int(t[2])
            elif cur == 'face':
                nf = int(t[2])
        elif t[:1] == ['property'] and cur == 'vertex':
            vprops.append(t[1])
    size = {'float': 4, 'float32': 4, 'double': 8, 'uchar': 1, 'uint8': 1, 'int': 4, 'uint': 4}
    code = {'float': 'f4', 'float32': 'f4', 'double': 'f8', 'uchar': 'u1', 'uint8': 'u1',
            'int': 'i4', 'uint': 'u4'}
    if fmt == 'ascii':
        rows = [f.readline().split() for _ in range(nv)]
        V = np.array([[float(v) for v in r[:3]] for r in rows])
        F = np.array([[int(v) for v in f.readline().split()[1:4]] for _ in range(nf)])
        return V, F
    dt = np.dtype([(f'p{i}', '<' + code[p]) for i, p in enumerate(vprops)])
    raw = np.frombuffer(f.read(nv * sum(size[p] for p in vprops)), dtype=dt)
    V = np.stack([raw['p0'], raw['p1'], raw['p2']], 1).astype(float)
    rest = np.frombuffer(f.read(), dtype=np.uint8)
    # 모든 면이 삼각형(uchar 3 + int32*3 = 13 byte)이라고 가정하고 검증한다
    F = np.ndarray((nf,), dtype=np.dtype([('n', 'u1'), ('i', '<i4', 3)]), buffer=rest[:nf * 13])
    assert np.all(F['n'] == 3), 'triangles only'
    return V, F['i'].astype(int)


def package_share(package):
    from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
    try:
        return Path(get_package_share_directory(package))
    except PackageNotFoundError:
        ws = {'dsr_description2': 'doosan_ws', 'spraygun_description': 'spraygun_ws'}[package]
        p = Path('/home/jhin') / ws / 'install' / package / 'share' / package
        if not p.is_dir():
            raise RuntimeError(f'Source the ROS workspace containing {package}')
        return p


def resolve(filename):
    if filename.startswith('package://'):
        pkg, rel = filename[len('package://'):].split('/', 1)
        return str(package_share(pkg) / rel)
    return filename.removeprefix('file://')


def hull_vertices(vertices, key):
    """볼록포 정점을 캐시한다 (키 = 원본 정점 해시)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(np.ascontiguousarray(vertices, dtype=np.float64).tobytes()).hexdigest()[:16]
    f = CACHE / f'hull_{key}_{h}.npz'
    if f.exists():
        z = np.load(f)
        return z['v'], z['f']
    hull = ConvexHull(vertices)
    m = trimesh.Trimesh(vertices=vertices, faces=hull.simplices, process=False)
    m.fix_normals()
    m.remove_unreferenced_vertices()
    np.savez(f, v=m.vertices, f=m.faces)
    return m.vertices, m.faces


def make_convex(v, f):
    vc, fc = coal.StdVec_Vec3s(), coal.StdVec_Triangle()
    vc.extend(v)
    fc.extend([coal.Triangle(*map(int, t)) for t in f])
    return coal.Convex(vc, fc)


class Cell:
    """기구학 + 충돌 거리 + 구간 연속 인증."""

    def __init__(self, urdf, name='bumper', gun=None, gantry=None):
        self.urdf = str(urdf)
        self.name = name
        self.tool = dict(GUN[gun]) if gun else dict(TOOL_STATE[name])
        # 고정 관절 추가 (예: FIXED_JOINTS="gantry_z=1.123" -> 상하축을 그 값에 고정, 플래너는 8축 그대로)
        for kv in filter(None, os.environ.get('FIXED_JOINTS', '').split(',')):
            k, v = kv.split('=')
            self.tool[k.strip()] = float(v)
        root = ET.parse(self.urdf).getroot()
        self.root = root
        self.model = pin.buildModelFromUrdf(self.urdf)
        self.data = self.model.createData()
        self.q = pin.neutral(self.model)
        self.gantry = list(gantry) if gantry else list(GANTRY)
        self.names = self.gantry + ARM
        self.iq = np.array([self.model.joints[self.model.getJointId(n)].idx_q for n in self.names])
        self.iv = np.array([self.model.joints[self.model.getJointId(n)].idx_v for n in self.names])
        for n, v in self.tool.items():
            self.q[self.model.joints[self.model.getJointId(n)].idx_q] = v
        self.lower = self.model.lowerPositionLimit[self.iq].copy()
        self.upper = self.model.upperPositionLimit[self.iq].copy()
        self.vel = self.model.velocityLimit[self.iv].copy()
        self.fid_tcp = self.model.getFrameId('gun_nozzle')
        wp = root.find(f"joint[@name='world_base-{name}']/origin")
        self.T_wp = origin_T(wp)                      # 워크피스 배치 (CSV 프레임 -> 월드)
        self._build_geometry(root)

    # ---------------- 기구학 ----------------
    def set(self, x):
        self.q[self.iq] = x
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

    def tcp(self, x):
        self.set(x)
        M = self.data.oMf[self.fid_tcp]
        return M.translation.copy(), M.rotation.copy()

    def frame(self, x, name):
        self.set(x)
        M = self.data.oMf[self.model.getFrameId(name)]
        return M.translation.copy(), M.rotation.copy()

    def jacobian_tcp(self, x):
        """TCP 6x8 자코비안 (LOCAL_WORLD_ALIGNED: [v; w] 월드축)."""
        self.q[self.iq] = x
        J = pin.computeFrameJacobian(self.model, self.data, self.q, self.fid_tcp,
                                     pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        M = self.data.oMf[self.fid_tcp]
        return J[:, self.iv], M.translation.copy(), M.rotation.copy()

    # ---------------- 충돌 형상 ----------------
    def _build_geometry(self, root):
        roots = [str(package_share(p).parent) for p in ('dsr_description2', 'spraygun_description')]
        self.geom = pin.buildGeomFromUrdf(self.model, self.urdf, pin.GeometryType.COLLISION,
                                         package_dirs=roots)
        wp_link = self.name + '_link'
        # 같은 셀에 놓인 다른 작업물 (예: 범퍼 셀의 펜더): 움직이지 않는 장애물로 본다
        others = [l.get('name') for l in root.findall('link')
                  if l.get('name') != wp_link and l.get('name').endswith('_link') and l.find('visual/geometry/mesh') is not None
                  and l.find('visual/geometry/mesh').get('filename', '').endswith('.ply')]
        for obj in list(self.geom.geometryObjects):
            if self.model.frames[obj.parentFrame].name in [wp_link] + others:
                self.geom.removeGeometryObject(obj.name)       # bbox 박스 대신 실제 메쉬
        for obj in self.geom.geometryObjects:
            g = obj.geometry
            if isinstance(g, coal.BVHModelOBBRSS):
                V = np.array([g.vertex(i) for i in range(g.num_vertices)])
                obj.geometry = make_convex(*hull_vertices(V, obj.name.replace('/', '_')))
        # 건 부품은 URDF 에 visual 만 있다 -> visual 메쉬 볼록포를 충돌 형상으로 추가
        for link in root.findall('link'):
            nm = link.get('name')
            if not nm.startswith('gun_'):
                continue
            for vi, vis in enumerate(link.findall('visual')):
                mesh = vis.find('geometry/mesh')
                if mesh is None:
                    continue                                   # gun_nozzle 8 mm 구는 표시용
                V = trimesh.load(resolve(mesh.get('filename')), force='mesh').vertices.copy()
                V *= np.array(list(map(float, mesh.get('scale', '1 1 1').split())))
                fid = self.model.getFrameId(nm)
                fr = self.model.frames[fid]
                obj = pin.GeometryObject(f'{nm}_envelope_{vi}', fr.parentJoint, fid,
                                         fr.placement * pin.SE3(origin_T(vis.find('origin'))),
                                         make_convex(*hull_vertices(V, f'{nm}_{vi}')))
                self.geom.addGeometryObject(obj)
        # 워크피스: 삼각형 그대로 (다른 작업물은 'obstacle_이름' 으로 같은 방식)
        for link_name, obj_name in [(wp_link, 'workpiece')] + [(o, 'obstacle_' + o[:-5]) for o in others]:
            vis = root.find(f"link[@name='{link_name}']/visual")
            mesh = vis.find('geometry/mesh')
            V, F = read_ply(resolve(mesh.get('filename')))
            V = V * np.array(list(map(float, mesh.get('scale', '1 1 1').split())))
            vc, fc = coal.StdVec_Vec3s(), coal.StdVec_Triangle()
            vc.extend(V)
            fc.extend([coal.Triangle(*map(int, t)) for t in F])
            bvh = coal.BVHModelOBBRSS()
            bvh.beginModel(len(F), len(V))
            bvh.addSubModel(vc, fc)
            bvh.endModel()
            fid = self.model.getFrameId(link_name)
            fr = self.model.frames[fid]
            self.geom.addGeometryObject(pin.GeometryObject(
                obj_name, 0, fid, fr.placement * pin.SE3(origin_T(vis.find('origin'))), bvh))

        def rank(obj):
            nm = self.model.frames[obj.parentFrame].name
            if nm == 'base_link':
                return 0
            if nm.startswith('link_'):
                return int(nm[5:])
            if nm.startswith('gun_'):
                return 7
            return -1

        self.exclusions = []
        pairs = []
        for i, a in enumerate(self.geom.geometryObjects):
            for j in range(i + 1, self.geom.ngeoms):
                b = self.geom.geometryObjects[j]
                ra, rb = rank(a), rank(b)
                obst = a.name.startswith('obstacle_') or b.name.startswith('obstacle_')
                if ra < 0 and rb < 0 and 'workpiece' not in (a.name, b.name) and not obst:
                    continue                                   # 셀 부품끼리 = 기존 조립품
                if obst and a.parentJoint == 0 and b.parentJoint == 0:
                    continue                                   # 다른 작업물 <-> 작업물/고정 프레임: 둘 다 안 움직임
                if ra >= 0 and rb >= 0 and (abs(ra - rb) <= 1 or {ra, rb} == {4, 6}):
                    self.exclusions.append((a.name, b.name, 'adjacent assembly'))
                    continue
                pairs.append((i, j))
        for i, j in pairs:
            self.geom.addCollisionPair(pin.CollisionPair(i, j))
        self._finish_pairs()

    def _finish_pairs(self):
        self.geom_data = self.geom.createData()
        for req in self.geom_data.distanceRequests:
            req.enable_signed_distance = True
        b = []
        for obj in self.geom.geometryObjects:
            obj.geometry.computeLocalAABB()
            bb = obj.geometry.aabb_local
            b.append([bb.min_, bb.max_])
        b = np.array(b)
        self.centers = b.mean(1)
        self.halves = (b[:, 1] - b[:, 0]) / 2
        self.ia = np.array([p.first for p in self.geom.collisionPairs])
        self.ib = np.array([p.second for p in self.geom.collisionPairs])
        self.motion_bounds = self._motion_bounds()
        self.evaluations = 0

    def exclude_pairs(self, idx, reason):
        """셀 조립 구조상 항상 겹치는 쌍 제거 (이유 기록)."""
        idx = set(int(i) for i in idx)
        keep = []
        for k, p in enumerate(self.geom.collisionPairs):
            if k in idx:
                self.exclusions.append((self.geom.geometryObjects[p.first].name,
                                        self.geom.geometryObjects[p.second].name, reason))
            else:
                keep.append((p.first, p.second))
        self.geom.removeAllCollisionPairs()
        for i, j in keep:
            self.geom.addCollisionPair(pin.CollisionPair(i, j))
        self._finish_pairs()

    def pair_names(self, k):
        p = self.geom.collisionPairs[int(k)]
        return self.geom.geometryObjects[p.first].name, self.geom.geometryObjects[p.second].name

    # ---------------- 거리 ----------------
    def distances(self, x, near=.10):
        """쌍별 거리. near(스칼라 또는 쌍별 배열) 이상 떨어진 쌍은 AABB 간격(=거리의 하한)을
        그대로 돌려준다 -> 여유를 과대평가하지 않으면서 정확 계산을 줄인다."""
        self.set(x)
        pin.updateGeometryPlacements(self.model, self.data, self.geom, self.geom_data)
        self.evaluations += 1
        Rm = np.array([o.rotation for o in self.geom_data.oMg])
        tm = np.array([o.translation for o in self.geom_data.oMg])
        c = np.einsum('nij,nj->ni', Rm, self.centers) + tm
        h = np.einsum('nij,nj->ni', abs(Rm), self.halves)
        gap = np.maximum(abs(c[self.ia] - c[self.ib]) - h[self.ia] - h[self.ib], 0)
        d = np.linalg.norm(gap, axis=1)
        for k in np.flatnonzero(d < near):
            d[k] = pin.computeDistance(self.geom, self.geom_data, int(k)).min_distance
        return d

    def _motion_bounds(self):
        """쌍 거리의 관절별 립시츠 상수 (공통 조상 관절은 상대 기하를 안 바꾸므로 제외)."""
        ng = self.geom.ngeoms
        radii = np.zeros((ng, len(self.names)))
        anc = np.zeros_like(radii, dtype=bool)
        col_of = {self.model.getJointId(n): i for i, n in enumerate(self.names)}
        for gi, obj in enumerate(self.geom.geometryObjects):
            r = np.linalg.norm(obj.placement.act(self.centers[gi])) + np.linalg.norm(self.halves[gi])
            j = obj.parentJoint
            while j:
                if j in col_of:
                    c = col_of[j]
                    anc[gi, c] = True
                    radii[gi, c] = 1.0 if self.model.joints[j].shortname().startswith('JointModelP') else r   # 직선축: 1, 회전축: 반지름
                r += np.linalg.norm(self.model.jointPlacements[j].translation)
                if self.model.names[j] == 'gun_lift':
                    r += abs(self.q[self.model.joints[j].idx_q])
                j = self.model.parents[j]
        ex = anc[self.ia] ^ anc[self.ib]
        return (radii[self.ia] + radii[self.ib]) * ex

    def certify_edge(self, a, b, clearance=.005, max_depth=18):
        """관절 선분 a->b 위 '모든' 배치가 clearance 이상임을 증명 (샘플링 아님)."""
        stack = [(np.asarray(a, float), np.asarray(b, float), 0)]
        count, lo_min = 0, float('inf')
        while stack:
            l, r, depth = stack.pop()
            m = (l + r) / 2
            mb = self.motion_bounds @ (abs(r - l) / 2)
            # AABB 하한만으로 이미 증명되는 쌍은 정확 거리를 생략 (하한 - mb >= clearance)
            d = self.distances(m, near=clearance + mb + 1e-6)
            count += 1
            if not np.all(np.isfinite(d)) or d.min() < clearance:
                return False, float(d.min()), count
            lower = d - mb
            if lower.min() >= clearance:
                lo_min = min(lo_min, float(lower.min()))
                continue
            if depth >= max_depth:
                return False, float(lower.min()), count
            stack += [(l, m, depth + 1), (m, r, depth + 1)]
        return True, lo_min, count