#!/usr/bin/env python3
"""범퍼 셀(2.31 m 배치, 같은 겐트리·로봇·건)에서 범퍼만 다른 부품으로 바꾼 장면을 만든다.

부품 배치 (월드 = 셀 기준, 원점 ≈ 셀 중앙, 단위 m):
  fender  (ES300h 왼쪽 펜더)  : 스캔 자세 그대로 세움, 점군 중심 X 0 / Y -1.0, 밑면 1.55 m. 칠할 면 +Y (셀 중앙 쪽)
  rfender (NX350h 오른쪽 펜더): 왼쪽 펜더의 반대편. Z축 180° 회전, 점군 중심 X 0 / Y +1.0, 밑면 1.55 m. 칠할 면 -Y
  hood    (ES300h 후드)       : 눕힌 그대로 셀 중앙, 노즐 경로 중심 높이 1.5 m (위에서 아래로 분사)
  trunk   (ES300h 트렁크)     : 눕힌 그대로 셀 중앙, 노즐 경로 중심 높이 1.5 m
펜더 밑면 1.55~1.60 m: 1.15 m(예전 배치를 0.45 m 높인 값)에서는 아래쪽 줄(1.4 m 이하)에 팔이 닿지 않아 올렸다.
왼쪽 펜더 1.55 m(1.50 m 실패), 오른쪽 펜더도 1.55 m로 맞춤(최저 1.54 m, 1.53 m 실패). 노즐 경로 약 1.5~2.5 m로 범퍼(약 1.7~2.6 m)와 비슷한 높이. 모두 시뮬레이션 가정이다.
출력: paint_cell_<부품>.urdf (RViz), planner/paint_cell_collision_<부품>.urdf (계획)
"""
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from cell_model import read_ply

ROOT = Path(__file__).resolve().parent.parent
SRC_VIS = ROOT / 'paint_cell_231.urdf'
SRC_COL = ROOT / 'planner' / 'paint_cell_collision_231.urdf'
PARTS = {   # 이름: (폴더, yaw[deg], 배치 기준)
    'fender':  ('ES300h_left_fender', 0.0, dict(cx=0.0, cy=-1.0, zmin=1.55)),
    'rfender': ('NX350h_right_fender', 180.0, dict(cx=0.0, cy=1.0, zmin=1.55)),
    'hood':    ('ES300h_hood', 0.0, dict(cx=0.0, cy=0.0, path_z=1.5)),
    'trunk':   ('ES300h_trunk', 0.0, dict(cx=0.0, cy=0.0, path_z=1.5)),
}


def placement(folder, yaw, ref):
    V, _ = read_ply(ROOT / f'{folder}.ply')
    d = np.genfromtxt(ROOT / folder / 'painting_trajectory.csv', delimiter=',', names=True)
    P = np.c_[d['position_x'], d['position_y'], d['position_z']]
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    V, P = V @ Rz.T, P @ Rz.T
    mid = (V.min(0) + V.max(0)) / 2
    t = np.array([ref['cx'] - mid[0], ref['cy'] - mid[1], 0.0])
    if 'zmin' in ref:
        t[2] = ref['zmin'] - V[:, 2].min()
    else:
        t[2] = ref['path_z'] - (P[:, 2].min() + P[:, 2].max()) / 2
    return t, V + t, P + t


# RViz 가 ES300h_trunk.ply 를 읽다가 비정상 종료(Segmentation fault)한다 (메쉬 정리 후에도 동일, 다른 PLY 는 정상).
# 표시용 URDF 에만 같은 형상의 STL 을 쓴다. 계획(충돌)은 원본 PLY 를 그대로 쓴다.
VIS_STL = {'ES300h_trunk'}


def rviz_mesh(folder):
    stl = ROOT / f'{folder}_rviz.stl'
    if not stl.exists():
        import trimesh
        V, F = read_ply(ROOT / f'{folder}.ply')
        trimesh.Trimesh(V, F, process=False).export(stl)
    return stl


def build(src, dst, name, folder, t, yaw, mesh=None):
    tree = ET.parse(src)
    root = tree.getroot()
    link = root.find("link[@name='bumper_link']")
    link.set('name', f'{name}_link')
    for col in link.findall('collision'):          # 계획기는 PLY 삼각형을 직접 쓴다
        link.remove(col)
    link.find('visual/geometry/mesh').set('filename', (mesh or ROOT / f'{folder}.ply').as_uri())
    if mesh is not None and link.find('visual/material') is None:   # STL 은 색 정보가 없어 RViz 가 빨갛게 그린다 -> 회색 지정
        mat = ET.SubElement(link.find('visual'), 'material', name=f'{name}_gray')
        ET.SubElement(mat, 'color', rgba='0.78 0.78 0.80 1')
    joint = root.find("joint[@name='world_base-bumper']")
    joint.set('name', f'world_base-{name}')
    joint.find('child').set('link', f'{name}_link')
    o = joint.find('origin')
    o.set('xyz', ' '.join(f'{v:.6f}' for v in t))
    o.set('rpy', f'0 0 {math.radians(yaw):.9f}')
    tree.write(dst, encoding='utf-8', xml_declaration=True)


if __name__ == '__main__':
    for name, (folder, yaw, ref) in PARTS.items():
        t, V, P = placement(folder, yaw, ref)
        build(SRC_VIS, ROOT / f'paint_cell_{name}.urdf', name, folder, t, yaw, rviz_mesh(folder) if folder in VIS_STL else None)
        build(SRC_COL, ROOT / 'planner' / f'paint_cell_collision_{name}.urdf', name, folder, t, yaw)
        print(f'{name:8s} 이동 {np.round(t, 3)} yaw {yaw:5.1f} | 점군 X {V[:,0].min():.2f}~{V[:,0].max():.2f} '
              f'Y {V[:,1].min():.2f}~{V[:,1].max():.2f} Z {V[:,2].min():.2f}~{V[:,2].max():.2f} | 경로 Z {P[:,2].min():.2f}~{P[:,2].max():.2f}')
