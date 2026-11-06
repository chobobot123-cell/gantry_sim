#!/usr/bin/env python3
"""기존 STL 분리 모델에서 RViz 시각화용 8 x 5 x 3 m 셀을 생성한다."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/home/jhin/gantry_test_40mm')
OUT = ROOT / 'meshes'
OUT.mkdir(exist_ok=True)
tree = ET.parse(SOURCE / 'paint_cell_bumper_h255.urdf')
robot = tree.getroot()
links = {link.get('name'): link for link in robot.findall('link')}
joints = {joint.get('name'): joint for joint in robot.findall('joint')}

# 기존의 7 x 5 x 2.55 m STL 셀에서 X 레일을 1 m, 기둥을 0.45 m 늘린다.
# 중앙 절단면을 건너는 삼각형을 늘리는 형상 변형이다. 실제 제작 도면이 아니다.
for part, link_name in [('fixed', 'world_base'), ('bridge', 'gantry_x_link'), ('carriage', 'gantry_y_link')]:
    source = SOURCE / 'meshes' / f'stl7x5h255_gantry_{part}.stl'
    mesh = trimesh.load(source, force='mesh')
    vertices = mesh.vertices.copy()
    if part == 'fixed':
        vertices[:, 0] += np.where(vertices[:, 0] > .071, .5, -.5)
        vertices[:, 2] += np.where(vertices[:, 2] > 1.4, .45, 0.)
    target = OUT / f'gantry_{part}.stl'
    trimesh.Trimesh(vertices, mesh.faces, process=False).export(target)
    links[link_name].find('visual/geometry/mesh').set('filename', target.as_uri())
    # 기존 근사 충돌 박스는 늘어난 형상에 맞지 않으므로 사용하지 않는다.
    for collision in links[link_name].findall('collision'):
        links[link_name].remove(collision)

gantry_x = joints['gantry_x']
origin = gantry_x.find('origin')
xyz = [float(v) for v in origin.get('xyz').split()]
xyz[2] += .45
origin.set('xyz', ' '.join(f'{v:.6f}' for v in xyz))
limit = gantry_x.find('limit')
limit.set('lower', f"{float(limit.get('lower')) - .5:.6f}")
limit.set('upper', f"{float(limit.get('upper')) + .5:.6f}")
for name in ('gantry_x', 'gantry_y'):
    joints[name].find('limit').set('velocity', '0.5')
links['bumper_link'].find('visual/geometry/mesh').set('filename', (ROOT / 'ES300h_front_bumper.ply').as_uri())
tree.write(ROOT / 'paint_cell.urdf', encoding='utf-8', xml_declaration=True)
# 첫 점 비교 실험용: 범퍼와 CSV를 같은 프레임에서 0.45 m 올린 배치.
part_origin = joints['world_base-bumper'].find('origin')
part_xyz = [float(v) for v in part_origin.get('xyz').split()]
part_xyz[2] += .45
part_origin.set('xyz', ' '.join(f'{v:.6f}' for v in part_xyz))
tree.write(ROOT / 'paint_cell_raised.urdf', encoding='utf-8', xml_declaration=True)
part_xyz[2] += .30
part_origin.set('xyz', ' '.join(f'{v:.6f}' for v in part_xyz))
tree.write(ROOT / 'paint_cell_233.urdf', encoding='utf-8', xml_declaration=True)
print('생성: paint_cell.urdf, paint_cell_raised.urdf, paint_cell_233.urdf 및 갠트리 메쉬')
