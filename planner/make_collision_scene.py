#!/usr/bin/env python3
"""시각화 셀에 기존 7x5 갠트리의 근사 충돌 박스를 늘려 붙인다.

기존 박스는 근사치이며 새 형상에 대한 독립적인 적합성 검증은 필요하다.
"""
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
source = ET.parse('/home/jhin/gantry_test_40mm/paint_cell_bumper_h255.urdf').getroot()
tree = ET.parse(ROOT / 'paint_cell.urdf')
for name in ('world_base', 'gantry_x_link', 'gantry_y_link'):
    dst = tree.find(f"link[@name='{name}']")
    for old in dst.findall('collision'):
        dst.remove(old)
    for old in source.find(f"link[@name='{name}']").findall('collision'):
        col = deepcopy(old)
        box = col.find('geometry/box')
        if name == 'world_base' and box is not None:
            origin = col.find('origin')
            pos = list(map(float, origin.get('xyz').split()))
            size = list(map(float, box.get('size').split()))
            for axis, cut, delta in ((0, .071, 1.), (2, 1.4, .45)):
                lo, hi = pos[axis] - size[axis]/2, pos[axis] + size[axis]/2
                if axis == 0:
                    lo += delta/2 if lo > cut else -delta/2
                    hi += delta/2 if hi > cut else -delta/2
                else:
                    lo += delta if lo > cut else 0
                    hi += delta if hi > cut else 0
                pos[axis], size[axis] = (lo + hi)/2, hi - lo
            origin.set('xyz', ' '.join(f'{v:.6f}' for v in pos))
            box.set('size', ' '.join(f'{v:.6f}' for v in size))
        dst.append(col)
out = ROOT / 'planner' / 'paint_cell_collision.urdf'
tree.write(out, encoding='utf-8', xml_declaration=True)
print(f'{out}: 기존 갠트리 근사 박스 적용, 실제 형상에 대한 독립 검증 필요')
