#!/usr/bin/env python3
"""도장 경로 CSV(painting_trajectory.csv)를 RViz 마커로 발행.

사용법:  publish_path.py bumper | fender

CSV 좌표는 워크피스 PLY 와 같은 원본 스캔 프레임이므로 마커를 워크피스 링크
(bumper_link / fender_link) 프레임에 그대로 찍는다. URDF 배치를 바꿔도 자동으로 따라간다.
  초록 선 = 분사(paint=1), 회색 선 = 비분사 이동(paint=0)
  파란 선 = 줄 끝 -> 다음 줄 시작 (줄 사이 이동). --traj npz 를 주면 계획된 궤적이 실제로 지나가는 길
  --hide json = 계획에서 건너뛴 점은 그리지 않음, --mark json = 닿지 않는 점을 빨간 점으로 표시
"""
import csv
import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

DIR = os.path.dirname(os.path.abspath(__file__))
VARIANTS = {'bumper': ('bumper_link', 'ES300h_front_bumper'),
            'fender': ('fender_link', 'ES300h_left_fender'),
            'rfender': ('rfender_link', 'NX350h_right_fender'),
            'hood': ('hood_link', 'ES300h_hood'),
            'trunk': ('trunk_link', 'ES300h_trunk')}


def height_markers(urdf, marker, link='bumper_link'):
    """지면(z=0) -> 범퍼 중앙부 밑면 치수선 + 범퍼 최저점 높이 표시 (월드 프레임)."""
    import xml.etree.ElementTree as ET
    sys.path.insert(0, os.path.join(DIR, 'planner'))
    from cell_model import read_ply, origin_T
    r = ET.parse(urdf).getroot()
    J = {j.get('name'): j for j in r.findall('joint')}
    T = origin_T(J['world_base-' + link[:-5]].find('origin'))                  # 작업물 배치 (bumper_link / fender_link)
    mesh = r.find(f"link[@name='{link}']/visual/geometry/mesh").get('filename').replace('file://', '')
    V = read_ply(mesh)[0]
    W = (T[:3, :3] @ V.T).T + T[:3, 3]
    lo, hi = W.min(0), W.max(0)
    jz = float(J['gantry_x'].find('origin').get('xyz').split()[2])
    bridge = min(float(c.find('origin').get('xyz').split()[2]) - float(c.find('geometry/box').get('size').split()[2]) / 2
                 for c in r.find("link[@name='gantry_x_link']").findall('collision') if c.find('geometry/box') is not None) + jz
    x0, y0 = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2       # 범퍼 중앙 (바운딩박스 중심) 바로 아래에 치수선
    dxy = np.hypot(W[:, 0] - x0, W[:, 1] - y0)
    rad = .03
    while not (dxy < rad).any():
        rad *= 1.5
    col = W[dxy < rad, 2]
    zc, zt = col.min(), col.max()                            # 중앙부 범퍼 밑면 / 윗면
    low_txt = f'lowest={lo[2]:.3f}m'                         # 범퍼 전체 최저점 (받침대 높이)
    top_z = .08                                              # 작은 글씨 높이 (범퍼: 바닥 근처)
    if link != 'bumper_link':                                # 범퍼 외 작업물: 치수선 = 지면 -> 실제 가장 낮은 점 바로 아래, 작은 글씨 = 꼭대기 높이에 표시
        il = int(np.argmin(W[:, 2]))
        x0, y0 = float(W[il, 0]), float(W[il, 1])            # 가장 낮은 점의 바로 아래 (가운데가 아니라)
        zc, low_txt, top_z = lo[2], f'top={hi[2]:.3f}m', hi[2]
    out = []
    dim = marker(10, Marker.LINE_LIST, 'height')
    dim.header.frame_id = 'world_base'
    dim.scale.x = 0.01
    dim.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
    tk = .08
    for z in (0.0, zc):                                      # 눈금: 지면 / 중앙 밑면
        dim.points += [Point(x=float(x0 - tk), y=float(y0), z=float(z)), Point(x=float(x0 + tk), y=float(y0), z=float(z))]
    dim.points += [Point(x=float(x0), y=float(y0), z=0.0), Point(x=float(x0), y=float(y0), z=float(zc))]   # 지면 -> 중앙 밑면
    out.append(dim)
    # RViz 텍스트는 띄어쓰기를 크게 벌려 그리므로 공백 없이
    for k, (z, txt, h) in enumerate(((zc / 2, f'{zc:.3f}m', .16),                           # 치수선 옆: 지면 -> 중앙 밑면
                                     (top_z, low_txt, .07))):
        t = marker(11 + k, Marker.TEXT_VIEW_FACING, 'height_text')
        t.header.frame_id = 'world_base'
        t.pose.position.x = float(x0 - .12 - .3 * len(txt) * h)   # 텍스트 오른쪽 끝이 치수선에 닿게
        t.pose.position.y, t.pose.position.z = float(y0), float(z)
        t.scale.z = h
        t.color = ColorRGBA(r=1.0, g=1.0, b=0.2, a=1.0)
        t.text = txt
        out.append(t)
    for k in (13, 14):                                       # 예전 버전 라벨(top/bridge 등)이 RViz 에 남지 않게 지움
        d = marker(k, Marker.TEXT_VIEW_FACING, 'height_text')
        d.header.frame_id = 'world_base'
        d.action = Marker.DELETE
        out.append(d)
    # 겐트리 높이: 범퍼 치수선 옆(-X 쪽 0.6 m)에 하늘색 치수선 — 지면 -> 겐트리 꼭대기, 눈금: 브릿지 밑면 / 팔 장착면
    import trimesh                                           # 표시용 높이는 실제 겐트리 형상(STL) 기준
    def mesh_z(link, base_z):
        el = r.find(f"link[@name='{link}']/visual/geometry/mesh")
        if el is None:
            return None
        m = trimesh.load(el.get('filename').replace('file://', ''), force='mesh')
        oz = float(r.find(f"link[@name='{link}']/visual/origin").get('xyz').split()[2])
        return m.bounds[:, 2] + oz + base_z
    zs = [z for z in (mesh_z('world_base', 0.0), mesh_z('gantry_x_link', jz), mesh_z('gantry_y_link', jz)) if z is not None]
    gtop = max(z[1] for z in zs)
    bz = mesh_z('gantry_x_link', jz)
    if bz is not None:
        bridge = float(bz[0])                                 # 브릿지 밑면 = 브릿지 형상 최저점
    arm = J['gantry_to_arm']
    mount = jz + float(arm.find('origin').get('xyz').split()[2])          # 팔 장착면 (상하축이 있으면 Z=0 기준)
    if arm.find('parent').get('link') == 'gantry_z_link':
        zj = J['gantry_z']
        mount += float(zj.find('origin').get('xyz').split()[2])
    gx = x0 - .6
    CY = ColorRGBA(r=0.2, g=0.85, b=1.0, a=1.0)
    gl = marker(30, Marker.LINE_LIST, 'gantry_height')
    gl.header.frame_id = 'world_base'
    gl.scale.x = 0.01
    gl.color = CY
    gl.points += [Point(x=float(gx), y=float(y0), z=0.0), Point(x=float(gx), y=float(y0), z=float(gtop))]
    for z in (0.0, gtop):                                   # 눈금: 지면 / 겐트리 꼭대기
        gl.points += [Point(x=float(gx - tk), y=float(y0), z=float(z)), Point(x=float(gx + tk), y=float(y0), z=float(z))]
    out.append(gl)
    labels = [(gtop, f'gantry_top={gtop:.3f}m')]
    labels.sort(key=lambda t: -t[0])
    for i in range(1, len(labels)):                          # 라벨이 겹치면 아래로 비킴
        if labels[i - 1][0] - labels[i][0] < .12:
            labels[i] = (labels[i - 1][0] - .12, labels[i][1])
    for k, (z, txt) in enumerate(labels):
        t = marker(31 + k, Marker.TEXT_VIEW_FACING, 'gantry_height_text')
        t.header.frame_id = 'world_base'
        t.pose.position.x = float(gx - .12 - .3 * len(txt) * .09)
        t.pose.position.y, t.pose.position.z = float(y0), float(z)
        t.scale.z = .09
        t.color = CY
        t.text = txt
        out.append(t)
    # 고정 기둥 길이: 캐리지(gantry_y_link) 프레임에 붙여 겐트리와 같이 움직임. 기둥 = 팔 장착 축 위의 세로 박스, 아래 끝 = 팔 장착면
    ao = list(map(float, arm.find('origin').get('xyz').split()))
    if arm.find('parent').get('link') == 'gantry_y_link':
        for c in r.find("link[@name='gantry_y_link']").findall('collision'):
            bx = c.find('geometry/box')
            if bx is None:
                continue
            o = list(map(float, c.find('origin').get('xyz').split())); sz = list(map(float, bx.get('size').split()))
            if abs(o[0] - ao[0]) < 1e-3 and abs(o[1] - ao[1]) < 1e-3 and abs(o[2] - sz[2] / 2 - ao[2]) < 1e-3 and sz[2] > .05:
                L, top, cx_ = sz[2], o[2] + sz[2] / 2, o[0] + sz[0] / 2 + .08
                OR = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)       # 기둥 길이 표시 (흰색)
                cl = marker(40, Marker.LINE_LIST, 'column_length')
                cl.header.frame_id = 'gantry_y_link'
                cl.scale.x = 0.01
                cl.color = OR
                cl.points += [Point(x=cx_, y=o[1], z=top), Point(x=cx_, y=o[1], z=ao[2])]
                for z in (top, ao[2]):
                    cl.points += [Point(x=cx_ - .05, y=o[1], z=z), Point(x=cx_ + .05, y=o[1], z=z)]
                out.append(cl)
                t = marker(41, Marker.TEXT_VIEW_FACING, 'column_length_text')
                t.header.frame_id = 'gantry_y_link'
                t.pose.position.x, t.pose.position.y, t.pose.position.z = cx_ + .12 + .3 * 12 * .08, o[1], (top + ao[2]) / 2
                t.scale.z = .08
                t.color = OR
                t.text = f'column={L:.2f}m'
                out.append(t)
                print(f'[height] column {L:.3f} m', flush=True)
                break
    for k in (32, 33):                                       # 예전 라벨(bridge_bottom / arm_mount) 지움
        d = marker(k, Marker.TEXT_VIEW_FACING, 'gantry_height_text')
        d.header.frame_id = 'world_base'
        d.action = Marker.DELETE
        out.append(d)
    print(f'[height] center bottom {zc:.3f} m, center top {zt:.3f} m, lowest {lo[2]:.3f} m, top {hi[2]:.3f} m, bridge bottom {bridge:.3f} m, gantry top {gtop:.3f} m, arm mount {mount:.3f} m', flush=True)
    return out


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in VARIANTS:
        raise SystemExit('사용법: publish_path.py bumper|fender|rfender|hood|trunk')
    frame, folder = VARIANTS[sys.argv[1]]
    csv_path = sys.argv[sys.argv.index('--csv') + 1] if '--csv' in sys.argv else f'{DIR}/{folder}/painting_trajectory.csv'
    rows = list(csv.DictReader(open(csv_path)))
    n_all = len(rows)
    if '--row' in sys.argv:
        selected_row = int(sys.argv[sys.argv.index('--row') + 1])
        rows = [r for r in rows if int(r['row_index']) == selected_row]
        if not rows:
            raise SystemExit(f'CSV에 {selected_row}번 줄이 없습니다.')
    if '--hide' in sys.argv:                            # 계획에서 건너뛴 점(닿지 않는 점)은 그리지 않는다. CSV 파일은 그대로
        import json
        hide = {p['csv_idx'] for p in json.load(open(sys.argv[sys.argv.index('--hide') + 1]))['points'] if p['status'] != 'ok'}
        rows = [r for i, r in enumerate(rows) if i not in hide]

    def P(r):
        return Point(x=float(r['position_x']), y=float(r['position_y']), z=float(r['position_z']))

    # --ns 접두어: 한 화면에 작업물 두 개의 경로를 같이 그릴 때 마커 이름이 겹치지 않게 (예: --ns fender_)
    nsp = sys.argv[sys.argv.index('--ns') + 1] if '--ns' in sys.argv else ''

    def marker(mid, typ, ns):
        m = Marker()
        m.header.frame_id = frame
        m.ns, m.id, m.type, m.action = nsp + ns, mid, typ, Marker.ADD
        m.pose.orientation.w = 1.0
        return m

    GREEN = ColorRGBA(r=0.1, g=0.9, b=0.2, a=1.0)
    GRAY = ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.8)

    line = marker(0, Marker.LINE_LIST, 'path')
    line.scale.x = 0.006
    for a, b in zip(rows, rows[1:]):
        if a['row_index'] != b['row_index']:
            continue
        c = GREEN if a['paint'] == '1' and b['paint'] == '1' else GRAY
        line.points += [P(a), P(b)]
        line.colors += [c, c]

    link = marker(1, Marker.LINE_LIST, 'row_link')      # 패스 끝 -> 다음 패스 시작
    link.scale.x = 0.003
    link.color = ColorRGBA(r=0.3, g=0.5, b=1.0, a=0.6)
    traj = sys.argv[sys.argv.index('--traj') + 1] if '--traj' in sys.argv else ''
    if traj and os.path.exists(traj):                    # 계획 궤적이 있으면: 줄 사이를 실제로 지나가는 길 (직선이 범퍼에 막히는 곳은 곡선)
        z = np.load(traj)
        tcp, kind, rw = z['tcp'], z['kind'], z['row']
        link.header.frame_id = 'world_base'              # 궤적의 건 끝 위치는 월드 좌표
        link.scale.x = 0.005
        tg = np.flatnonzero(kind == 'target')
        bad = z['bad_edge'] if 'bad_edge' in z.files else np.zeros(len(tcp), bool)
        BLUE, RED = ColorRGBA(r=0.3, g=0.5, b=1.0, a=0.9), ColorRGBA(r=1.0, g=0.1, b=0.1, a=1.0)
        for a, b in zip(tg, tg[1:]):
            if rw[a] != rw[b]:
                for i in range(a, b):                        # 충돌 인증을 통과하지 못한 간선은 빨간색
                    u, v = tcp[i], tcp[i + 1]
                    link.points += [Point(x=float(u[0]), y=float(u[1]), z=float(u[2])), Point(x=float(v[0]), y=float(v[1]), z=float(v[2]))]
                    link.colors += [RED if bad[i] else BLUE] * 2
    else:
        from transition_curves import lane_curve
        groups = {}
        for row in rows:
            groups.setdefault(int(row['row_index']), []).append(row)
        for a, b in zip(rows, rows[1:]):
            if a['row_index'] != b['row_index']:
                row_id, next_id = int(a['row_index']), int(b['row_index'])
                if frame == 'bumper_link' and (row_id, next_id) in {(1, 2), (12, 13)}:
                    def xyz(row):
                        return [float(row['position_' + axis]) for axis in 'xyz']
                    curve = lane_curve([xyz(r) for r in groups[row_id]],
                                       [xyz(r) for r in groups[next_id]],
                                       1 if row_id == 1 else 2)
                    for p, q in zip(curve, curve[1:]):
                        link.points += [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                                        Point(x=float(q[0]), y=float(q[1]), z=float(q[2]))]
                else:
                    link.points += [P(a), P(b)]

    pts = marker(2, Marker.SPHERE_LIST, 'points')
    pts.scale.x = pts.scale.y = pts.scale.z = 0.012
    for r in rows:
        pts.points.append(P(r))
        pts.colors.append(GREEN if r['paint'] == '1' else GRAY)

    arrows = marker(3, Marker.LINE_LIST, 'spray_dir')    # 주황색 분사 방향 화살표는 표시하지 않는다 (열려 있는 RViz 에 남은 것도 지움)
    arrows.action = Marker.DELETE

    row_label = marker(4, Marker.TEXT_VIEW_FACING, 'row_label')
    if '--row' in sys.argv and '--no-row-label' not in sys.argv:
        anchor = rows[len(rows)//2]
        row_label.pose.position = P(anchor)
        row_label.pose.position.z += .20
        row_label.scale.z = .15
        row_label.color = ColorRGBA(r=1.0, g=0.85, b=0.05, a=1.0)
        row_label.text = f'ROW {selected_row}'
    else:
        row_label.action = Marker.DELETE

    rclpy.init()
    node = Node(f'paint_path_{sys.argv[1]}')
    qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
    pub = node.create_publisher(MarkerArray, 'paint_path', qos)
    extra = height_markers(sys.argv[sys.argv.index('--urdf') + 1], marker, frame) if '--urdf' in sys.argv else []
    if '--no-gantry-height' in sys.argv:                # 부품 높이만: 겐트리 높이·기둥 길이 표시는 뺀다
        extra = [m for m in extra if not m.ns.startswith(('gantry_height', 'column_length'))]
    if '--mark' in sys.argv:                            # 닿지 않는 점 (reach_points.py 결과) 빨간 큰 점
        import json
        bad = {p['csv_idx'] for p in json.load(open(sys.argv[sys.argv.index('--mark') + 1]))['points'] if p['status'] != 'ok'}
        red = marker(20, Marker.SPHERE_LIST, 'unreachable')
        red.scale.x = red.scale.y = red.scale.z = 0.03
        red.color = ColorRGBA(r=1.0, g=0.1, b=0.1, a=1.0)
        red.points = [P(r) for i, r in enumerate(rows) if i in bad]
        extra.append(red)
        node.get_logger().info(f'닿지 않는 점 {len(red.points)}개 빨간색 표시')
    else:                                               # 열려 있는 RViz 에 남은 빨간 점 지움
        red = marker(20, Marker.SPHERE_LIST, 'unreachable')
        red.action = Marker.DELETE
        extra.append(red)
    msg = MarkerArray(markers=[line, link, pts, arrows, row_label] + extra)

    def tick():
        now = node.get_clock().now().to_msg()
        for m in msg.markers:
            m.header.stamp = now
        pub.publish(msg)

    tick()
    node.create_timer(1.0, tick)
    n_rows = len({r['row_index'] for r in rows})
    n_paint = sum(r['paint'] == '1' for r in rows)
    node.get_logger().info(f'{folder}: {len(rows)}점 (CSV {n_all}점) / {n_rows}패스 / 분사 {n_paint}점 -> /paint_path ({frame})')
    rclpy.spin(node)


if __name__ == '__main__':
    main()
