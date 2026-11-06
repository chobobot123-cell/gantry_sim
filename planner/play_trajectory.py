#!/usr/bin/env python3
"""계획된 범퍼 궤적(out/trajectory_bumper.npz)을 /joint_states 로 재생.

사용법: play_trajectory.py [--rate 50] [--speed 1.0] [--once] [--file PATH]
노드 사이는 관절공간 선형 보간 (계획/인증과 같은 보간).
"""
import argparse
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker

DIR = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', nargs='+', default=[str(DIR / 'out' / 'trajectory_bumper.npz')],
                    help='궤적 파일. 여러 개를 주면 차례로 이어서 재생한다 (앞 궤적의 끝 = 뒤 궤적의 시작 = 같은 홈 자세여야 함)')
    ap.add_argument('--gap', type=float, default=1.5, help='궤적 사이 홈에서 쉬는 시간 [s]')
    ap.add_argument('--rate', type=float, default=50.0)
    ap.add_argument('--speed', type=float, default=1.0, help='재생 배속')
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--hold', type=float, default=2.0, help='끝에서 멈춰 있는 시간 [s]')
    ap.add_argument('--start-delay', type=float, default=0.0, help='처음 자세에서 재생 시작까지 대기 [s]')
    a, _ = ap.parse_known_args()
    if a.start_delay < 0:
        ap.error('--start-delay must be nonnegative')
    z = np.load(a.file[0])
    X, t = z['X'], z['t']
    paint = z['paint']
    names = list(z['names']) + list(z['tool_names'])
    tool = list(z['tool_values'].astype(float))
    for f in a.file[1:]:                                 # 다음 작업물 궤적을 뒤에 붙인다
        z2 = np.load(f)
        if list(z2['names']) + list(z2['tool_names']) != names or not np.allclose(z2['tool_values'], tool):
            raise SystemExit(f'{f}: 관절 이름이나 건 상태가 앞 궤적과 다르다')
        if np.abs(z2['X'][0] - X[-1]).max() > 1e-6:
            raise SystemExit(f'{f}: 시작 자세가 앞 궤적의 끝 자세와 다르다')
        X, t = np.vstack([X, z2['X']]), np.r_[t, t[-1] + a.gap + z2['t']]
        paint = np.r_[paint, z2['paint']]
    rclpy.init()
    node = Node('bumper_trajectory_player')
    pub = node.create_publisher(JointState, 'joint_states', 10)
    status_pub = node.create_publisher(Marker, 'first_point_status', 10)
    t0 = node.get_clock().now()
    total = t[-1] / a.speed

    def tick():
        el = max(0.0, (node.get_clock().now() - t0).nanoseconds * 1e-9 - a.start_delay)
        if a.once:
            el = min(el, total)
        else:
            el = el % (total + a.hold)
        tau = min(el * a.speed, t[-1])
        k = int(np.clip(np.searchsorted(t, tau) - 1, 0, len(t) - 2))
        u = 0.0 if t[k + 1] == t[k] else (tau - t[k]) / (t[k + 1] - t[k])
        x = (1 - u) * X[k] + u * X[k + 1]
        m = JointState()
        m.header.stamp = node.get_clock().now().to_msg()
        m.name = names
        m.position = [float(v) for v in x] + tool
        pub.publish(m)
        marker = Marker()
        marker.header.frame_id, marker.header.stamp = 'gun_nozzle', m.header.stamp
        marker.ns, marker.id, marker.type, marker.action = 'spray', 0, Marker.TEXT_VIEW_FACING, Marker.ADD
        marker.pose.position.z, marker.pose.orientation.w, marker.scale.z = .18, 1., .12
        on = bool(paint[k] and paint[k+1])
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = (0.2,1.,0.2,1.) if on else (1.,.7,.1,1.)
        marker.text = 'SPRAY ON' if on else 'SPRAY OFF'
        status_pub.publish(marker)

    node.create_timer(1.0 / a.rate, tick)
    node.get_logger().info(f'{len(X)} nodes, cycle {t[-1]:.1f}s (x{a.speed}) from {", ".join(a.file)}')
    rclpy.spin(node)


if __name__ == '__main__':
    main()
