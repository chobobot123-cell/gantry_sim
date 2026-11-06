#!/usr/bin/env python3
"""홈 상한에서 첫 점까지 관절 보간 재생. 실제 실행용 궤적이 아니다."""
import json
import math
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker

ROOT = Path(__file__).resolve().parent
result = json.loads((ROOT / 'first_point.json').read_text(encoding='utf-8'))
names = ['gantry_x', 'gantry_y'] + [f'joint_{i}' for i in range(1, 7)] + ['gun_lift', 'gun_swing']
# smoothstep의 최대 기울기 1.5를 반영해 XY 합성 속도도 0.5 m/s 이하로 제한한다.
gantry_distance = math.hypot(*(result['goal_q'][i] - result['home_q'][i] for i in range(2)))
duration = max(15.0, 1.5 * gantry_distance / 0.5)

class Playback(Node):
    def __init__(self):
        super().__init__('first_point_playback')
        self.joints = self.create_publisher(JointState, '/joint_states', 10)
        self.markers = self.create_publisher(Marker, '/first_point_status', 10)
        self.start = self.get_clock().now()
        self.get_logger().info(f'갠트리 XY 합성 최대 속도 0.5 m/s, 이동 시간 {duration:.2f} s')
        self.create_timer(.05, self.tick)

    def tick(self):
        seconds = (self.get_clock().now() - self.start).nanoseconds / 1e9
        u = max(0., min(1., (seconds - 3.) / duration))
        u = u * u * (3. - 2. * u)
        q = [a + (b - a) * u for a, b in zip(result['home_q'], result['goal_q'])]
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name, msg.position = names, q
        self.joints.publish(msg)
        marker = Marker()
        marker.header.frame_id = 'gun_nozzle'
        marker.header.stamp = msg.header.stamp
        marker.ns, marker.id, marker.type, marker.action = 'spray', 0, Marker.TEXT_VIEW_FACING, Marker.ADD
        marker.pose.position.z = .18
        marker.pose.orientation.w = 1.
        marker.scale.z = .12
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1., .7, .1, 1.
        marker.text = 'SPRAY OFF'  # 첫 점의 paint=0, 접근 구간 전체 OFF
        self.markers.publish(marker)

rclpy.init()
node = Playback()
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
finally:
    node.destroy_node()
    rclpy.shutdown()
