#!/usr/bin/env python3
"""계산된 RViz 궤적을 재생한다. 시각화 전용이며 실장비 명령이 아니다."""
import json
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker

ROOT = Path(__file__).resolve().parent
report = json.loads((ROOT/'follow_report.json').read_text())
z = np.load(ROOT/'follow_trajectory.npz')
q, times, paint = z['q'], z['t'], z['paint']
names = ['gantry_x','gantry_y'] + [f'joint_{i}' for i in range(1,7)] + ['gun_lift','gun_swing']

class Playback(Node):
    def __init__(self):
        super().__init__('follow_playback')
        self.pub = self.create_publisher(JointState, '/joint_states', 10)
        self.marker = self.create_publisher(Marker, '/first_point_status', 10)
        self.start = self.get_clock().now()
        self.create_timer(.02, self.tick)
        self.get_logger().info(f"{report['status']}: CSV {report['reached_csv_points']}/{report['total_csv_points']}, {times[-1]:.1f} s")

    def tick(self):
        elapsed = (self.get_clock().now()-self.start).nanoseconds / 1e9
        t = min(max(0., elapsed-3.), float(times[-1]))
        k = int(np.clip(np.searchsorted(times, t, side='right')-1, 0, len(times)-2))
        u = np.clip((t-times[k])/(times[k+1]-times[k]), 0., 1.)
        state = (1-u)*q[k]+u*q[k+1]
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name, msg.position = names, state.tolist()
        self.pub.publish(msg)
        on = bool(paint[k] and paint[k+1])
        m = Marker()
        m.header.frame_id, m.header.stamp = 'gun_nozzle', msg.header.stamp
        m.ns, m.id, m.type, m.action = 'spray', 0, Marker.TEXT_VIEW_FACING, Marker.ADD
        m.pose.position.z, m.pose.orientation.w, m.scale.z = .18, 1., .12
        m.color.r, m.color.g, m.color.b, m.color.a = (0.2,1.,0.2,1.) if on else (1.,.7,.1,1.)
        m.text = 'SPRAY ON' if on else 'SPRAY OFF'
        self.marker.publish(m)

rclpy.init()
node = Playback()
try:
    rclpy.spin(node)
except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
    pass
finally:
    node.destroy_node()
    if rclpy.ok(): rclpy.shutdown()
