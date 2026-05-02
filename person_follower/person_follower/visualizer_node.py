#!/usr/bin/env python3
import sys
sys.path.insert(0, '/usr/lib/python3/dist-packages')
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

class VisualizerNode(Node):
    def __init__(self):
        super().__init__('visualizer_node')

        self.bridge       = CvBridge()
        self.latest_frame = None
        self.detected     = False
        self.angle        = 0.0
        self.distance     = 0.0

        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=5)
        status_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10)

        self.create_subscription(Image, '/camera/annotated',
                                 self.image_callback, cam_qos)
        self.create_subscription(Float32MultiArray, '/user_following_info',
                                 self.status_callback, status_qos)

        self.timer = self.create_timer(1.0/30, self.display_callback)
        self.get_logger().info('VisualizerNode 시작')

    def image_callback(self, msg):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def status_callback(self, msg):
        if len(msg.data) >= 3:
            self.detected  = bool(msg.data[0])
            self.angle     = msg.data[1]
            self.distance  = msg.data[2]

    def display_callback(self):
        if self.latest_frame is None:
            return

        frame = self.latest_frame.copy()
        h, w = frame.shape[:2]
        self.get_logger().info(f'frame size: {w}x{h}')
        cx    = w // 2

        # 중심선
        cv2.line(frame, (cx, 0), (cx, h), (0, 255, 0), 2)

        if self.detected:
            # 각도 → 화면 x 좌표
            angle_x = int(cx + (self.angle / 60.0) * cx)
            angle_x = max(0, min(w-1, angle_x))

            # 사람 방향 선
            cv2.line(frame, (cx, h//2), (angle_x, h//2), (0, 165, 255), 2)
            cv2.circle(frame, (angle_x, h//2), 8, (0, 165, 255), -1)

            # 정보 텍스트
            cv2.putText(frame, f'Angle: {self.angle:.1f} deg',
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            cv2.putText(frame, f'Dist:  {self.distance:.1f} m',
                        (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            cv2.putText(frame, 'DETECTED', (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        else:
            cv2.putText(frame, 'NO PERSON', (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

        cv2.imshow('Person Follower', frame)
        cv2.waitKey(1)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
