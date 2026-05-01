#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import cv2
import numpy as np
from collections import deque
import threading
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from std_msgs.msg import Float32MultiArray

class PersonFollowerNode(Node):
    def __init__(self):
        super().__init__('person_follower_node')

        self.declare_parameter('camera_hfov', 120.0)
        self.declare_parameter('lidar_angle_margin', 5.0)
        self.declare_parameter('lidar_range_min', 0.26)
        self.declare_parameter('lidar_range_max', 16.0)
        self.declare_parameter('smooth_window', 5)
        self.declare_parameter('yolo_conf', 0.5)
        self.declare_parameter('yolo_model', 'yolov8n.pt')
        self.declare_parameter('image_width', 2304)
        self.declare_parameter('image_height', 1296)

        self.hfov         = self.get_parameter('camera_hfov').value
        self.angle_margin = self.get_parameter('lidar_angle_margin').value
        self.range_min    = self.get_parameter('lidar_range_min').value
        self.range_max    = self.get_parameter('lidar_range_max').value
        self.smooth_win   = self.get_parameter('smooth_window').value
        self.conf_thres   = self.get_parameter('yolo_conf').value
        self.img_w        = self.get_parameter('image_width').value

        self.model        = YOLO(self.get_parameter('yolo_model').value)
        self.angle_buf    = deque(maxlen=self.smooth_win)
        self.distance_buf = deque(maxlen=self.smooth_win)
        self.latest_scan  = None
        self.latest_frame = None
        self.frame_lock   = threading.Lock()
        self.bridge       = CvBridge()

        lidar_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10)
        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=5)

        self.create_subscription(LaserScan, '/scan', self.scan_callback, lidar_qos)
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, cam_qos)
        self.status_pub = self.create_publisher(Float32MultiArray, '/person_status', 10)

        self.infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self.infer_thread.start()
        self.get_logger().info('PersonFollowerNode 시작')

    def scan_callback(self, msg):
        self.latest_scan = msg

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        with self.frame_lock:
            self.latest_frame = (frame, msg.header)

    def _infer_loop(self):
        while rclpy.ok():
            with self.frame_lock:
                if self.latest_frame is None:
                    continue
                frame, header = self.latest_frame
                self.latest_frame = None

            results = self.model.predict(frame, classes=[0], conf=self.conf_thres, verbose=False)
            annotated = results[0].plot()
            cv2.imshow('YOLO Detection', annotated)
            cv2.waitKey(1)

            status = Float32MultiArray()
            if not results or len(results[0].boxes) == 0:
                status.data = [0.0, 0.0, 0.0]
                self.angle_buf.clear()
                self.distance_buf.clear()
                self.status_pub.publish(status)
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
            best  = boxes[np.argmax(areas)]
            x1, _, x2, _ = best

            cx        = (x1 + x2) / 2.0
            norm_x    = (cx - self.img_w / 2.0) / (self.img_w / 2.0)
            raw_angle = norm_x * (self.hfov / 2.0)
            raw_dist  = self._get_distance_from_scan(raw_angle)

            self.angle_buf.append(raw_angle)
            if raw_dist is not None:
                self.distance_buf.append(raw_dist)

            status.data = [
                1.0,
                round(float(np.mean(self.angle_buf)), 1),
                round(float(np.mean(self.distance_buf)), 1) if self.distance_buf else 0.0
            ]
            self.status_pub.publish(status)
            self.get_logger().info(f'angle={status.data[1]:.1f}° dist={status.data[2]:.1f}m')

    def _get_distance_from_scan(self, target_angle_deg):
        if self.latest_scan is None:
            return None

        scan      = self.latest_scan
        angle_min = np.degrees(scan.angle_min)
        angle_inc = np.degrees(scan.angle_increment)
        ranges    = np.array(scan.ranges)
        angles    = np.arange(len(ranges)) * angle_inc + angle_min

        lo, hi = target_angle_deg - self.angle_margin, target_angle_deg + self.angle_margin
        mask   = (angles >= lo) & (angles <= hi)
        if not np.any(mask):
            return None

        valid = ranges[mask]
        valid = valid[(valid >= self.range_min) & (valid <= self.range_max)]
        valid = valid[np.isfinite(valid)]
        if len(valid) == 0:
            return None

        valid_sorted = np.sort(valid)
        clusters = []
        current  = [valid_sorted[0]]

        for v in valid_sorted[1:]:
            if v - current[-1] < 0.3:
                current.append(v)
            else:
                clusters.append(current)
                current = [v]
        clusters.append(current)

        valid_clusters = [c for c in clusters if len(c) >= 2]
        if not valid_clusters:
            valid_clusters = clusters

        nearest_cluster = min(valid_clusters, key=lambda c: np.mean(c))
        return float(np.mean(nearest_cluster))

def main(args=None):
    rclpy.init(args=args)
    node = PersonFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
