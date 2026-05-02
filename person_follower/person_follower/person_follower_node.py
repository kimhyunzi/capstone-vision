#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/jetson/.local/lib/python3.10/site-packages')
import time
import threading

import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import cv2
import numpy as np
from collections import deque

from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from std_msgs.msg import Float32MultiArray


class PersonFollowerNode(Node):
    def __init__(self):
        super().__init__('person_follower_node')

        # ── 파라미터 선언 ──────────────────────────────────────────
        self.declare_parameter('camera_hfov',                   120.0)
        self.declare_parameter('lidar_range_min',                 0.26)  # YDLiDAR G4
        self.declare_parameter('lidar_range_max',                16.0)   # YDLiDAR G4
        self.declare_parameter('smooth_window',                     5)
        self.declare_parameter('yolo_conf',                        0.5)
        self.declare_parameter('yolo_model',          'yolov8n.engine')
        self.declare_parameter('show_display',                   False)
        self.declare_parameter('camera_to_lidar_yaw_offset_deg',  0.0)
        self.declare_parameter('max_scan_age_sec',                 0.20)
        self.declare_parameter('target_switch_penalty_deg',       12.0)
        self.declare_parameter('min_lidar_angle_margin',           2.0)
        self.declare_parameter('max_lidar_angle_margin',          10.0)
        self.declare_parameter('lost_count_threshold',            10)
        self.declare_parameter('image_width', 2304)
        self.declare_parameter('image_height', 1296)

        self.hfov           = self.get_parameter('camera_hfov').value
        self.range_min      = self.get_parameter('lidar_range_min').value
        self.range_max      = self.get_parameter('lidar_range_max').value
        self.smooth_win     = self.get_parameter('smooth_window').value
        self.conf_thres     = self.get_parameter('yolo_conf').value
        self.show_display   = self.get_parameter('show_display').value
        self.yaw_offset     = self.get_parameter('camera_to_lidar_yaw_offset_deg').value
        self.max_scan_age   = self.get_parameter('max_scan_age_sec').value
        self.switch_penalty = self.get_parameter('target_switch_penalty_deg').value
        self.min_margin     = self.get_parameter('min_lidar_angle_margin').value
        self.max_margin     = self.get_parameter('max_lidar_angle_margin').value
        self.lost_threshold = self.get_parameter('lost_count_threshold').value
        self.annotated_pub = self.create_publisher(Image, '/camera/annotated', 5)

        self.img_w = self.get_parameter('image_width').value
        self.get_logger().info(f'image_width: {self.img_w}')
        # ── TRT 엔진 로드 (task 명시 → WARNING 제거) ───────────────
       #  model_path = self.get_parameter('yolo_model').value
        model_path = '/home/jetson/yolov8n.engine'
        self.model = YOLO(model_path, task='detect')
        self.get_logger().info(f'모델 로드: {model_path}')

        # ── 상태 변수 ──────────────────────────────────────────────
        self.angle_buf         = deque(maxlen=self.smooth_win)
        self.distance_buf      = deque(maxlen=self.smooth_win)
        self.last_target_angle = None
        self.lost_count        = 0
        self.latest_scan       = None
        self.latest_frame      = None
        self.frame_lock        = threading.Lock()
        self.scan_lock         = threading.Lock()
        self.bridge            = CvBridge()

        # ── QoS ───────────────────────────────────────────────────
        lidar_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10)
        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=5)

        # ── 구독 / 퍼블리셔 ────────────────────────────────────────
        self.create_subscription(
            LaserScan, '/scan', self.scan_callback, lidar_qos)
        self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, cam_qos)

        # [detected(1/0), angle_deg, distance_m]
        # detected=0 이면 수신 측에서 반드시 무시할 것
        self.status_pub = self.create_publisher(
            Float32MultiArray, '/user_following_info', 10)

        # ── 추론 스레드 ────────────────────────────────────────────
        self.infer_thread = threading.Thread(
            target=self._infer_loop, daemon=True)
        self.infer_thread.start()
        self.get_logger().info('PersonFollowerNode 시작')

    # ── 콜백 ──────────────────────────────────────────────────────
    def scan_callback(self, msg):
        with self.scan_lock:
            self.latest_scan = msg

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        with self.frame_lock:
            self.latest_frame = (frame, msg.header)

    # ── 타겟 bbox 선택 ────────────────────────────────────────────
    def _select_target_box(self, boxes, img_w):
        """
        첫 감지: 가장 큰 bbox
        이후: 크기 점수 - 각도 변화 페널티 → 타겟 튐 방지
        """
        centers = (boxes[:, 0] + boxes[:, 2]) / 2.0
        areas   = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        angles  = ((centers - img_w / 2.0) / (img_w / 2.0)) * (self.hfov / 2.0)

        if self.last_target_angle is None:
            idx = int(np.argmax(areas))
        else:
            area_score = areas / max(float(np.max(areas)), 1.0)
            angle_cost = np.abs(angles - self.last_target_angle) / self.switch_penalty
            idx = int(np.argmax(area_score - angle_cost))

        return boxes[idx], float(angles[idx])

    # ── 추론 루프 ──────────────────────────────────────────────────
    def _infer_loop(self):
        while rclpy.ok():
            with self.frame_lock:
                data = self.latest_frame
                if data is not None:
                    self.latest_frame = None

            if data is None:
                time.sleep(0.005)
                continue

            frame, header = data
            img_w = frame.shape[1]  # 실제 프레임 폭 사용

            results = self.model.predict(
                frame, classes=[0], conf=self.conf_thres, verbose=False)
           
            # results 나온 직후
            annotated = results[0].plot()
            ann_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            ann_msg.header.stamp = self.get_clock().now().to_msg()
            self.annotated_pub.publish(ann_msg)
            
            if self.show_display:
                cv2.imshow('YOLO Detection', results[0].plot())
                cv2.waitKey(1)

            status = Float32MultiArray()

            # ── 미감지 처리 ────────────────────────────────────────
            if not results or len(results[0].boxes) == 0:
                self.lost_count += 1
                if self.lost_count > self.lost_threshold:
                    # lost_threshold 프레임 초과 시에만 타겟 초기화
                    # 잠깐 가려진 경우 버퍼 유지 → 모터 제어 안정
                    self.last_target_angle = None
                    self.angle_buf.clear()
                    self.distance_buf.clear()
                status.data = [0.0, 0.0, 0.0]
                self.status_pub.publish(status)
                continue

            self.lost_count = 0

            # ── 타겟 선택 ──────────────────────────────────────────
            boxes = results[0].boxes.xyxy.cpu().numpy()
            best, raw_camera_angle = self._select_target_box(boxes, img_w)
            x1, _, x2, _ = best

            # bbox 폭 → 동적 LiDAR window
            bbox_angle_width = ((x2 - x1) / img_w) * self.hfov
            dynamic_margin   = float(np.clip(
                bbox_angle_width / 2.0,
                self.min_margin,
                self.max_margin
            ))

            # 카메라 각도 → LiDAR 좌표계 보정
            lidar_angle = raw_camera_angle + self.yaw_offset

            raw_dist = self._get_distance_from_scan(lidar_angle, dynamic_margin)

            self.last_target_angle = raw_camera_angle
            self.angle_buf.append(raw_camera_angle)
            if raw_dist is not None:
                self.distance_buf.append(raw_dist)

            status.data = [
                1.0,
                round(float(np.mean(self.angle_buf)), 1),
                round(float(np.mean(self.distance_buf)), 1)
                if self.distance_buf else 0.0
            ]
            self.status_pub.publish(status)
            self.get_logger().info(
                f'angle={status.data[1]:.1f}° '
                f'dist={status.data[2]:.1f}m '
                f'margin={dynamic_margin:.1f}°')

    # ── YDLiDAR G4 거리 추출 ──────────────────────────────────────
    def _get_distance_from_scan(self, target_angle_deg, angle_margin_deg):
        with self.scan_lock:
            scan = self.latest_scan

        if scan is None:
            return None

        # scan 유효 시간 검사
        now       = self.get_clock().now()
        scan_time = rclpy.time.Time.from_msg(scan.header.stamp)
        age_sec   = (now - scan_time).nanoseconds * 1e-9
        if age_sec > self.max_scan_age:
            self.get_logger().warn(
                f'scan too old: {age_sec:.3f}s',
                throttle_duration_sec=1.0)
            return None

        angle_min = np.degrees(scan.angle_min)
        angle_inc = np.degrees(scan.angle_increment)
        ranges    = np.asarray(scan.ranges, dtype=np.float32)
        angles    = np.arange(len(ranges), dtype=np.float32) * angle_inc + angle_min

        # ±180° wrap-around 대응
        wrapped = ((angles - target_angle_deg + 180.0) % 360.0) - 180.0
        mask    = np.abs(wrapped) <= angle_margin_deg

        if not np.any(mask):
            return None

        rmin  = max(float(scan.range_min), self.range_min)
        rmax  = min(float(scan.range_max), self.range_max)
        valid = ranges[mask]
        valid = valid[np.isfinite(valid)]
        valid = valid[(valid >= rmin) & (valid <= rmax)]

        if len(valid) == 0:
            return None

        # 25th percentile — 평균보다 노이즈에 강함
        return float(np.percentile(valid, 25))


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
