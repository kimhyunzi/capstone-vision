#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraPublisherNode(Node):
    def __init__(self):
        super().__init__('camera_publisher_node')
        self.declare_parameter('width', 1920)
        self.declare_parameter('height', 1080)
        self.declare_parameter('framerate', 30)
        self.declare_parameter('sensor_id', 0)

        w   = self.get_parameter('width').value
        h   = self.get_parameter('height').value
        fps = self.get_parameter('framerate').value
        sid = self.get_parameter('sensor_id').value

        pipeline = (
            f"nvarguscamerasrc sensor-id={sid} ! "
            f"video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1 ! "
            f"queue ! nvvidconv ! "
            f"video/x-raw,format=BGRx ! videoconvert ! "
            f"video/x-raw,format=BGR ! appsink"
        )

        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.get_logger().error('카메라 열기 실패')
            raise RuntimeError('Camera open failed')

        self.bridge = CvBridge()
        self.pub    = self.create_publisher(Image, '/camera/image_raw', 5)
        self.timer  = self.create_timer(1.0/fps, self.timer_callback)
        self.get_logger().info(f'IMX708 시작: {w}x{h} @ {fps}fps')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('프레임 읽기 실패')
            return
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'
        self.pub.publish(msg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
