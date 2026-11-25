"""iPhone から API 経由で受け取った JPEG を ROS CompressedImage へ載せる."""

from __future__ import annotations

import threading
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage

from shigure_api.config import (
    IPHONE_CAMERA_INFO_TOPIC,
    IPHONE_DEPTH_TOPIC,
    IPHONE_DUMMY_DEPTH_MM,
    IPHONE_IMAGE_FRAME_ID,
    IPHONE_IMAGE_TOPIC,
    IPHONE_MODE,
)
from shigure_core.util.compressed_depth_util import convert_cv2_to_compressed_depth_img

_JPEG_SOI = b'\xff\xd8'

_holder_lock = threading.Lock()
_holder: Optional['IphoneImageIngressNode'] = None


def _jpeg_size(jpeg: bytes) -> Optional[Tuple[int, int]]:
    """JPEG バイト列から (width, height) を得る。失敗時は None。"""
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    height, width = img.shape[:2]
    return width, height


def _build_camera_info(
    width: int,
    height: int,
    frame_id: str,
    stamp,
) -> CameraInfo:
    """people_tracking 向けの粗い pinhole CameraInfo。"""
    msg = CameraInfo()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = height
    msg.width = width
    fx = float(width)
    fy = float(width)
    cx = float(width) * 0.5
    cy = float(height) * 0.5
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    msg.distortion_model = 'plumb_bob'
    return msg


class IphoneImageIngressNode(Node):
    """iPhone カメラ JPEG を指定 ROS トピックへ publish するノード."""

    def __init__(
        self,
        topic: str,
        frame_id: str,
        *,
        context: Optional[Context] = None,
    ) -> None:
        super().__init__('shigure_iphone_image_ingress', context=context)
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._publisher = self.create_publisher(CompressedImage, topic, qos)
        self._depth_publisher = None
        self._camera_info_publisher = None
        if IPHONE_MODE:
            self._depth_publisher = self.create_publisher(
                CompressedImage, IPHONE_DEPTH_TOPIC, qos
            )
            self._camera_info_publisher = self.create_publisher(
                CameraInfo, IPHONE_CAMERA_INFO_TOPIC, qos
            )
        self._frame_id = frame_id
        self._bridge = CvBridge()
        self._publish_lock = threading.Lock()
        self._frame_count = 0
        self._last_size: Optional[Tuple[int, int]] = None
        self.get_logger().info(
            f'iPhone ingress -> {topic} frame_id={frame_id} '
            f'iphone_mode={IPHONE_MODE} depth_mm={IPHONE_DUMMY_DEPTH_MM}'
        )

    def _publish_dummy_depth_and_camera_info(self, stamp, width: int, height: int) -> None:
        if self._depth_publisher is None or self._camera_info_publisher is None:
            return
        depth_u16 = np.full((height, width), IPHONE_DUMMY_DEPTH_MM, dtype=np.uint16)
        depth_msg = convert_cv2_to_compressed_depth_img(depth_u16, self._bridge)
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = self._frame_id
        cam_msg = _build_camera_info(width, height, self._frame_id, stamp)
        self._depth_publisher.publish(depth_msg)
        self._camera_info_publisher.publish(cam_msg)

    def publish_jpeg(self, jpeg: bytes) -> None:
        """JPEG バイト列を CompressedImage として配信する."""
        if not jpeg:
            return
        if not jpeg.startswith(_JPEG_SOI):
            self.get_logger().warning(
                'iPhone ingress: payload is not JPEG (missing SOI)'
            )
            return
        msg = CompressedImage()
        stamp = self.get_clock().now().to_msg()
        msg.header.stamp = stamp
        msg.header.frame_id = self._frame_id
        msg.format = 'jpeg'
        msg.data = jpeg
        size = _jpeg_size(jpeg)
        if size is not None:
            self._last_size = size
        elif self._last_size is not None:
            size = self._last_size
        with self._publish_lock:
            self._publisher.publish(msg)
            if IPHONE_MODE and size is not None:
                self._publish_dummy_depth_and_camera_info(
                    stamp, size[0], size[1]
                )
        self._frame_count += 1


def iphone_image_ingress_ready() -> bool:
    """iPhone 用 ROS ingress が publish 可能か."""
    with _holder_lock:
        return _holder is not None


def publish_iphone_jpeg_from_api(jpeg: bytes) -> None:
    """FastAPI / WebSocket から受け取った JPEG を ROS へ流す."""
    with _holder_lock:
        node = _holder
    if node is None:
        raise RuntimeError('iPhone image ingress ROS node is not running')
    node.publish_jpeg(jpeg)


def run_iphone_image_ingress(stop_event: threading.Event) -> None:
    """専用 Context で iPhone ingress ノードを spin する."""
    global _holder
    context = Context()
    node: Optional[IphoneImageIngressNode] = None
    executor: Optional[SingleThreadedExecutor] = None
    try:
        rclpy.init(context=context)
        node = IphoneImageIngressNode(
            IPHONE_IMAGE_TOPIC,
            IPHONE_IMAGE_FRAME_ID,
            context=context,
        )
        with _holder_lock:
            _holder = node
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        while context.ok() and not stop_event.is_set():
            executor.spin_once(timeout_sec=0.05)
    finally:
        with _holder_lock:
            _holder = None
        if executor is not None:
            if node is not None:
                executor.remove_node(node)
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        if context.ok():
            rclpy.shutdown(context=context)