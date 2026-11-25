"""iPhone から API 経由で受け取った JPEG を ROS CompressedImage へ載せる."""

from __future__ import annotations

import threading
from typing import Optional

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage

from shigure_api.config import IPHONE_IMAGE_FRAME_ID, IPHONE_IMAGE_TOPIC

_JPEG_SOI = b'\xff\xd8'

_holder_lock = threading.Lock()
_holder: Optional['IphoneImageIngressNode'] = None


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
        self._frame_id = frame_id
        self._publish_lock = threading.Lock()
        self._frame_count = 0
        self.get_logger().info(
            f'iPhone ingress -> {topic} frame_id={frame_id}'
        )

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
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.format = 'jpeg'
        msg.data = jpeg
        with self._publish_lock:
            self._publisher.publish(msg)
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
