import tf2_geometry_msgs
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from tf2_ros import LookupException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from shigure_core_msgs.msg import TrackedObject


class MarkerFrameTransformer:
    """object_trackingのcollider(カメラ相対mm単位)をARマーカー基準のワールド座標(m)に変換する."""

    def __init__(self, node: Node):
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, node)

    def to_marker_frame(self, tracked_object: TrackedObject, stamp):
        """物体colliderの中心点をARマーカー基準の座標(m)に変換する。マーカー未検出ならNone。

        collider は object_tracking 側でmm単位のカメラ相対最小角+寸法として計算されているため、
        中心点をmに変換してからtfでマーカー座標系に変換する。
        """
        try:
            trans = self._tf_buffer.lookup_transform('aruco_marker', 'camera', rclpy.time.Time())
        except LookupException:
            return None

        collider = tracked_object.collider
        point = PointStamped()
        point.header.frame_id = 'camera'
        point.header.stamp = stamp
        point.point.x = (collider.x + collider.width / 2.0) / 1000.0
        point.point.y = (collider.y + collider.height / 2.0) / 1000.0
        point.point.z = (collider.z + collider.depth / 2.0) / 1000.0

        transformed = tf2_geometry_msgs.do_transform_point(point, trans)
        return (transformed.point.x, transformed.point.y, transformed.point.z,
               collider.width / 1000.0, collider.height / 1000.0, collider.depth / 1000.0)
