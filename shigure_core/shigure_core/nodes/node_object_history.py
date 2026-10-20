import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from shigure_core_msgs.msg import TrackedObjectList

from shigure_core.db.db_util import DbConnectionError
from shigure_core.db.object_history_repository import ObjectHistoryRepository
from shigure_core.nodes.common_model.marker_frame_transform import MarkerFrameTransformer

_STAY_UPDATE_INTERVAL_SEC = 5.0


class ObjectHistoryNode(Node):
    """object_trackingの結果をARマーカー基準のワールド座標でMySQLに永続化するノード."""

    def __init__(self):
        super().__init__('object_history_node')

        shigure_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self._transformer = MarkerFrameTransformer(self)

        # 現在オープン中(take_out未確定)の物体idと、直近stay更新時刻
        self._open_object_ids: set = set()
        self._last_stay_update: dict = {}

        self.create_subscription(
            TrackedObjectList, '/shigure/object_tracking', self.callback, shigure_qos)

        self.get_logger().info('ObjectHistoryNode started')

    def callback(self, msg: TrackedObjectList):
        now = datetime.datetime.now()

        for tracked_object in msg.tracked_object_list:
            position = self._transformer.to_marker_frame(tracked_object, msg.header.stamp)
            if position is None:
                self.get_logger().warn(
                    'aruco_marker tf not available yet, skipping object_history update',
                    throttle_duration_sec=5.0)
                continue

            x, y, z, width, height, depth = position
            object_id = tracked_object.object_id
            class_id = tracked_object.class_id
            action = tracked_object.action

            try:
                if action == 'bring_in':
                    if object_id not in self._open_object_ids and not ObjectHistoryRepository.has_open_row(object_id):
                        ObjectHistoryRepository.insert_bring_in(
                            object_id, class_id, x, y, z, width, height, depth, now)
                    self._open_object_ids.add(object_id)
                    self._last_stay_update[object_id] = now
                elif action == 'take_out':
                    ObjectHistoryRepository.close_take_out(object_id, now)
                    self._open_object_ids.discard(object_id)
                    self._last_stay_update.pop(object_id, None)
                elif action == 'obj_move':
                    ObjectHistoryRepository.update_stay(object_id, x, y, z, width, height, depth, now)
                    self._last_stay_update[object_id] = now
                elif action == 'stay':
                    last_update = self._last_stay_update.get(object_id)
                    if last_update is None or (now - last_update).total_seconds() >= _STAY_UPDATE_INTERVAL_SEC:
                        ObjectHistoryRepository.update_stay(object_id, x, y, z, width, height, depth, now)
                        self._last_stay_update[object_id] = now
            except DbConnectionError as e:
                self.get_logger().error(
                    f'object_history update failed (DB unavailable), skipping this tick: {e}',
                    throttle_duration_sec=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectHistoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
