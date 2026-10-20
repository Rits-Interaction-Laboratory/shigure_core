import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from shigure_core_msgs.msg import TrackedObjectList

from shigure_core.db.db_util import DbConnectionError
from shigure_core.db.object_tracking_event_history_repository import ObjectTrackingEventHistoryRepository
from shigure_core.db.object_tracking_summary_repository import ObjectTrackingSummaryRepository
from shigure_core.nodes.common_model.marker_frame_transform import MarkerFrameTransformer

# DBに記録する対象action。
_RECORDED_ACTIONS = {'bring_in', 'obj_move', 'take_out', 'stay'}


class ObjectTrackingEventHistoryNode(Node):
    """object_trackingのobject_idごとのaction変化(bring_in/obj_move/take_out/stay)を、
    ARマーカー基準のワールド座標とタイムスタンプでMySQLに追記(INSERT)保存するノード。

    既存のobject_historyテーブルは物体ごとに1行を保持しUPDATEしていくため、
    過去のイベント時点の座標が失われる。本ノードはactionが変化した瞬間ごとに
    新規行をINSERTし、時系列の履歴として残す。同じactionが連続している間は
    (stayが継続している間も含め)記録しない。

    object_idごとに専用テーブル(object_tracking_event_history_<object_id>)を
    動的に作成し、そのobject_idのイベントのみを書き込む。

    加えて、object_tracking_summaryテーブルに object_id 1件につき1行、
    bring_in時刻/座標とtake_out時刻/座標をまとめて記録する(親テーブル)。
    take_outされるまではtake_out側の列はNULLのまま。
    """

    def __init__(self):
        super().__init__('object_tracking_event_history_node')

        shigure_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self._transformer = MarkerFrameTransformer(self)

        # object_idごとの直近action(記録対象外のstayも含めて変化検知に使う)
        self._last_action: dict = {}
        # 専用テーブルを作成済みのobject_id
        self._known_object_ids: set = set()

        self.create_subscription(
            TrackedObjectList, '/shigure/object_tracking', self.callback, shigure_qos)

        self.get_logger().info('ObjectTrackingEventHistoryNode started')

    def callback(self, msg: TrackedObjectList):
        now = datetime.datetime.now()

        for tracked_object in msg.tracked_object_list:
            object_id = tracked_object.object_id
            class_id = tracked_object.class_id
            action = tracked_object.action

            action_changed = self._last_action.get(object_id) != action
            self._last_action[object_id] = action

            if action_changed and action in _RECORDED_ACTIONS:
                position = self._transformer.to_marker_frame(tracked_object, msg.header.stamp)
                if position is None:
                    self.get_logger().warn(
                        'aruco_marker tf not available yet, skipping object_tracking_event_history record',
                        throttle_duration_sec=5.0)
                else:
                    try:
                        if object_id not in self._known_object_ids:
                            ObjectTrackingEventHistoryRepository.ensure_table(object_id)
                            self._known_object_ids.add(object_id)

                        x, y, z, width, height, depth = position
                        ObjectTrackingEventHistoryRepository.insert_event(
                            object_id, class_id, action, x, y, z, width, height, depth, now)

                        if action == 'bring_in':
                            ObjectTrackingSummaryRepository.insert_bring_in(
                                object_id, class_id, x, y, z, width, height, depth, now)
                        elif action == 'take_out':
                            ObjectTrackingSummaryRepository.update_take_out(
                                object_id, x, y, z, width, height, depth, now)
                    except DbConnectionError as e:
                        self.get_logger().error(
                            f'object_tracking_event_history record failed (DB unavailable), '
                            f'skipping this tick: {e}',
                            throttle_duration_sec=5.0)

            if action == 'take_out':
                # take_outが来たらそのobject_idのライフサイクルは終了
                self._last_action.pop(object_id, None)
                self._known_object_ids.discard(object_id)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectTrackingEventHistoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
