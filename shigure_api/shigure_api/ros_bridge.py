"""ROS2 subscriber that forwards recognition events to the FastAPI layer."""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from shigure_core_msgs.msg import (
    DebugImage,
    DictionaryUpdate,
    FaceRecognitionResult,
    FeatureInfo,
    PoseKeyPointsList,
    RecognitionHistory,
)

from shigure_api.config import (
    DICTIONARY_UPDATE_TOPIC,
    FACE_RECOGNITION_TOPIC,
    FEATURE_INFO_TOPIC,
    PCA_REBUILD_ON_NEW_USER,
    PEOPLE_DETECTION_TOPIC,
    PRESENCE_TICK_SEC,
    RECOGNITION_DEBOUNCE_SEC,
    RECOGNITION_HISTORY_TOPIC,
    RECOGNITION_SCORE_THRESHOLD,
    TRACKING_DEBUG_IMAGE_TOPIC,
)
from shigure_api.events import event_from_dictionary_update, event_from_face_recognition, recognition_scores_to_dict
from shigure_api.pca_plot import (
    PcaPlotStateBuilder,
    labeled_update_to_dict,
    segment_closed_to_dict,
    state_to_dict,
    unlabeled_update_to_dict,
)
from shigure_api.tracking_debug import debug_image_msg_to_payload


class RosEventBridge(Node):
    """Subscribe to Shigure topics and invoke callbacks for events and PCA plot."""

    def __init__(
        self,
        on_event: Callable,
        pca_builder: Optional[PcaPlotStateBuilder] = None,
        on_pca_payload: Optional[Callable[[dict], None]] = None,
        on_tracking_debug: Optional[Callable[[dict], None]] = None,
        on_recognition_scores: Optional[Callable[[dict], None]] = None,
    ):
        super().__init__('shigure_api_bridge')
        self._on_event = on_event
        self._pca_builder = pca_builder
        self._on_pca_payload = on_pca_payload
        self._on_tracking_debug = on_tracking_debug
        self._on_recognition_scores = on_recognition_scores
        self._recognition_last_sent: Dict[str, float] = {}
        # PCAプロットに反映済みの在室ユーザー集合（変化検知用）。
        self._pca_present_published: frozenset = frozenset()
        self._lock = threading.Lock()

        shigure_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(
            DictionaryUpdate,
            DICTIONARY_UPDATE_TOPIC,
            self._on_dictionary_update,
            10,
        )
        self.create_subscription(
            FaceRecognitionResult,
            FACE_RECOGNITION_TOPIC,
            self._on_face_recognition,
            shigure_qos,
        )
        if self._pca_builder is not None and self._on_pca_payload is not None:
            self.create_subscription(
                FeatureInfo,
                FEATURE_INFO_TOPIC,
                self._on_feature_info,
                shigure_qos,
            )
            # 骨格追跡: 顔が検出されなくても、確定ユーザーの在室を people_id で維持する。
            self.create_subscription(
                PoseKeyPointsList,
                PEOPLE_DETECTION_TOPIC,
                self._on_people_detection,
                shigure_qos,
            )
            self.get_logger().info(
                f'PCA plot listening on {FEATURE_INFO_TOPIC} and {PEOPLE_DETECTION_TOPIC}'
            )
            # 在室ユーザーの変化（登場/退室）を定期的に検知し、PCAプロットを再配信する。
            self.create_timer(PRESENCE_TICK_SEC, self._on_pca_presence_tick)
            self._publish_pca_state()

        # RecognitionHistory は PCAプロットと累積スコア配信の両方で使う。
        if self._pca_builder is not None or self._on_recognition_scores is not None:
            self.create_subscription(
                RecognitionHistory,
                RECOGNITION_HISTORY_TOPIC,
                self._on_recognition_history,
                10,
            )
            self.get_logger().info(
                f'Recognition history listening on {RECOGNITION_HISTORY_TOPIC}'
            )

        if self._on_tracking_debug is not None:
            self.create_subscription(
                DebugImage,
                TRACKING_DEBUG_IMAGE_TOPIC,
                self._on_tracking_debug_image,
                shigure_qos,
            )
            self.get_logger().info(
                f'Tracking debug images on {TRACKING_DEBUG_IMAGE_TOPIC}'
            )

        self.get_logger().info(
            f'Listening on {DICTIONARY_UPDATE_TOPIC} and {FACE_RECOGNITION_TOPIC}'
        )

    def _emit(self, event) -> None:
        try:
            self._on_event(event)
        except Exception as exc:
            self.get_logger().error(f'Failed to emit event: {exc}')

    def _emit_pca(self, payload: dict) -> None:
        if self._on_pca_payload is None:
            return
        try:
            self._on_pca_payload(payload)
        except Exception as exc:
            self.get_logger().error(f'Failed to emit PCA payload: {exc}')

    def _emit_tracking_debug(self, payload: dict) -> None:
        if self._on_tracking_debug is None:
            return
        try:
            self._on_tracking_debug(payload)
        except Exception as exc:
            self.get_logger().error(f'Failed to emit tracking debug image: {exc}')

    def emit_recognition_scores(self, payload: dict) -> None:
        if self._on_recognition_scores is None:
            return
        try:
            self._on_recognition_scores(payload)
        except Exception as exc:
            self.get_logger().error(f'Failed to emit recognition scores: {exc}')

    def _on_tracking_debug_image(self, msg: DebugImage) -> None:
        self._emit_tracking_debug(debug_image_msg_to_payload(msg))

    def _publish_pca_state(self) -> None:
        if self._pca_builder is None:
            return
        self._pca_present_published = self._pca_builder.presence_signature()
        self._emit_pca(state_to_dict(self._pca_builder.build_state()))

    def _on_pca_presence_tick(self) -> None:
        """在室変化の再配信に加え、確定後 pending 点の増分ドリップも行う。"""
        if self._pca_builder is None:
            return
        if self._pca_builder.tick_labeled_drip():
            self._publish_labeled_update()
        if self._pca_builder.presence_signature() != self._pca_present_published:
            self._publish_pca_state()

    def _publish_unlabeled_update(self) -> None:
        if self._pca_builder is None:
            return
        self._emit_pca(unlabeled_update_to_dict(self._pca_builder.build_unlabeled_update()))

    def _publish_labeled_update(self) -> None:
        if self._pca_builder is None:
            return
        self._emit_pca(labeled_update_to_dict(self._pca_builder.build_labeled_update()))

    def _on_dictionary_update(self, msg: DictionaryUpdate) -> None:
        event = event_from_dictionary_update(msg.people_id, msg.name)
        if event is not None:
            self.get_logger().info(f'Dictionary update: {event.type} user_id={event.user_id}')
            self._emit(event)

        if self._pca_builder is not None:
            closed = self._pca_builder.on_dictionary_update(msg.people_id, msg.name)
            if closed is not None:
                self._emit_pca(segment_closed_to_dict(closed))

            if event is not None and event.type == 'user_registered':
                if PCA_REBUILD_ON_NEW_USER:
                    rebuilt = self._pca_builder.rebuild_pca_from_disk(
                        registered_user_id=event.user_id
                    )
                    if rebuilt:
                        self.get_logger().info(
                            f'PCA model rebuilt after new user: {event.user_id}'
                        )
                # 確定瞬間は pending を全部色付けしてから配信する。
                self._pca_builder.flush_pending_labeled(event.user_id)
                self._publish_pca_state()
                self._publish_labeled_update()
            elif event is not None and event.type == 'user_confirmed':
                # 確定瞬間: 溜まっていた点を一括で色付け。以降の新点は増分で追加。
                self._pca_builder.flush_pending_labeled(event.user_id)
                self._publish_pca_state()
                self._publish_labeled_update()

    def _on_face_recognition(self, msg: FaceRecognitionResult) -> None:
        now = time.monotonic()
        for face in msg.faces:
            if face.score < RECOGNITION_SCORE_THRESHOLD:
                continue
            event = event_from_face_recognition(face.id, float(face.score))
            if event is None:
                continue
            key = event.user_id
            # PCAプロットの在室判定用に、認識フレームごとに在室をマークする。
            if self._pca_builder is not None:
                self._pca_builder.mark_present(key)
            # フロントへの認識通知は従来どおり30秒デバウンスで間引く。
            with self._lock:
                last = self._recognition_last_sent.get(key, 0.0)
                if now - last < RECOGNITION_DEBOUNCE_SEC:
                    continue
                self._recognition_last_sent[key] = now
            self._emit(event)

    def _on_feature_info(self, msg: FeatureInfo) -> None:
        if self._pca_builder is None:
            return
        push_unlabeled, push_labeled = self._pca_builder.on_feature_info(
            int(msg.feature_num),
            list(msg.feature),
        )
        if push_unlabeled:
            self._publish_unlabeled_update()
        if push_labeled:
            self._publish_labeled_update()

    def _on_recognition_history(self, msg: RecognitionHistory) -> None:
        # 今映っている people_id ごとの候補累積スコアをフロントへ配信する。
        self.emit_recognition_scores(recognition_scores_to_dict(msg))
        if self._pca_builder is None:
            return
        # 表示名確定（累積>=FACE_NAME_SCORE_THRESHOLD）と同じタイミングで色付けする。
        newly_colored = self._pca_builder.on_recognition_history(msg)
        for user_id in newly_colored:
            self._pca_builder.flush_pending_labeled(user_id)
            self._publish_pca_state()
            self._publish_labeled_update()
        # 既に色付け済みユーザーの新点を増分反映する。
        if self._pca_builder.tick_labeled_drip():
            self._publish_labeled_update()

    def _on_people_detection(self, msg: PoseKeyPointsList) -> None:
        """骨格追跡フレームごとに確定ユーザーの在室を更新する（顔検出不要）。"""
        if self._pca_builder is None:
            return
        self._pca_builder.on_people_tracking(msg)

    def clear_debounce(self, user_id: Optional[str] = None) -> None:
        with self._lock:
            if user_id is None:
                self._recognition_last_sent.clear()
            else:
                self._recognition_last_sent.pop(user_id, None)


def run_ros_bridge(
    on_event: Callable,
    stop_event: threading.Event,
    *,
    pca_builder: Optional[PcaPlotStateBuilder] = None,
    on_pca_payload: Optional[Callable[[dict], None]] = None,
    on_tracking_debug: Optional[Callable[[dict], None]] = None,
    on_recognition_scores: Optional[Callable[[dict], None]] = None,
) -> None:
    rclpy.init()
    node = RosEventBridge(
        on_event,
        pca_builder=pca_builder,
        on_pca_payload=on_pca_payload,
        on_tracking_debug=on_tracking_debug,
        on_recognition_scores=on_recognition_scores,
    )
    try:
        while rclpy.ok() and not stop_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()
