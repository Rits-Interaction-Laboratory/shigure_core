"""ROS2 subscriber that forwards recognition events to the FastAPI layer."""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from shigure_core_msgs.msg import (
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
    FACE_MODELS_DIR,
    PEOPLE_DETECTION_TOPIC,
    PRESENCE_TICK_SEC,
    RECOGNITION_DEBOUNCE_SEC,
    RECOGNITION_HISTORY_TOPIC,
    RECOGNITION_SCORE_THRESHOLD,
)
from shigure_api.tracking_debug_bridge import TrackingDebugBridge, debug_gap_since_last_ms
from shigure_api.events import (
    entry_from_dictionary_update,
    event_from_dictionary_update,
    event_from_face_recognition,
    recognition_scores_to_dict,
    ros_stamp_to_jst_iso,
)
from shigure_api.pca_plot import (
    PcaPlotStateBuilder,
    segment_closed_to_dict,
    state_to_dict,
    unlabeled_update_to_dict,
)
from shigure_api.perf_confirm import ENABLED as _PERF_CONFIRM
from shigure_api.perf_confirm import perf_log as _perf_confirm_log
from shigure_api.pca_worker import PcaHeavyWorker
from shigure_core.util.face_models_dir import face_models_signature


class RosEventBridge(Node):
    """Subscribe to Shigure topics and invoke callbacks for events and PCA plot."""

    def __init__(
        self,
        on_event: Callable,
        pca_builder: Optional[PcaPlotStateBuilder] = None,
        on_pca_payload: Optional[Callable[[dict], None]] = None,
        on_recognition_scores: Optional[Callable[[dict], None]] = None,
        on_entry: Optional[Callable] = None,
    ):
        super().__init__('shigure_api_bridge')
        self._on_event = on_event
        self._pca_builder = pca_builder
        self._on_pca_payload = on_pca_payload
        self._on_recognition_scores = on_recognition_scores
        self._on_entry = on_entry
        self._recognition_last_sent: Dict[str, float] = {}
        # 同じ追跡IDへの再確定では入室通知を繰り返さない。
        self._entry_sent: Dict[str, str] = {}
        # PCAプロットに反映済みの在室ユーザー集合（変化検知用）。
        self._pca_present_published: frozenset = frozenset()
        self._face_models_signature = face_models_signature(FACE_MODELS_DIR)
        self._lock = threading.Lock()
        self._pca_worker: Optional[PcaHeavyWorker] = None
        if self._pca_builder is not None and self._on_pca_payload is not None:
            self._pca_worker = PcaHeavyWorker(
                pca_builder=self._pca_builder,
                publish_state=self._publish_pca_state,
                on_error=self._on_pca_worker_error,
            )
            self._face_models_signature = self._pca_worker.face_models_signature

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

        self.get_logger().info(
            f'Listening on {DICTIONARY_UPDATE_TOPIC} and {FACE_RECOGNITION_TOPIC}'
        )

    def _emit(self, event) -> None:
        try:
            self._on_event(event)
        except Exception as exc:
            self.get_logger().error(f'Failed to emit event: {exc}')

    def _emit_entry(self, entry) -> None:
        if self._on_entry is None:
            return
        try:
            self._on_entry(entry)
        except Exception as exc:
            self.get_logger().error(f'Failed to emit entry: {exc}')

    def _maybe_emit_entry(self, msg: DictionaryUpdate) -> None:
        """ユーザー確定（新規登録を含む）の初回だけ入室 JSON を送る."""
        name = msg.name
        people_id = msg.people_id
        if not name or name == 'none':
            return
        with self._lock:
            if self._entry_sent.get(people_id) == name:
                return
            self._entry_sent[people_id] = name
        entered_at = ros_stamp_to_jst_iso(
            int(msg.header.stamp.sec),
            int(msg.header.stamp.nanosec),
        )
        entry = entry_from_dictionary_update(people_id, name, entered_at)
        if entry is None:
            return
        self.get_logger().info(
            f'User entered: user_id={entry.user_id} display_name={entry.display_name}'
        )
        self._emit_entry(entry)

    def _emit_pca(self, payload: dict) -> None:
        if self._on_pca_payload is None:
            return
        try:
            if _PERF_CONFIRM:
                b64_len = 0
                for key in ('dictionary', 'unlabeled', 'trajectories_pre_confirm'):
                    chunk = payload.get(key)
                    if chunk is not None:
                        b64_len += len(str(chunk))
                _perf_confirm_log(
                    'pca_enqueue',
                    payload_type=str(payload.get('type', '')),
                    approx_payload_chars=b64_len or len(str(payload)),
                )
            self._on_pca_payload(payload)
        except Exception as exc:
            self.get_logger().error(f'Failed to emit PCA payload: {exc}')

    def emit_recognition_scores(self, payload: dict) -> None:
        if self._on_recognition_scores is None:
            return
        try:
            self._on_recognition_scores(payload)
        except Exception as exc:
            self.get_logger().error(f'Failed to emit recognition scores: {exc}')

    def _publish_pca_state(self) -> None:
        if self._pca_builder is None:
            return
        self._pca_present_published = self._pca_builder.presence_signature()
        t0 = time.perf_counter()
        state = self._pca_builder.build_state()
        build_ms = (time.perf_counter() - t0) * 1000.0
        if _PERF_CONFIRM:
            _perf_confirm_log('pca_build_state_ms', ms=build_ms)
        self._emit_pca(state_to_dict(state))

    def _enqueue_pca_heavy(self, kind: str, **kwargs) -> None:
        if self._pca_worker is None:
            return
        self._pca_worker.enqueue(kind, **kwargs)

    def _on_pca_worker_error(self, kind: str, exc: Exception) -> None:
        self.get_logger().error(f'PCA worker failed ({kind}): {exc}')

    def _on_pca_presence_tick(self) -> None:
        """在室（確定ユーザー＋未確定people）が変化していたら、退室者を除いた最新のPCAプロットを再配信する。"""
        if self._pca_builder is None:
            return
        if self._pca_builder.presence_signature() != self._pca_present_published:
            self._enqueue_pca_heavy('publish_state')

    def _sync_face_models_from_disk_if_changed(self) -> bool:
        """手動削除などで face_models が変わっていれば PCA 辞書を再読込する."""
        if self._pca_worker is None:
            return False
        if not self._pca_worker.check_disk_signature_changed():
            return False
        self.get_logger().info('[dict_sync] face_models changed; scheduling PCA reload')
        self._enqueue_pca_heavy('sync_reload_publish')
        return True

    def _publish_unlabeled_update(self) -> None:
        if self._pca_builder is None:
            return
        self._emit_pca(unlabeled_update_to_dict(self._pca_builder.build_unlabeled_update()))

    def _on_dictionary_update(self, msg: DictionaryUpdate) -> None:
        gap_before_ms = debug_gap_since_last_ms() if _PERF_CONFIRM else 0.0
        t0 = time.perf_counter()
        event = event_from_dictionary_update(msg.people_id, msg.name)
        if event is not None:
            self.get_logger().info(f'Dictionary update: {event.type} user_id={event.user_id}')
            self._emit(event)
        self._maybe_emit_entry(msg)

        if self._pca_builder is not None:
            closed = self._pca_builder.on_dictionary_update(msg.people_id, msg.name)
            if closed is not None:
                self._emit_pca(segment_closed_to_dict(closed))

            if event is not None and event.type == 'user_registered':
                self._enqueue_pca_heavy(
                    'rebuild_publish',
                    registered_user_id=event.user_id,
                )
            elif event is not None and event.type == 'user_confirmed':
                # ディスク辞書の再読込とフル state はワーカーへ（追跡デバッグ画像を塞がない）。
                self._enqueue_pca_heavy('reload_publish')
        _perf_confirm_log(
            'dictionary_update_handler_ms',
            ms=(time.perf_counter() - t0) * 1000.0,
            gap_since_last_debug_ms=gap_before_ms,
            name=str(msg.name),
        )

    def _on_face_recognition(self, msg: FaceRecognitionResult) -> None:
        self._sync_face_models_from_disk_if_changed()
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
        should_push = self._pca_builder.on_feature_info(
            int(msg.feature_num),
            list(msg.feature),
        )
        if should_push:
            self._publish_unlabeled_update()

    def _on_recognition_history(self, msg: RecognitionHistory) -> None:
        # 今映っている people_id ごとの候補累積スコアをフロントへ配信する。
        self.emit_recognition_scores(recognition_scores_to_dict(msg))
        if self._pca_builder is not None:
            # 確定済み複数人の新規点を色付けした場合はフル state を再配信する。
            if self._pca_builder.on_recognition_history(msg):
                self._enqueue_pca_heavy('publish_state')

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


def _spin_tracking_debug_bridge(
    on_tracking_debug: Callable[[dict], None],
    stop_event: threading.Event,
) -> None:
    """追跡デバッグ専用スレッド: 別 Context で DebugImage のみ spin する.

    イベント橋と同じ default Context 上で並行 ``spin_once`` すると
    ``IndexError: wait set index too big`` で落ちるため、Context を分離する。
    スレッド分離（base64 化がイベント橋を塞がない）はそのまま維持する。
    domain は ``ROS_DOMAIN_ID``（rclpy 既定）に従う。
    """
    from rclpy.context import Context

    context = Context()
    node: Optional[TrackingDebugBridge] = None
    try:
        rclpy.init(context=context)
        node = TrackingDebugBridge(on_tracking_debug, context=context)
        while context.ok() and not stop_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        if node is not None:
            node.destroy_node()
        if context.ok():
            rclpy.shutdown(context=context)


def run_ros_bridge(
    on_event: Callable,
    stop_event: threading.Event,
    *,
    pca_builder: Optional[PcaPlotStateBuilder] = None,
    on_pca_payload: Optional[Callable[[dict], None]] = None,
    on_tracking_debug: Optional[Callable[[dict], None]] = None,
    on_recognition_scores: Optional[Callable[[dict], None]] = None,
    on_entry: Optional[Callable] = None,
) -> None:
    """イベント用ノードと追跡デバッグ用ノードを別スレッドで spin する."""
    if not rclpy.ok():
        rclpy.init()
    event_node = RosEventBridge(
        on_event,
        pca_builder=pca_builder,
        on_pca_payload=on_pca_payload,
        on_recognition_scores=on_recognition_scores,
        on_entry=on_entry,
    )
    debug_thread: Optional[threading.Thread] = None
    if on_tracking_debug is not None:
        debug_thread = threading.Thread(
            target=_spin_tracking_debug_bridge,
            args=(on_tracking_debug, stop_event),
            name='shigure_tracking_debug_bridge',
            daemon=True,
        )
        debug_thread.start()

    try:
        while rclpy.ok() and not stop_event.is_set():
            rclpy.spin_once(event_node, timeout_sec=0.1)
    finally:
        # 先に stop して debug スレッドの Context を閉じる。
        stop_event.set()
        if debug_thread is not None:
            debug_thread.join(timeout=5.0)
        if event_node._pca_worker is not None:
            event_node._pca_worker.shutdown()
        event_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
