"""追跡デバッグ画像だけを購読する ROS2 ノード（イベント橋とスレッド分離）."""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from rclpy.context import Context
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from shigure_core_msgs.msg import DebugImage

from shigure_api.config import TRACKING_DEBUG_IMAGE_TOPIC
from shigure_api.perf_confirm import ENABLED as _PERF_CONFIRM
from shigure_api.perf_confirm import perf_log as _perf_confirm_log
from shigure_api.tracking_debug import debug_image_msg_to_payload

_last_debug_arrival_mono: float = 0.0
_last_debug_arrival_lock = threading.Lock()


def mark_debug_image_arrival(now: Optional[float] = None) -> None:
    """直近のデバッグ画像到着時刻を記録する（確定 perf 用）."""
    global _last_debug_arrival_mono
    mono = time.monotonic() if now is None else now
    with _last_debug_arrival_lock:
        _last_debug_arrival_mono = mono


def debug_gap_since_last_ms() -> float:
    """前回のデバッグ画像到着からの経過ミリ秒."""
    with _last_debug_arrival_lock:
        if _last_debug_arrival_mono <= 0.0:
            return 0.0
        return (time.monotonic() - _last_debug_arrival_mono) * 1000.0


class TrackingDebugBridge(Node):
    """DebugImage のみ購読し WebSocket 層へ渡す."""

    def __init__(
        self,
        on_tracking_debug: Callable[[dict], None],
        *,
        context: Optional[Context] = None,
    ) -> None:
        # context を渡すとイベント橋と wait set を共有せず別スレッドで spin できる。
        super().__init__('shigure_api_tracking_debug_bridge', context=context)
        self._on_tracking_debug = on_tracking_debug
        self._perf_last_debug_mono: float = 0.0
        shigure_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(
            DebugImage,
            TRACKING_DEBUG_IMAGE_TOPIC,
            self._on_tracking_debug_image,
            shigure_qos,
        )
        self.get_logger().info(
            f'Tracking debug images on {TRACKING_DEBUG_IMAGE_TOPIC}'
        )

    def _emit_tracking_debug(self, payload: dict) -> None:
        try:
            self._on_tracking_debug(payload)
        except Exception as exc:
            self.get_logger().error(f'Failed to emit tracking debug image: {exc}')

    def _on_tracking_debug_image(self, msg: DebugImage) -> None:
        now = time.monotonic()
        mark_debug_image_arrival(now)
        if _PERF_CONFIRM and self._perf_last_debug_mono > 0.0:
            gap_ms = (now - self._perf_last_debug_mono) * 1000.0
            if gap_ms >= 50.0:
                _perf_confirm_log('debug_inter_arrival_ms', gap_ms=gap_ms)
        self._perf_last_debug_mono = now
        t0 = time.perf_counter()
        self._emit_tracking_debug(debug_image_msg_to_payload(msg))
        _perf_confirm_log(
            'debug_handler_ms',
            ms=(time.perf_counter() - t0) * 1000.0,
        )
