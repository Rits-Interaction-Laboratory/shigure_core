"""日本時間 3:00 に face_models を別リポジトリへ定期 push するノード.

認識コールバックとは独立したタイマーで動かし、git 処理で推論を止めない。
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo

import rclpy
from rcl_interfaces.msg import Parameter, ParameterDescriptor, ParameterType
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node

from shigure_core.util.push_face_models import (
    DEFAULT_REPO_DIR,
    push_face_models,
)

JST = ZoneInfo('Asia/Tokyo')
# 1分間隔。3時台のどこかで1回 push できればよい
TIMER_PERIOD_SEC = 60.0
PUSH_HOUR_JST = 3


class FaceModelsPushNode(Node):
    """JST 3:00 台に face_models スナップショットを push する."""

    def __init__(self):
        super().__init__('face_models_push_node')

        enabled_desc = ParameterDescriptor(
            type=ParameterType.PARAMETER_BOOL,
            description='false のとき定時 push を行わない。',
        )
        self.declare_parameter('enabled', True, enabled_desc)
        self.enabled: bool = (
            self.get_parameter('enabled').get_parameter_value().bool_value
        )

        repo_desc = ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING,
            description='バックアップ用 git 作業コピーのパス。',
        )
        self.declare_parameter('repo_dir', str(DEFAULT_REPO_DIR), repo_desc)
        self.repo_dir: str = (
            self.get_parameter('repo_dir').get_parameter_value().string_value
        )

        remote_desc = ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING,
            description='push 先 git remote 名。',
        )
        self.declare_parameter('remote', 'origin', remote_desc)
        self.remote: str = (
            self.get_parameter('remote').get_parameter_value().string_value
        )

        hour_desc = ParameterDescriptor(
            type=ParameterType.PARAMETER_INTEGER,
            description='push する日本時間の時（0-23）。既定 3。',
        )
        self.declare_parameter('push_hour_jst', PUSH_HOUR_JST, hour_desc)
        self.push_hour_jst: int = (
            self.get_parameter('push_hour_jst')
            .get_parameter_value()
            .integer_value
        )

        self._last_push_date: Optional[date] = None
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self.create_timer(TIMER_PERIOD_SEC, self._tick)

        self.get_logger().info(
            f'FaceModelsPush enabled={self.enabled} '
            f'repo_dir={self.repo_dir} hour_jst={self.push_hour_jst}'
        )

    def _on_set_parameters(
        self,
        params: List[Parameter],
    ) -> SetParametersResult:
        """実行中のパラメータ変更を反映する."""
        for param in params:
            if param.name == 'enabled':
                self.enabled = bool(param.value)
                self.get_logger().info(f'FaceModelsPush enabled={self.enabled}')
            elif param.name == 'repo_dir':
                self.repo_dir = str(param.value)
                self.get_logger().info(f'FaceModelsPush repo_dir={self.repo_dir}')
            elif param.name == 'remote':
                self.remote = str(param.value)
            elif param.name == 'push_hour_jst':
                self.push_hour_jst = int(param.value)
                self.get_logger().info(
                    f'FaceModelsPush hour_jst={self.push_hour_jst}'
                )
        return SetParametersResult(successful=True)

    def _tick(self) -> None:
        """1分ごとに JST 時刻を見て、対象時なら1日1回 push する."""
        if not self.enabled:
            return
        now = datetime.now(JST)
        if now.hour != self.push_hour_jst:
            return
        if self._last_push_date == now.date():
            return

        try:
            result = push_face_models(
                Path(self.repo_dir),
                remote=self.remote,
            )
            self.get_logger().info(f'[face_models_push] {result}')
            self._last_push_date = now.date()
        except Exception as exc:
            # 失敗日は再試行できるよう日付を進めない（3時台の間リトライ）
            self.get_logger().error(f'[face_models_push] failed: {exc}')


def main(args=None):
    """face_models 定時 push ノードを起動する."""
    rclpy.init(args=args)
    node = FaceModelsPushNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
