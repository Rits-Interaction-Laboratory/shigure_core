import cv2
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge


class NodeImagePreview(Node):
    """ImagePreviewするためのベースNode."""

    def __init__(self, node_name: str):
        super().__init__(node_name)

        self.bridge = CvBridge()

        self.frame_count = 0

        # fps計測
        self.measurement_count = 10
        self.before_frame = 0
        self.fps = 0
        self.tm = cv2.TickMeter()
        self.tm.start()

    def frame_count_up(self):
        """
        fpsを計算するためのフレーム計算をします.
        コールバックが呼ばれたときに呼び出す必要があります.

        :return:
        """
        self.frame_count += 1
        # fps計算
        if self.frame_count % self.measurement_count == 0:
            self.tm.stop()
            self.fps = (self.frame_count - self.before_frame) / self.tm.getTimeSec()
            self.before_frame = self.frame_count
            self.tm.reset()
            self.tm.start()

    def print_fps(self, src: np.ndarray) -> np.ndarray:
        """
        fpsを画像に印字します.

        :param src:
        :return:
        """
        img = src

        if src.ndim == 2:
            # 2次元 -> モノクロ画像
            img = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

        cv2.putText(img, "frame = " + str(self.frame_count), (0, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0))
        cv2.putText(img, 'FPS: {:.2f}'.format(self.fps),
                    (0, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0))

        return img
