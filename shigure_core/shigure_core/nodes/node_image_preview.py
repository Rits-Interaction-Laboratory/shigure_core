import cv2
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from cv_bridge import CvBridge


class ImagePreviewNode(Node):
    """ImagePreviewするためのベースNode."""

    def __init__(self, node_name: str):
        super().__init__(node_name)

        # ros params
        is_debug_mode_descriptor = ParameterDescriptor(type=ParameterType.PARAMETER_BOOL,
                                                       description='If true, run debug mode.')
        self.declare_parameter('is_debug_mode', False, is_debug_mode_descriptor)
        self.is_debug_mode: bool = self.get_parameter("is_debug_mode").get_parameter_value().bool_value

        self.bridge = CvBridge()

        self.frame_count = 0

        # fps計測
        self.measurement_count = 10
        self.before_frame = 0
        self.fps = 0
        self.tm = cv2.TickMeter()
        self.tm.start()

        # show info
        self.get_logger().info('IsDebugMode : ' + str(self.is_debug_mode))

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

        cv2.putText(img, "frame = " + str(self.frame_count), (0, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0))
        cv2.putText(img, 'FPS: {:.2f}'.format(self.fps),
                    (0, 40), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0))

        return img
    
    def draw_outer_frame_line(self, src: np.ndarray, band_width=5, color=[255, 255, 255]):
        """
        外枠線を追加します.

        :param src: 基となる画像
        :param band_width: 枠線の太さ
        :param color: 色
        :return:
        """

        img = src

        height, width = img.shape[0], img.shape[1]


        # 1. bottom, 2. top, 3. left, 4. right line
        img[height - band_width:height, :, :] = color
        img[0: band_width, :, :] = color
        img[:, 0:band_width, :] = color
        img[:, width - band_width:width, :] = color

        return img