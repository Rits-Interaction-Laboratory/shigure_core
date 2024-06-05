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
     
    def overlay_image(self, overlapping_img: np.ndarray, underlying_img: np.ndarray, shift, resize_scale, is_frame_line: bool=True):
        """
        2枚の画像を重ね合わせます.
        :param overlapping_img: 重ね合わせる画像
        :param underlying_img: 下地画像
        :param shift: 左上を原点としたときの移動量(x, y)
        :param resize_scale: 重ね合わせる画像の拡大・縮小倍率(width, height)
        :param is_frame_line: 重ね合わせる画像に外枠線を付与するか
        """

        shift_x, shift_y = shift
        resize_width, resize_height = resize_scale

        # 重ね合わせる画像のリサイズ
        overlapping_img = cv2.resize(overlapping_img, None, None, fx=resize_width, fy=resize_height)
        if is_frame_line:
            overlapping_img = self.draw_outer_frame_line(overlapping_img, band_width=4, color=(0, 0, 0))

        # 重ね合わせる画像のパラメータ
        overlapping_height, overlapping_width = overlapping_img.shape[:2]
        overlapping_x_min, overlapping_x_max = 0, overlapping_width
        overlapping_y_min, overlapping_y_max = 0, overlapping_height

        # 下地画像のパラメータ
        underlying_height, underlying_width = underlying_img.shape[:2]
        underlying_x_min, underlying_x_max = shift_x, shift_x + overlapping_width 
        underlying_y_min, underlying_y_max = shift_y, shift_y + overlapping_height

        # 下地画像の上下左右端に合わせる
        # 
        if underlying_x_min < 0:
            overlapping_x_min = overlapping_x_min - underlying_x_min
            underlying_x_min = 0

        if underlying_width < underlying_x_max:
            overlapping_x_max = overlapping_x_max - (underlying_x_max - underlying_width)
            underlying_x_max = underlying_width

        if underlying_y_min < 0:
            overlapping_y_min = overlapping_y_min - underlying_y_min
            underlying_y_min = 0

        if underlying_height < underlying_y_max:
            overlapping_y_max = overlapping_y_max - (underlying_y_max - underlying_height)
            underlying_y_max = underlying_height

        underlying_img[underlying_y_min:underlying_y_max, underlying_x_min:underlying_x_max] = overlapping_img[overlapping_y_min:overlapping_y_max, overlapping_x_min:overlapping_x_max]

        return underlying_img
        
        
        
        
        
