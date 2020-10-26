import rclpy
import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from shigure.nodes.bg_subtraction.depth_frames import DepthFrames
from shigure.nodes.bg_subtraction.logic import BgSubtractionLogic


class BgSubtractionNode(Node):

    def __init__(self):
        super().__init__('bg_subtraction_node')
        self.depth_frames = DepthFrames()
        self.bg_subtraction_logic = BgSubtractionLogic()
        self.subscription = self.create_subscription(Image, '/camera/depth/image_rect_raw',
                                                     self.get_depth_callback, 10)
        self.frame_count = 0

        # fps計測
        self.measurement_count = 10
        self.before_frame = 0
        self.fps = 0
        self.tm = cv2.TickMeter()
        self.tm.start()

    def get_depth_callback(self, image_rect_raw: Image):
        try:
            self.frame_count += 1
            bridge = CvBridge()
            frame: np.ndarray = bridge.imgmsg_to_cv2(image_rect_raw)
            result, data = self.bg_subtraction_logic.execute(self.depth_frames, frame)
            self.depth_frames.add_frame(frame)

            if result:
                # 平均/分散の算出(有効フレーム考慮)
                valid_pixel = self.depth_frames.get_valid_pixel()
                avg = self.depth_frames.get_average() * valid_pixel
                var = self.depth_frames.get_var() * valid_pixel

                # cv2描画用にuint8に丸め込む
                rounded_frame = convert_frame_to_uint8(self.depth_frames.frames[-1], 1500)
                rounded_avg = convert_frame_to_uint8(avg, 1500)
                rounded_var = convert_frame_to_uint8(var, 1500 * 1500)

                # fps計算
                if self.frame_count % self.measurement_count == 0:
                    self.tm.stop()
                    self.fps = (self.frame_count - self.before_frame) / self.tm.getTimeSec()
                    self.before_frame = self.frame_count
                    self.tm.reset()
                    self.tm.start()

                cv2.imshow('depth', cv2.applyColorMap(rounded_frame, cv2.COLORMAP_OCEAN))
                cv2.imshow('avg', cv2.applyColorMap(rounded_avg, cv2.COLORMAP_OCEAN))
                cv2.imshow('var', cv2.applyColorMap(rounded_var, cv2.COLORMAP_OCEAN))
                img = np.zeros(np.append(data.shape, 3), np.uint8)
                img[:, :, 0], img[:, :, 1], img[:, :, 2] = data, data, data
                cv2.putText(img, "frame = " + str(self.frame_count), (0, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0))
                cv2.putText(img, 'FPS: {:.2f}'.format(self.fps),
                            (0, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0))
                cv2.imshow('bg_subtraction', img)
                cv2.waitKey(1)
        except Exception as err:
            self.get_logger().error(err)


def convert_frame_to_uint8(frame_float32: np.ndarray, threshold: int = 3000) -> np.ndarray:
    """
    cv2用にuint32のデータをuint8へ丸め込む
    デフォルトは3m

    :param frame_float32: uint32の深度画素データ
    :param threshold: 切り出す深度のしきい値
    :return: uint8に変換したframe配列データ
    """
    frame_float32 = np.where(frame_float32 > threshold, threshold, frame_float32)

    # cv2で表示するためにuint32をuint8へ変換
    # 近いほど白色に近づくように反転
    return (255 - frame_float32 * 255 / threshold).astype(np.uint8)


def main(args=None):
    rclpy.init(args=args)

    bg_subtraction_node = BgSubtractionNode()

    rclpy.spin(bg_subtraction_node)

    # 終了処理
    bg_subtraction_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
