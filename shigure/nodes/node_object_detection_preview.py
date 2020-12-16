import random

import rclpy
import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from shigure.nodes.object_detection.subtraction_frames import SubtractionFrames
from shigure.nodes.object_detection.logic import ObjectDetectionLogic


class BgSubtractionNode(Node):

    def __init__(self):
        super().__init__('object_detection_preview_node')
        self.bridge = CvBridge()
        self.subtraction_frames = SubtractionFrames(3)
        self.bg_subtraction_logic = ObjectDetectionLogic()
        self._publisher = self.create_publisher(Image,
                                                '/shigure/preview/object_detection/result/compressed', 10)
        self._subscription = self.create_subscription(Image, '/shigure/bg_subtraction',
                                                      self.get_subtraction_callback, 10)
        self.frame_count = 0

        # fps計測
        self.measurement_count = 10
        self.before_frame = 0
        self.fps = 0
        self.tm = cv2.TickMeter()
        self.tm.start()

    def get_subtraction_callback(self, image_rect_raw: Image):
        try:
            self.frame_count += 1
            frame: np.ndarray = self.bridge.imgmsg_to_cv2(image_rect_raw)
            result, data = self.bg_subtraction_logic.execute(self.subtraction_frames)
            self.subtraction_frames.add_frame(frame)

            if result:
                # fps計算
                if self.frame_count % self.measurement_count == 0:
                    self.tm.stop()
                    self.fps = (self.frame_count - self.before_frame) / self.tm.getTimeSec()
                    self.before_frame = self.frame_count
                    self.tm.reset()
                    self.tm.start()

                img: np.ndarray = np.zeros(np.append(data.shape, 3), np.uint8)
                img[:, :, 0], img[:, :, 1], img[:, :, 2] = data, data, data
                cv2.putText(img, "frame = " + str(self.frame_count), (0, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0))
                cv2.putText(img, 'FPS: {:.2f}'.format(self.fps),
                            (0, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0))
                self._publisher.publish(self.bridge.cv2_to_imgmsg(img))
                cv2.imshow("Result", img)

                # ラベリング処理
                ret, markers = cv2.connectedComponents(frame)

                # ラベリング結果を画面に表示
                # 各オブジェクトをランダム色でペイント
                tmp = markers > 0
                color_src = np.array([tmp, tmp, tmp]) * np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])

                # オブジェクトの総数を黄文字で表示
                cv2.putText(color_src, str(ret - 1), (100, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))

                # 結果の表示
                cv2.imshow("color_src", color_src)
                cv2.waitKey(1)
        except Exception as err:
            self.get_logger().error(err)


def main(args=None):
    rclpy.init(args=args)

    bg_subtraction_node = BgSubtractionNode()

    try:
        rclpy.spin(bg_subtraction_node)

    except KeyboardInterrupt:
        pass

    finally:
        # 終了処理
        bg_subtraction_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
