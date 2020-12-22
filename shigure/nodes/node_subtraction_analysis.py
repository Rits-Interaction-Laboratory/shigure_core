import rclpy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from shigure.nodes.node_image_preview import ImagePreviewNode
from shigure.nodes.subtraction_analysis.subtraction_frames import SubtractionFrames
from shigure.nodes.subtraction_analysis.logic import SubtractionAnalysisLogic
from shigure.nodes.subtraction_analysis.timestamp import Timestamp


class SubtractionAnalysisNode(ImagePreviewNode):

    def __init__(self):
        super().__init__('subtraction_analysis_node')
        self.bridge = CvBridge()
        self.subtraction_frames = SubtractionFrames()
        self.bg_subtraction_logic = SubtractionAnalysisLogic()
        self._publisher = self.create_publisher(Image,
                                                '/shigure/subtraction_analysis', 10)
        self._subscription = self.create_subscription(Image, '/shigure/bg_subtraction',
                                                      self.get_subtraction_callback, 10)

    def get_subtraction_callback(self, image_rect_raw: Image):
        try:
            # FPS計算
            self.frame_count_up()

            frame: np.ndarray = self.bridge.imgmsg_to_cv2(image_rect_raw)
            result, data = self.bg_subtraction_logic.execute(self.subtraction_frames)
            self.subtraction_frames.add_frame(frame, Timestamp(image_rect_raw.header.stamp.sec,
                                                               image_rect_raw.header.stamp.nanosec))

            if result:
                msg: Image = self.bridge.cv2_to_imgmsg(data)
                sec, nano_sec = self.subtraction_frames.get_timestamp()
                msg.header.stamp.sec = sec
                msg.header.stamp.nanosec = nano_sec
                self._publisher.publish(msg)

                if self.is_debug_mode:
                    img = self.print_fps(data)
                    cv2.imshow("Result", img)
                    cv2.waitKey(1)

                # ラベリング処理(移行予定)
                # ret, labels, stats, centroids = cv2.connectedComponentsWithStats(data.astype(np.uint8), connectivity=8)
                #
                # color_img = cv2.cvtColor(np.zeros_like(data).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                #
                # for i, row in enumerate(stats):
                #     area = row[cv2.CC_STAT_AREA]
                #     x = row[cv2.CC_STAT_LEFT]
                #     y = row[cv2.CC_STAT_TOP]
                #     height = row[cv2.CC_STAT_HEIGHT]
                #     width = row[cv2.CC_STAT_WIDTH]
                #
                #     if i != 0 and area > 200:
                #         color_img += cv2.cvtColor(np.where(labels == i, 255, 0).astype(np.uint8),
                #                                   cv2.COLOR_GRAY2BGR) * np.array([100, 255, 255]).astype(np.uint8)
                #         cv2.rectangle(color_img, (x, y), (x + width, y + height), (255, 255, 100))
                #
                # # 結果の表示
                # cv2.imshow("color_src", color_img)
        except Exception as err:
            self.get_logger().error(err)


def main(args=None):
    rclpy.init(args=args)

    subtraction_analysis_node = SubtractionAnalysisNode()

    try:
        rclpy.spin(subtraction_analysis_node)

    except KeyboardInterrupt:
        pass

    finally:
        # 終了処理
        subtraction_analysis_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
