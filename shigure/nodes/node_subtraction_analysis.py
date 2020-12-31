import rclpy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
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
        self._publisher = self.create_publisher(CompressedImage,
                                                '/shigure/subtraction_analysis', 10)
        self._subscription = self.create_subscription(CompressedImage, '/shigure/bg_subtraction',
                                                      self.get_subtraction_callback, 10)

    def get_subtraction_callback(self, image_rect_raw: CompressedImage):
        try:
            # FPS計算
            self.frame_count_up()

            frame: np.ndarray = self.bridge.compressed_imgmsg_to_cv2(image_rect_raw)
            result, data = self.bg_subtraction_logic.execute(self.subtraction_frames)
            self.subtraction_frames.add_frame(frame, Timestamp(image_rect_raw.header.stamp.sec,
                                                               image_rect_raw.header.stamp.nanosec))

            if result:
                msg: Image = self.bridge.cv2_to_compressed_imgmsg(data)
                sec, nano_sec = self.subtraction_frames.get_timestamp()
                msg.header.stamp.sec = sec
                msg.header.stamp.nanosec = nano_sec
                self._publisher.publish(msg)

                if self.is_debug_mode:
                    img = self.print_fps(data)
                    cv2.imshow("Result", cv2.hconcat([cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), img]))
                    cv2.waitKey(1)
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
