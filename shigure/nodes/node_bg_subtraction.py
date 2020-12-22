import rclpy
import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from shigure.util import compressed_depth_util

from shigure.nodes.bg_subtraction.depth_frames import DepthFrames
from shigure.nodes.bg_subtraction.logic import BgSubtractionLogic


class BgSubtractionNode(Node):

    def __init__(self):
        super().__init__('bg_subtraction_node')
        self.bridge = CvBridge()
        self.depth_frames = DepthFrames()
        self.bg_subtraction_logic = BgSubtractionLogic()
        self.publisher_ = self.create_publisher(Image, '/shigure/bg_subtraction', 10)
        self.subscription = self.create_subscription(CompressedImage, '/rs/aligned_depth_to_color/compressedDepth',
                                                     self.get_depth_callback, 10)

    def get_depth_callback(self, image_rect_raw: CompressedImage):
        try:
            frame: np.ndarray = compressed_depth_util.convert_compressed_depth_img_to_cv2(image_rect_raw)
            result, data = self.bg_subtraction_logic.execute(self.depth_frames, frame)
            self.depth_frames.add_frame(frame)

            if result:
                msg: Image = self.bridge.cv2_to_imgmsg(data.astype(np.uint8))
                msg.header.stamp = image_rect_raw.header.stamp
                self.publisher_.publish(msg)

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
