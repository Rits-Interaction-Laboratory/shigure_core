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
        self.publisher_ = self.create_publisher(Image, '/shigure/bg_subtraction', 10)
        self.subscription = self.create_subscription(Image, '/camera/depth/image_rect_raw',
                                                     self.get_depth_callback, 10)

    def get_depth_callback(self, image_rect_raw: Image):
        try:
            bridge = CvBridge()
            frame: np.ndarray = bridge.imgmsg_to_cv2(image_rect_raw)
            result, data = self.bg_subtraction_logic.execute(self.depth_frames, frame)
            self.depth_frames.add_frame(frame)
            if result:
                cv2.imshow('depth', frame)
                cv2.waitKey(1)
                cv2.imshow('bg_subtraction', np.uint8(data))
                cv2.waitKey(1)
        except Exception as err:
            self.get_logger().error(err)


def main(args=None):
    rclpy.init(args=args)

    bg_subtraction_node = BgSubtractionNode()

    rclpy.spin(bg_subtraction_node)

    # 終了処理
    bg_subtraction_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
