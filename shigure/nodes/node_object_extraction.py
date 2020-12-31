import cv2
import numpy as np
import rclpy
from sensor_msgs.msg import Image, CompressedImage

from shigure.nodes.frame_combiner.frame_combiner import FrameCombiner
from shigure.nodes.node_image_preview import ImagePreviewNode
from shigure.nodes.object_extraction.logic import ObjectExtractionLogic


class ObjectExtractionNode(ImagePreviewNode):
    frame_combiner: FrameCombiner[np.ndarray, np.ndarray]

    def __init__(self):
        super().__init__("object_extraction_node")

        self.detection_publisher = self.create_publisher(Image, '/shigure/object_extraction', 10)
        self.object_detection_subscription = self.create_subscription(Image, '/shigure/object_detection',
                                                                      self.get_object_detection_callback, 10)
        self.color_subscription = self.create_subscription(CompressedImage, '/rs/color/compressed',
                                                           self.get_color_callback, 10)

        self.frame_combiner = FrameCombiner[np.ndarray, np.ndarray]()
        self.object_extraction_logic = ObjectExtractionLogic()

    def get_color_callback(self, src: CompressedImage):
        img = self.bridge.compressed_imgmsg_to_cv2(src)
        self.frame_combiner.enqueue_to_left_queue(src.header.stamp.sec, src.header.stamp.nanosec, img.copy())

        self.combine()

    def get_object_detection_callback(self, src: Image):
        img = self.bridge.imgmsg_to_cv2(src)
        self.frame_combiner.enqueue_to_right_queue(src.header.stamp.sec, src.header.stamp.nanosec, img.copy())

        self.combine()

    def combine(self):
        result, sec, nano_sec, color_img, object_detection_img = self.frame_combiner.dequeue()

        if not result:
            return

        result_img = self.object_extraction_logic.execute(color_img, object_detection_img, 200)

        msg: Image = self.bridge.cv2_to_imgmsg(result_img)
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nano_sec
        self.detection_publisher.publish(msg)

        if self.is_debug_mode:
            cv2.imshow("Result",
                       cv2.hconcat([color_img, cv2.cvtColor(object_detection_img, cv2.COLOR_GRAY2BGR), result_img]))
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    object_extraction_node = ObjectExtractionNode()

    try:
        rclpy.spin(object_extraction_node)

    except KeyboardInterrupt:
        pass

    finally:
        print()
        # 終了処理
        object_extraction_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
