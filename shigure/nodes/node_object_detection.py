import cv2
import numpy as np
import rclpy
from sensor_msgs.msg import Image

from shigure.nodes.frame_combiner.frame_combiner import FrameCombiner
from shigure.nodes.node_image_preview import ImagePreviewNode
from shigure.nodes.object_detection.logic import ObjectDetectionLogic


class ObjectDetectionNode(ImagePreviewNode):
    frame_combiner: FrameCombiner[np.ndarray, np.ndarray]
    _known_mask: np.ndarray

    def __init__(self):
        super().__init__("object_detection_node")

        self.detection_publisher = self.create_publisher(Image, '/shigure/object_detection', 10)
        self.people_mask_subscription = self.create_subscription(Image, '/people_detection',
                                                                 self.get_people_mask_callback, 10)
        self.subtraction_analysis_subscription = self.create_subscription(Image, '/shigure/subtraction_analysis',
                                                                          self.get_subtraction_analysis_callback, 10)

        self.frame_combiner = FrameCombiner[np.ndarray, np.ndarray](round_nano_sec=500 * 1000000)
        self.object_detection_logic = ObjectDetectionLogic()

    def get_subtraction_analysis_callback(self, src: Image):
        img = self.bridge.imgmsg_to_cv2(src)
        self.frame_combiner.enqueue_to_left_queue(src.header.stamp.sec, src.header.stamp.nanosec, img.copy())

        self.combine()

    def get_people_mask_callback(self, src: Image):
        img = self.bridge.imgmsg_to_cv2(src)
        self.frame_combiner.enqueue_to_right_queue(src.header.stamp.sec, src.header.stamp.nanosec, img.copy())

        self.combine()

    def combine(self):
        result, sec, nano_sec, subtraction_analysis_img, people_mask = self.frame_combiner.dequeue_with_left()

        if not result:
            return

        if not hasattr(self, '_known_mask'):
            self._known_mask = np.ones(subtraction_analysis_img.shape[:2], np.uint8)

        if people_mask is None:
            people_mask = np.zeros(subtraction_analysis_img.shape[:2], np.uint8)

        result_img, self._known_mask = self.object_detection_logic.execute(subtraction_analysis_img, people_mask,
                                                                           self._known_mask)

        msg: Image = self.bridge.cv2_to_imgmsg(result_img)
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nano_sec
        self.detection_publisher.publish(msg)

        if self.is_debug_mode:
            cv2.imshow("Result", cv2.hconcat([subtraction_analysis_img, people_mask, result_img]))
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    object_detection_node = ObjectDetectionNode()

    try:
        rclpy.spin(object_detection_node)

    except KeyboardInterrupt:
        pass

    finally:
        print()
        # 終了処理
        object_detection_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
