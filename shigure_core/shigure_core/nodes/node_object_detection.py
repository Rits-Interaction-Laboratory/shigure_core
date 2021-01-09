import datetime

import cv2
import message_filters
import numpy as np
import rclpy
from sensor_msgs.msg import CompressedImage

from shigure_core.nodes.node_image_preview import ImagePreviewNode
from shigure_core.nodes.object_detection.logic import ObjectDetectionLogic


class ObjectDetectionNode(ImagePreviewNode):
    _known_mask: np.ndarray

    def __init__(self):
        super().__init__("object_detection_node")

        self.detection_publisher = self.create_publisher(CompressedImage, '/shigure/object_detection', 10)

        people_mask_subscriber = message_filters.Subscriber(self, CompressedImage, '/people_detection')
        subtraction_analysis_subscriber = message_filters.Subscriber(self, CompressedImage,
                                                                     '/shigure/subtraction_analysis')
        self.time_synchronizer = message_filters.TimeSynchronizer(
            [people_mask_subscriber, subtraction_analysis_subscriber], 10000)
        self.time_synchronizer.registerCallback(self.callback)

        self.object_detection_logic = ObjectDetectionLogic()

    def callback(self, people_mask_src: CompressedImage, subtraction_analysis_src: CompressedImage):
        self.frame_count_up()

        people_mask_img = self.bridge.compressed_imgmsg_to_cv2(people_mask_src)
        subtraction_analysis_img = self.bridge.compressed_imgmsg_to_cv2(subtraction_analysis_src)

        if not hasattr(self, '_known_mask'):
            self._known_mask = np.ones(subtraction_analysis_img.shape[:2], np.uint8)

        result_img, self._known_mask = self.object_detection_logic.execute(subtraction_analysis_img, people_mask_img,
                                                                           self._known_mask)

        msg: CompressedImage = self.bridge.cv2_to_compressed_imgmsg(result_img)
        msg.header.stamp = people_mask_src.header.stamp
        self.detection_publisher.publish(msg)

        if self.is_debug_mode:
            img = self.print_fps(result_img)
            cv2.imshow("Result", cv2.hconcat([cv2.cvtColor(subtraction_analysis_img, cv2.COLOR_GRAY2BGR),
                                              cv2.cvtColor(people_mask_img, cv2.COLOR_GRAY2BGR), img]))
            cv2.waitKey(1)
        else:
            print(f'[{datetime.datetime.now()}] fps : {self.fps}', end='\r')


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
