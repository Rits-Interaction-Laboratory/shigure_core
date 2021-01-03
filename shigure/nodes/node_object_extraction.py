import datetime

import cv2
import message_filters
import numpy as np
import rclpy
from sensor_msgs.msg import CompressedImage

from shigure.nodes.node_image_preview import ImagePreviewNode
from shigure.nodes.object_extraction.logic import ObjectExtractionLogic


class ObjectExtractionNode(ImagePreviewNode):
    _result_buffer_img: np.ndarray

    def __init__(self):
        super().__init__("object_extraction_node")

        self.detection_publisher = self.create_publisher(CompressedImage, '/shigure/object_extraction', 10)

        object_detection_subscriber = message_filters.Subscriber(self, CompressedImage, '/shigure/object_detection')
        color_subscriber = message_filters.Subscriber(self, CompressedImage, '/rs/color/compressed')
        self.time_synchronizer = message_filters.TimeSynchronizer(
            [object_detection_subscriber, color_subscriber], 10000)
        self.time_synchronizer.registerCallback(self.callback)

        self.object_extraction_logic = ObjectExtractionLogic()

    def callback(self, object_detection_src: CompressedImage, color_src: CompressedImage):
        self.frame_count_up()

        object_detection_img: np.ndarray = self.bridge.compressed_imgmsg_to_cv2(object_detection_src)
        color_img: np.ndarray = self.bridge.compressed_imgmsg_to_cv2(color_src)

        if not hasattr(self, '_result_buffer_img'):
            self._result_buffer_img = np.zeros(color_img.shape[:2])

        result_img, self._result_buffer_img = self.object_extraction_logic.execute(color_img, object_detection_img,
                                                                                   self._result_buffer_img, 200)

        msg: CompressedImage = self.bridge.cv2_to_compressed_imgmsg(result_img)
        msg.header.stamp = color_src.header.stamp
        self.detection_publisher.publish(msg)

        if self.is_debug_mode:
            img = self.print_fps(result_img)
            cv2.imshow("Result",
                       cv2.hconcat([color_img, cv2.cvtColor(object_detection_img, cv2.COLOR_GRAY2BGR), img]))
            cv2.waitKey(1)
        else:
            print(f'[{datetime.datetime.now()}] fps : {self.fps}', end='\r')


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
