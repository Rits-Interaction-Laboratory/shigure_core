import datetime

import cv2
import numpy as np
import rclpy
from people_detection_ros2_msg.msg import People
from sensor_msgs.msg import Image, CompressedImage

from shigure_core.nodes.common_model.timestamp import Timestamp
from shigure_core.nodes.node_image_preview import ImagePreviewNode
from shigure_core.nodes.people_mask_buffer.logic import PeopleMaskBufferLogic
from shigure_core.nodes.people_mask_buffer.people_mask_buffer_frame import PeopleMaskBufferFrame
from shigure_core.nodes.people_mask_buffer.people_mask_buffer_frames import PeopleMaskBufferFrames


class PeopleMaskBufferNode(ImagePreviewNode):
    mask_count: np.ndarray

    def __init__(self):
        super().__init__('subtraction_analysis_node')
        self._publisher = self.create_publisher(CompressedImage,
                                                '/shigure/people_mask', 10)
        self._subscriber = self.create_subscription(People, '/people_detection', self.callback, 10)

        self._people_mask_buffer_logic = PeopleMaskBufferLogic()
        self._buffer_frames = PeopleMaskBufferFrames()

    def callback(self, people: People):
        try:
            self.get_logger().info('Buffering start', once=True)

            # FPS計算
            self.frame_count_up()

            people_mask: np.ndarray = self.bridge.compressed_imgmsg_to_cv2(people.mask)

            frame = PeopleMaskBufferFrame(people_mask.copy(), Timestamp(people.header.stamp.sec,
                                                                        people.header.stamp.nanosec))
            self._buffer_frames.add(frame)

            result, data = self._people_mask_buffer_logic.execute(self._buffer_frames)

            if result:
                self.get_logger().info('Buffering end', once=True)

                msg: Image = self.bridge.cv2_to_compressed_imgmsg(data, 'png')
                msg.header = people.header
                self._publisher.publish(msg)

                if self.is_debug_mode:
                    img = self.print_fps(data)
                    cv2.imshow("Result", cv2.hconcat([cv2.cvtColor(people_mask, cv2.COLOR_GRAY2BGR),
                                                      img]))
                    cv2.waitKey(1)
                else:
                    print(f'[{datetime.datetime.now()}] fps : {self.fps}', end='\r')

        except Exception as err:
            self.get_logger().error(err)


def main(args=None):
    rclpy.init(args=args)

    people_mask_buffer_node = PeopleMaskBufferNode()

    try:
        rclpy.spin(people_mask_buffer_node)

    except KeyboardInterrupt:
        pass

    finally:
        # 終了処理
        people_mask_buffer_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
