import datetime
import random

import cv2
import message_filters
import numpy as np
import rclpy
from people_detection_ros2_msg.msg import People
from sensor_msgs.msg import Image, CompressedImage, CameraInfo

from shigure_core.nodes.common_model.timestamp import Timestamp
from shigure_core.nodes.node_image_preview import ImagePreviewNode
from shigure_core.nodes.subtraction_analysis.logic import SubtractionAnalysisLogic
from shigure_core.nodes.subtraction_analysis.subtraction_frame import SubtractionFrame
from shigure_core.nodes.subtraction_analysis.subtraction_frames import SubtractionFrames


class SubtractionAnalysisNode(ImagePreviewNode):

    def __init__(self):
        super().__init__('subtraction_analysis_node')
        self.subtraction_frames = SubtractionFrames()
        self.bg_subtraction_logic = SubtractionAnalysisLogic()
        self._publisher = self.create_publisher(CompressedImage,
                                                '/shigure/subtraction_analysis', 10)
        people_subscriber = message_filters.Subscriber(self, People, '/people_detection')
        subtraction_subscriber = message_filters.Subscriber(self, CompressedImage,
                                                            '/shigure/bg_subtraction')
        depth_camera_info_subscriber = message_filters.Subscriber(self, CameraInfo, '/rs/aligned_depth_to_color/cameraInfo')
        self.time_synchronizer = message_filters.TimeSynchronizer(
            [people_subscriber, subtraction_subscriber, depth_camera_info_subscriber], 1000)
        self.time_synchronizer.registerCallback(self.callback)

        self._colors = []
        for i in range(255):
            self._colors.append(tuple([random.randint(128, 255) for _ in range(3)]))

    def callback(self, people: People, subtraction_src: CompressedImage, camera_info: CameraInfo):
        try:
            self.get_logger().info('Buffering start', once=True)

            # FPS計算
            self.frame_count_up()

            subtraction_img: np.ndarray = self.bridge.compressed_imgmsg_to_cv2(subtraction_src)
            people_mask: np.ndarray = self.bridge.compressed_imgmsg_to_cv2(people.mask)
            frame = SubtractionFrame(subtraction_img.copy(), people_mask.copy(),
                                     Timestamp(subtraction_src.header.stamp.sec,
                                               subtraction_src.header.stamp.nanosec))
            self.subtraction_frames.add_frame(frame)
            result, data, result_frame = self.bg_subtraction_logic.execute(self.subtraction_frames)

            if result:
                self.get_logger().info('Buffering end', once=True)

                msg: Image = self.bridge.cv2_to_compressed_imgmsg(data, 'png')
                sec, nano_sec = self.subtraction_frames.get_timestamp()
                msg.header.stamp.sec = sec
                msg.header.stamp.nanosec = nano_sec
                msg.header.frame_id = camera_info.header.frame_id
                self._publisher.publish(msg)

                if self.is_debug_mode:
                    img = self.print_fps(data)
                    cv2.imshow("subtraction_analysis", cv2.hconcat([cv2.cvtColor(result_frame.subtraction_img, cv2.COLOR_GRAY2BGR),
                                                      cv2.cvtColor(result_frame.people_mask, cv2.COLOR_GRAY2BGR),
                                                      img]))
                    cv2.waitKey(1)
                else:
                    print(f'[{datetime.datetime.now()}] fps : {self.fps}', end='\r')

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
