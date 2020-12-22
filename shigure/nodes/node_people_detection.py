import cv2
import numpy as np
import rclpy
from openpose_ros2_msgs.msg import PoseKeyPointsList
from sensor_msgs.msg import CompressedImage, Image

from shigure.nodes.frame_combiner.frame_combiner import FrameCombiner
from shigure.nodes.node_image_preview import ImagePreviewNode
from shigure.nodes.people_detection.logic import PeopleDetectionLogic
from shigure.util import compressed_depth_util, frame_util


class PeopleDetectionNode(ImagePreviewNode):
    frame_combiner: FrameCombiner[np.ndarray, PoseKeyPointsList]

    def __init__(self):
        super().__init__("people_detection_node")

        self.mask_publisher = self.create_publisher(Image, '/shigure/people_subtraction', 10)
        self.depth_subscription = self.create_subscription(CompressedImage,
                                                           '/rs/aligned_depth_to_color/compressedDepth',
                                                           self.get_depth_callback, 10)
        self.openpose_subscription = self.create_subscription(PoseKeyPointsList, '/openpose/pose_key_points',
                                                              self.get_openpose_callback, 10)
        if self.is_debug_mode:
            self.openpose_preview_subscription = self.create_subscription(CompressedImage, '/openpose/preview',
                                                                          self.get_openpose_img_callback, 10)

        self.frame_combiner = FrameCombiner[np.ndarray, PoseKeyPointsList]()
        self.people_detection_logic = PeopleDetectionLogic()

    def get_depth_callback(self, src: CompressedImage):
        print('Depth received', end='\r')

        raw_img = compressed_depth_util.convert_compressed_depth_img_to_cv2(src)
        rounded_img = frame_util.convert_frame_to_uint8(raw_img.astype(np.float32), 4000)
        img = cv2.applyColorMap(rounded_img, cv2.COLORMAP_OCEAN)

        self.frame_combiner.enqueue_to_left_queue(src.header.stamp.sec, src.header.stamp.nanosec, img)

        self.combine()

    def get_openpose_callback(self, src: PoseKeyPointsList):
        print('KeyPoint received', end='\r')

        self.frame_combiner.enqueue_to_right_queue(src.header.stamp.sec, src.header.stamp.nanosec, src)

        self.combine()

    def combine(self):
        result, sec, nano_sec, img, pose = self.frame_combiner.dequeue()

        if not result:
            return

        result_img, debug_img = self.people_detection_logic.execute(img, pose, self.is_debug_mode)

        msg: Image = self.bridge.cv2_to_imgmsg(result_img)
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nano_sec
        self.mask_publisher.publish(msg)

        if self.is_debug_mode:
            cv2.imshow("Result", cv2.hconcat([cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR), debug_img]))
            cv2.waitKey(1)

    def get_openpose_img_callback(self, src: CompressedImage):
        img = self.bridge.compressed_imgmsg_to_cv2(src)
        cv2.imshow("Preview", img)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    people_detection_node = PeopleDetectionNode()

    try:
        rclpy.spin(people_detection_node)

    except KeyboardInterrupt:
        pass

    finally:
        print()
        # 終了処理
        people_detection_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
