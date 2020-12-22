import queue

import cv2
import numpy as np
import rclpy
from sensor_msgs.msg import CompressedImage
from openpose_ros2_msgs.msg import PoseKeyPointsList
from shigure.util import compressed_depth_util, frame_util

from shigure.nodes.node_image_preview import ImagePreviewNode


class CombineTestNode(ImagePreviewNode):
    depth_img_head: (int, int, np.ndarray)
    pose_key_points_head: (int, int, PoseKeyPointsList)

    def __init__(self):
        super().__init__("combine_test_node")

        self.depth_subscription = \
            self.create_subscription(CompressedImage,
                                     '/rs/aligned_depth_to_color/compressedDepth',
                                     self.get_depth_callback,
                                     10)
        self.openpose_subscription = self.create_subscription(PoseKeyPointsList, '/openpose/pose_key_points',
                                                              self.get_openpose_callback, 10)
        self.openpose_preview_subscription = self.create_subscription(CompressedImage, '/openpose/preview',
                                                                      self.get_openpose_img_callback, 10)

        self.depth_img_queue = queue.Queue()
        self.pose_key_points_queue = queue.Queue()

    def get_depth_callback(self, src: CompressedImage):
        print('Depth received', end='\r')

        raw_img = compressed_depth_util.convert_compressed_depth_img_to_cv2(src)
        rounded_img = frame_util.convert_frame_to_uint8(raw_img.astype(np.float32), 4000)
        color_img = cv2.applyColorMap(rounded_img, cv2.COLORMAP_OCEAN)
        img = cv2.pyrMeanShiftFiltering(color_img, 20, 50)

        if hasattr(self, 'depth_img_head'):
            self.depth_img_queue.put((src.header.stamp.sec, src.header.stamp.nanosec, img))
        else:
            self.depth_img_head = (src.header.stamp.sec, src.header.stamp.nanosec, img)

        self.combine()

    def get_openpose_callback(self, src: PoseKeyPointsList):
        print('KeyPoint received', end='\r')

        if hasattr(self, 'pose_key_points_head'):
            self.pose_key_points_queue.put((src.header.stamp.sec, src.header.stamp.nanosec, src))
        else:
            self.pose_key_points_head = (src.header.stamp.sec, src.header.stamp.nanosec, src)

        self.combine()

    def combine(self):
        if not hasattr(self, 'depth_img_head'):
            return
        if not hasattr(self, 'pose_key_points_head'):
            return

        img_sec, img_nano_sec, img = self.depth_img_head
        pose_sec, pose_nano_sec, pose = self.pose_key_points_head

        # 同時刻の画像を取得する
        while img_sec != pose_sec and img_nano_sec != pose_nano_sec:
            # imgの時刻が新しい -> key_pointが古いので一つ進める
            if img_sec > pose_sec or img_nano_sec > pose_nano_sec:
                if self.pose_key_points_queue.empty():
                    return
                self.pose_key_points_head = self.pose_key_points_queue.get()
                pose_sec, pose_nano_sec, pose = self.pose_key_points_head
            if img_sec < pose_sec or img_nano_sec < pose_nano_sec:
                if self.depth_img_queue.empty():
                    return
                self.depth_img_head = self.depth_img_queue.get()
                img_sec, img_nano_sec, img = self.depth_img_head

        for pose_key_points in pose.pose_key_points_list:
            for pose_key_point in pose_key_points.pose_key_points:
                x = int(pose_key_point.x)
                y = int(pose_key_point.y)

                if x == 0 and y == 0:
                    continue

                img = cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

        if not self.depth_img_queue.empty():
            self.depth_img_head = self.depth_img_queue.get()
        if not self.pose_key_points_queue.empty():
            self.pose_key_points_head = self.pose_key_points_queue.get()

        cv2.imshow("Result", img)
        cv2.waitKey(1)

    def get_openpose_img_callback(self, src: CompressedImage):
        img = self.bridge.compressed_imgmsg_to_cv2(src)
        cv2.imshow("Preview", img)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    combine_test_node = CombineTestNode()

    try:
        rclpy.spin(combine_test_node)

    except KeyboardInterrupt:
        pass

    finally:
        print()
        # 終了処理
        combine_test_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
