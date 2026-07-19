import cv2
import message_filters
import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, CameraInfo
from shigure_core_msgs.msg import DetectedObjectList, TrackedObjectList, TrackedObject

from shigure_core.nodes.node_image_preview import ImagePreviewNode
from shigure_core.nodes.object_tracking.collider import build_collider, compute_depth_range
from shigure_core.nodes.object_tracking.logic import ObjectTrackingLogic
from shigure_core.nodes.object_tracking.tracking_info import TrackingInfo
from shigure_core.nodes.object_tracking.visualizer import ObjectTrackingVisualizer
from shigure_core.util import compressed_depth_util


class ObjectTrackingNode(ImagePreviewNode):

    def __init__(self):
        super().__init__('object_tracking_node')

        # QoS Settings
        shigure_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # publisher, subscriber
        self._publisher = self.create_publisher(
            TrackedObjectList, 
            '/shigure/object_tracking', 
            10
        )
        depth_subscriber = message_filters.Subscriber(
            self, 
            CompressedImage,
            '/rs/aligned_depth_to_color/compressedDepth', 
            qos_profile=shigure_qos
        )
        depth_camera_info_subscriber = message_filters.Subscriber(
            self, 
            CameraInfo,
            '/rs/aligned_depth_to_color/cameraInfo', 
            qos_profile=shigure_qos
        )
        object_detection_subscriber = message_filters.Subscriber(
            self, 
            DetectedObjectList, 
            '/shigure/object_detection', 
            qos_profile=shigure_qos
        )

        if not self.is_debug_mode:
            self.time_synchronizer = message_filters.TimeSynchronizer(
                [depth_subscriber, object_detection_subscriber, depth_camera_info_subscriber], 1000)
            self.time_synchronizer.registerCallback(self.callback)
        else:
            color_subscriber = message_filters.Subscriber(
                self, 
                CompressedImage, 
                '/rs/color/compressed',
                qos_profile=shigure_qos
            )
            self.time_synchronizer = message_filters.TimeSynchronizer(
                [depth_subscriber, object_detection_subscriber, depth_camera_info_subscriber, color_subscriber], 400000)
            self.time_synchronizer.registerCallback(self.callback_debug)

        self._tracking_info = TrackingInfo()

    def _decode_mask_roi(self, mask_msg, roi_shape):
        """物体マスク(CompressedImage)を復号し depth_roi と同形に切り出す（ROS境界）.

        マスクは bbox 左上を原点とするため、depth_roi の大きさに合わせて左上を切り出す。
        復号失敗・2次元でない場合は None を返す（呼び出し側で bbox 深度にフォールバック）。
        """
        try:
            mask = self.bridge.compressed_imgmsg_to_cv2(mask_msg)
        except Exception:
            return None
        if mask is None or mask.ndim != 2:
            return None
        return mask[0:roi_shape[0], 0:roi_shape[1]]

    def callback(self, depth_src: CompressedImage, detected_object_list: DetectedObjectList,
                 camera_info: CameraInfo):
        self.frame_count_up()

        depth_img = compressed_depth_util.convert_compressed_depth_img_to_cv2(depth_src)
        depth_img: np.ndarray = depth_img.astype(np.float32)

        # 焦点距離取得
        #     [fx  0 cx]
        # K = [ 0 fy cy]
        #     [ 0  0  1]
        k = camera_info.k.reshape((3, 3))

        self._tracking_info = ObjectTrackingLogic.execute(depth_img, detected_object_list, self._tracking_info)

        # publish
        publish_msg = TrackedObjectList()
        publish_msg.header = detected_object_list.header
        publish_msg.header.frame_id = camera_info.header.frame_id

        k_inv = np.linalg.inv(k)
        height, width = depth_img.shape[:2]
        for object_id, item in self._tracking_info.object_dict.items():
            stay_object, bounding_box = item

            tracked_object = TrackedObject()
            tracked_object.object_id = object_id
            tracked_object.action = stay_object.action
            tracked_object.bounding_box = stay_object.bounding_box

            bounding_box = stay_object.bounding_box
            left = min(int(bounding_box.x), width - 1)
            top = min(int(bounding_box.y), height - 1)
            right = min(int(bounding_box.x + bounding_box.width), width - 1)
            bottom = min(int(bounding_box.y + bounding_box.height), height - 1)

            depth_roi = depth_img[top:bottom, left:right]
            mask_roi = self._decode_mask_roi(stay_object.mask, depth_roi.shape)
            depth_min, depth_max = compute_depth_range(depth_roi, mask_roi)
            tracked_object.collider = build_collider(bounding_box, depth_min, depth_max, k_inv)

            publish_msg.tracked_object_list.append(tracked_object)
            

        self._publisher.publish(publish_msg)
        publish_msg.tracked_object_list = []

    def callback_debug(self, depth_src: CompressedImage, detected_object_list: DetectedObjectList,
                       camera_info: CameraInfo, color_src: CompressedImage):
        self.callback(depth_src, detected_object_list, camera_info)

        if not self.is_debug_mode:
            cv2.destroyAllWindows()
            return

        color_img: np.ndarray = self.bridge.compressed_imgmsg_to_cv2(color_src)
        ObjectTrackingVisualizer.draw(color_img, self._tracking_info.object_dict,
                                      self.frame_count, self.fps)


def main(args=None):
    rclpy.init(args=args)

    object_tracking_node = ObjectTrackingNode()

    try:
        rclpy.spin(object_tracking_node)

    except KeyboardInterrupt:
        pass

    finally:
        # 終了処理
        object_tracking_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
