import random
import re

import cv2
import message_filters
import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, CameraInfo
from shigure_core_msgs.msg import DetectedObjectList, TrackedObjectList, TrackedObject, Cube
from bboxes_ex_msgs.msg import Segments

from shigure_core.nodes.node_image_preview import ImagePreviewNode
from shigure_core.nodes.object_tracking.logic import ObjectTrackingLogic
from shigure_core.nodes.object_tracking.tracking_info import TrackingInfo
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

        # /Segments は TimeSynchronizer に入れず最新を直接キャッシュする
        self._latest_segments = None
        self.create_subscription(
            Segments,
            '/Segments',
            self._on_segments,
            shigure_qos,
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

        self.people_tracking_logic = ObjectTrackingLogic()

        self._tracking_info = TrackingInfo()

        self._colors = []
        for i in range(255):
            self._colors.append(tuple([random.randint(128, 192) for _ in range(3)]))

    def _on_segments(self, msg: Segments):
        self._latest_segments = msg

    @staticmethod
    def _bbox_iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return float(inter) / float(union) if union > 0 else 0.0

    def _build_fresh_bbox_mask(self, bbox_x, bbox_y, bbox_w, bbox_h, img_h, img_w):
        """最新の /Segments から該当 BBOX 用のバイナリマスクを生成する。
        見つからなければ None を返す。"""
        if self._latest_segments is None:
            return None, None
        best_seg = None
        best_iou = 0.0
        bbox_a = (float(bbox_x), float(bbox_y), float(bbox_x + bbox_w), float(bbox_y + bbox_h))
        for seg in self._latest_segments.segments:
            if len(seg.x_masks) == 0 or len(seg.x_masks) != len(seg.y_masks):
                continue
            iou = self._bbox_iou(bbox_a, (float(seg.xmin), float(seg.ymin), float(seg.xmax), float(seg.ymax)))
            if iou > best_iou:
                best_iou = iou
                best_seg = seg
        if best_seg is None or best_iou < 0.3:
            return None, best_iou
        # x_masks/y_masks は (row, col)=(y, x) 順なので入れ替え
        pts = np.stack([np.asarray(best_seg.y_masks, dtype=np.int32),
                        np.asarray(best_seg.x_masks, dtype=np.int32)], axis=1).reshape(-1, 1, 2)
        full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(full_mask, [pts], 255)
        x1 = max(0, min(int(bbox_x), img_w - 1))
        y1 = max(0, min(int(bbox_y), img_h - 1))
        x2 = max(x1 + 1, min(int(bbox_x + bbox_w), img_w))
        y2 = max(y1 + 1, min(int(bbox_y + bbox_h), img_h))
        return full_mask[y1:y2, x1:x2], best_iou

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
        EROSION_PIXELS = 5
        for object_id, item in self._tracking_info.object_dict.items():
            stay_object, bounding_box = item

            tracked_object = TrackedObject()
            tracked_object.object_id = object_id
            tracked_object.action = stay_object.action
            tracked_object.bounding_box = stay_object.bounding_box

            bounding_box = stay_object.bounding_box
            left = max(0, min(int(bounding_box.x), width - 1))
            top = max(0, min(int(bounding_box.y), height - 1))
            right = max(left + 1, min(int(bounding_box.x + bounding_box.width), width))
            bottom = max(top + 1, min(int(bounding_box.y + bounding_box.height), height))

            depth_roi = depth_img[top:bottom, left:right]

            # まず最新 /Segments から BBOX に対応するマスクを再生成 (毎フレーム最新)
            seg_mask = None
            fresh_mask, fresh_iou = self._build_fresh_bbox_mask(
                bounding_box.x, bounding_box.y, bounding_box.width, bounding_box.height,
                height, width,
            )
            mask_source = 'none'
            if fresh_mask is not None:
                if fresh_mask.shape[:2] != depth_roi.shape[:2]:
                    fresh_mask = cv2.resize(fresh_mask, (depth_roi.shape[1], depth_roi.shape[0]),
                                            interpolation=cv2.INTER_NEAREST)
                seg_mask = (fresh_mask > 0).astype(np.uint8) * 255
                mask_source = f'fresh(iou={fresh_iou:.2f})'
            elif stay_object.mask is not None and len(stay_object.mask.data) > 0:
                try:
                    decoded = self.bridge.compressed_imgmsg_to_cv2(stay_object.mask)
                    if decoded.ndim == 3:
                        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
                    if decoded.shape[:2] != depth_roi.shape[:2]:
                        decoded = cv2.resize(decoded, (depth_roi.shape[1], depth_roi.shape[0]),
                                             interpolation=cv2.INTER_NEAREST)
                    seg_mask = (decoded > 0).astype(np.uint8) * 255
                    mask_source = 'stale_msg'
                except Exception as e:
                    self.get_logger().warning(f'mask decode failed for {object_id}: {e}')
                    seg_mask = None

            depth_min = None
            depth_max = None
            path_taken = 'none'
            mask_coverage = 0.0
            depth_stats_log = ''
            if seg_mask is not None:
                mask_coverage = float((seg_mask > 0).sum()) / max(1, seg_mask.size)
                kernel_size = EROSION_PIXELS * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                eroded_mask = cv2.erode(seg_mask, kernel, iterations=1)
                valid = (eroded_mask > 0) & (depth_roi > 0)
                if valid.any():
                    vals = depth_roi[valid]
                    p5, p50, p95 = np.percentile(vals, [5, 50, 95])
                    depth_min = float(p5)
                    depth_max = float(p95)
                    depth_stats_log = f' p5={p5:.0f} median={p50:.0f} p95={p95:.0f} raw_min={vals.min():.0f} raw_max={vals.max():.0f}'
                    path_taken = f'eroded({valid.sum()}px)'
                else:
                    valid_fallback = (seg_mask > 0) & (depth_roi > 0)
                    if valid_fallback.any():
                        vals_fb = depth_roi[valid_fallback]
                        depth_min = float(np.percentile(vals_fb, 5))
                        depth_max = float(np.percentile(vals_fb, 95))
                        path_taken = f'mask_only({valid_fallback.sum()}px)'

            if depth_min is None or depth_max is None:
                masked_depth_img = np.ma.masked_equal(depth_roi, 0.0, copy=False)
                if masked_depth_img.count() == 0:
                    continue
                vals_bb = masked_depth_img.compressed()
                depth_min = float(np.percentile(vals_bb, 5))
                depth_max = float(np.percentile(vals_bb, 95))
                path_taken = f'bbox_fallback({masked_depth_img.count()}px)'

            self.get_logger().info(
                f'[{object_id}] mask={mask_source} path={path_taken} mask_cov={mask_coverage:.2f} '
                f'roi={depth_roi.shape} dmin={depth_min:.0f} dmax={depth_max:.0f} '
                f'depth_range={depth_max - depth_min:.0f}mm{depth_stats_log}'
            )

            s1 = np.asarray([[bounding_box.x, bounding_box.y, 1]]).T
            s2 = np.asarray([[bounding_box.x + bounding_box.width,
                              bounding_box.y + bounding_box.height, 1]]).T

            m1 = (depth_min * np.matmul(k_inv, s1)).T
            m2 = (depth_min * np.matmul(k_inv, s2)).T

            collider = Cube()
            collider.x, collider.y = float(m1[0, 0]), float(m1[0, 1])
            collider.width, collider.height = float(m2[0, 0] - m1[0, 0]), float(m2[0, 1] - m1[0, 1])
            collider.z = float(depth_min)
            collider.depth = float(depth_max - depth_min)
            tracked_object.collider = collider

            publish_msg.tracked_object_list.append(tracked_object)
            

        self._publisher.publish(publish_msg)
        publish_msg.tracked_object_list = []

    def callback_debug(self, depth_src: CompressedImage, detected_object_list: DetectedObjectList,
                       camera_info: CameraInfo, color_src: CompressedImage):
        color_img: np.ndarray = self.bridge.compressed_imgmsg_to_cv2(color_src)

        self.callback(depth_src, detected_object_list, camera_info)

        height, width = color_img.shape[:2]
        for object_id, item in self._tracking_info.object_dict.items():
            stay_object, bounding_box = item

            bounding_box = stay_object.bounding_box
            left = min(int(bounding_box.x), width - 1)
            top = min(int(bounding_box.y), height - 1)
            right = min(int(bounding_box.x + bounding_box.width), width - 1)
            bottom = min(int(bounding_box.y + bounding_box.height), height - 1)

            object_id_num = int(re.sub(".*_", "", object_id))
            color = self._colors[object_id_num % 255]
            cv2.rectangle(color_img, (left, top), (right, bottom), color, thickness=3)
            text_w, text_h = cv2.getTextSize(f'ID : {object_id_num}',
                                             cv2.FONT_HERSHEY_PLAIN, 1.5, 2)[0]
            cv2.rectangle(color_img, (left, top), (left + text_w, top - text_h), color, -1)
            cv2.putText(color_img, f'ID : {object_id_num}({stay_object.action})', (left, top),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), thickness=2)

        self.print_fps(color_img)
        cv2.namedWindow('object_tracking', cv2.WINDOW_NORMAL)
        cv2.imshow('object_tracking', color_img)
        cv2.waitKey(1)


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
