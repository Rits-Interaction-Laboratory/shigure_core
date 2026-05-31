import datetime
import re
from typing import List, Tuple
from copy import deepcopy

import cv2
import message_filters
import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, CameraInfo
from shigure_core_msgs.msg import TrackedObjectList, ContactedList, Contacted
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Point

from shigure_core.enum.contact_action_enum import ContactActionEnum
from shigure_core.enum.tracked_object_action_enum import TrackedObjectActionEnum
from shigure_core.nodes.contact_detection.id_manager import IdManager
from shigure_core.nodes.contact_detection.raycast_hit_logic import RaycastHitDetectionLogic
from shigure_core.nodes.node_image_preview import ImagePreviewNode
from shigure_core.nodes.object_detection.frame_object import FrameObject

#----------------------------------------------------------------------------------------------------
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
import tf2_geometry_msgs
#----------------------------------------------------------------------------------------------------

class RaycastHitDetectionNode(ImagePreviewNode):
    def __init__(self):
        super().__init__("raycast_hit_detection_node")

        # QoS Settings
        shigure_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # publisher, subscriber
        self._publisher = self.create_publisher(
            ContactedList, 
            '/shigure/RaycastHit', 
            10
        )
        object_subscriber = message_filters.Subscriber(
            self, 
            TrackedObjectList, 
            '/shigure/object_tracking', 
            qos_profile=shigure_qos
        )
        raycast_hit_point_subscriber = message_filters.Subscriber(
            self, 
            PointStamped, 
            'raycastHit', 
            qos_profile=shigure_qos
        )
        color_subscriber = message_filters.Subscriber(
            self, 
            CompressedImage, 
            '/rs/color/compressed',
            qos_profile=shigure_qos
        )
        depth_camera_info_subscriber = message_filters.Subscriber(
            self, 
            CameraInfo, 
            '/rs/aligned_depth_to_color/cameraInfo', 
            qos_profile=shigure_qos
        )
        bbox_generate_button_subscriber = message_filters.Subscriber(
            self, 
            PointStamped, 
            '/shigure/generate_bbox_switch', 
            qos_profile=shigure_qos
        )
        
        #--------------------------------------------
        self.debug_publisher = self.create_publisher(
            Point,
            'debug/do_transform/point', 
            10
        )
        #--------------------------------------------

        queue_size = 1000
        fps = 10.
        delay = 1 / fps * 0.5

        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [object_subscriber, raycast_hit_point_subscriber, color_subscriber, depth_camera_info_subscriber], queue_size, delay)
        self.time_synchronizer.registerCallback(self.callback)
        
        self.time_synchronizer_2 = message_filters.ApproximateTimeSynchronizer(
            [object_subscriber, bbox_generate_button_subscriber, color_subscriber, depth_camera_info_subscriber], queue_size, delay)
        self.time_synchronizer_2.registerCallback(self.generate_all_bbox)

        self.raycast_hit_logic = RaycastHitDetectionLogic() # 接触検出ロジックのインスタンスを作成

        self.hand_collider_distance = 50  # 手の当たり判定の距離
        self.is_touch = False # 接触していないかどうかのフラグを初期化
        self._id_manager = IdManager() # IdManagerのインスタンスを作成
        
        #----------------------------------------------------------------------------------------------------
        self.from_frame_name = "camera"
        self.to_frame_name = "hololens_origin"
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.pos = PointStamped()
        #----------------------------------------------------------------------------------------------------
    
    def _transform_cube_to_hololens_aabb(self, collider, trans):
        """camera 座標系で AABB として定義された Cube(単位: mm)を、
        hololens_origin 座標系における AABB(単位: m)に変換する。

        手法: 8頂点を全て do_transform_point で変換し、変換後の点群から
        hololens 軸並行の min/max を取って AABB を再構成する。
        camera 軸 と hololens 軸 が非平行な場合、本来の OBB を覆う最小 AABB
        になる(=寸法が若干大きめに出る)が、向きズレ・上下逆転・回転ズレは
        この方法で全て解消する。

        Returns: (x, y, z, width, height, depth) tuple in meters.
        """
        # mm→m に変換しつつ 8 隅を列挙
        x0 = collider.x / 1000.0
        y0 = collider.y / 1000.0
        z0 = collider.z / 1000.0
        w = collider.width / 1000.0
        h = collider.height / 1000.0
        d = collider.depth / 1000.0

        corners_camera = (
            (x0,     y0,     z0    ),
            (x0 + w, y0,     z0    ),
            (x0,     y0 + h, z0    ),
            (x0 + w, y0 + h, z0    ),
            (x0,     y0,     z0 + d),
            (x0 + w, y0,     z0 + d),
            (x0,     y0 + h, z0 + d),
            (x0 + w, y0 + h, z0 + d),
        )

        pt = PointStamped()
        xs, ys, zs = [], [], []
        for cx, cy, cz in corners_camera:
            pt.point.x = cx
            pt.point.y = cy
            pt.point.z = cz
            transformed = tf2_geometry_msgs.do_transform_point(pt, trans)
            xs.append(transformed.point.x)
            ys.append(transformed.point.y)
            zs.append(transformed.point.z)

        aabb_x = min(xs)
        aabb_y = min(ys)
        aabb_z = min(zs)
        return (
            aabb_x,
            aabb_y,
            aabb_z,
            max(xs) - aabb_x,
            max(ys) - aabb_y,
            max(zs) - aabb_z,
        )

    def callback(self, object_list: TrackedObjectList, hit_point: PointStamped, color_img_src: CompressedImage, camera_info: CameraInfo):
        self.get_logger().info('Messages synchronized')
        self.get_logger().info('Received position: x=%f, y=%f, z=%f' % (hit_point.point.x, hit_point.point.y, hit_point.point.z))

        result_list, self.is_touch = self.raycast_hit_logic.execute(object_list, hit_point.point)
        self.get_logger().info(f'is_touch: {self.is_touch}')

        if self.is_touch:
            publish_msg = ContactedList()
            publish_msg.header.stamp = color_img_src.header.stamp
            publish_msg.header.frame_id = camera_info.header.frame_id
            for object_item in result_list:
                tracked_object, _ = object_item
                
                #----------------------------------------------------------------------------------------------------
                trans = self._tf_buffer.lookup_transform(self.to_frame_name, self.from_frame_name, rclpy.time.Time())
                ax, ay, az, aw, ah, ad = self._transform_cube_to_hololens_aabb(tracked_object.collider, trans)
                tracked_object.collider.x = ax
                tracked_object.collider.y = ay
                tracked_object.collider.z = az
                tracked_object.collider.width = aw
                tracked_object.collider.height = ah
                tracked_object.collider.depth = ad
                print({'collider.x': tracked_object.collider.x, 'collider.y': tracked_object.collider.y, 'collider.z': tracked_object.collider.z, 'width': tracked_object.collider.width,
                       'height': tracked_object.collider.height, 'depth': tracked_object.collider.depth})
                #----------------------------------------------------------------------------------------------------
                
                contacted = Contacted()
                contacted.event_id = self._id_manager.new_event_id()
                contacted.people_id = 'null'
                contacted.object_id = tracked_object.object_id
                contacted.action = 'RAYCAST_HIT'
                contacted.people_bounding_box = tracked_object.bounding_box
                contacted.object_bounding_box = tracked_object.bounding_box
                contacted.object_cube = tracked_object.collider
                publish_msg.contacted_list.append(contacted)

                self.get_logger().info('Success to contact detection')
                print(f'ObjectId: {tracked_object.object_id}')

            self._publisher.publish(publish_msg)
            print("Publish Message to Hololens") 
        elif not self.is_touch:
            self.get_logger().info('Failed to contact detection')
            
    def generate_all_bbox(self, object_list: TrackedObjectList, hit_point: PointStamped, color_img_src: CompressedImage, camera_info: CameraInfo):
        self.get_logger().info('execute generate_all_bbox')
        publish_msg = ContactedList()
        publish_msg.header.stamp = color_img_src.header.stamp
        publish_msg.header.frame_id = camera_info.header.frame_id

        for tracked_object in object_list.tracked_object_list:
            trans = self._tf_buffer.lookup_transform(self.to_frame_name, self.from_frame_name, rclpy.time.Time())
            ax, ay, az, aw, ah, ad = self._transform_cube_to_hololens_aabb(tracked_object.collider, trans)
            tracked_object.collider.x = ax
            tracked_object.collider.y = ay
            tracked_object.collider.z = az
            tracked_object.collider.width = aw
            tracked_object.collider.height = ah
            tracked_object.collider.depth = ad
            
            #-------------------------------------------
            debug_msg = Point()
            debug_msg.x = tracked_object.collider.x
            debug_msg.y = tracked_object.collider.y
            debug_msg.z = tracked_object.collider.z

            self.debug_publisher.publish(debug_msg)
            #-------------------------------------------

            contacted = Contacted()
            contacted.event_id = self._id_manager.new_event_id()
            contacted.people_id = 'null'
            contacted.object_id = tracked_object.object_id
            contacted.action = 'RAYCAST_HIT'
            contacted.people_bounding_box = tracked_object.bounding_box
            contacted.object_bounding_box = tracked_object.bounding_box
            contacted.object_cube = tracked_object.collider
            publish_msg.contacted_list.append(contacted)
            
            print(f'ObjectId: {tracked_object.object_id}')
        self._publisher.publish(publish_msg)
        print("Publish Message to Hololens")

def main():
       rclpy.init()
       raycast_hit_detection_node = RaycastHitDetectionNode()
       rclpy.spin(raycast_hit_detection_node)
       raycast_hit_detection_node.destroy_node()
       rclpy.shutdown()

if __name__ == '__main__':
       main()
