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
                self.pos.point.x = tracked_object.collider.x / 1000
                self.pos.point.y = tracked_object.collider.y / 1000
                self.pos.point.z = tracked_object.collider.z / 1000

                transformed_point = tf2_geometry_msgs.do_transform_point(self.pos, trans)
                tracked_object.collider.x = transformed_point.point.x
                tracked_object.collider.y = transformed_point.point.y
                tracked_object.collider.z = transformed_point.point.z 
                tracked_object.collider.width = tracked_object.collider.width / 1000
                tracked_object.collider.height = tracked_object.collider.height / 1000
                tracked_object.collider.depth = tracked_object.collider.depth / 4000
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
            self.pos.point.x = tracked_object.collider.x / 1000
            self.pos.point.y = tracked_object.collider.y / 1000
            self.pos.point.z = tracked_object.collider.z / 1000
            transformed_point = tf2_geometry_msgs.do_transform_point(self.pos, trans)

            tracked_object.collider.x = transformed_point.point.x
            tracked_object.collider.y = transformed_point.point.y
            tracked_object.collider.z = transformed_point.point.z
            tracked_object.collider.width = tracked_object.collider.width / 1000
            tracked_object.collider.height = tracked_object.collider.height / 1000
            tracked_object.collider.depth = tracked_object.collider.depth / 4000
            
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
