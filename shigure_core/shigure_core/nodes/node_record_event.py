import datetime
import os
import threading
from multiprocessing import Process
from typing import List

import cv2
import message_filters
import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
import shigure_core_msgs
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from sensor_msgs.msg import CompressedImage, CameraInfo
from shigure_core_msgs.msg import HeaderString
from shigure_core_msgs.msg import ContactedList, Contacted

from shigure_core.enum.contact_action_enum import ContactActionEnum
from shigure_core.nodes.common_model.bounding_box import BoundingBox
from shigure_core.nodes.node_image_preview import ImagePreviewNode
from shigure_core.nodes.record_event.event import Event
from shigure_core.nodes.record_event.scene import Scene
from shigure_core.util import compressed_depth_util
from shigure_core.db.event_repository import EventRepository
from shigure_core.db.convert_format import ConvertMsg


class SubtractionAnalysisNode(ImagePreviewNode):

    def __init__(self):
        super().__init__('record_event_node')

        # QoS Settings
        shigure_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # ros params
        save_path_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING,
            description='Root path of save images.'
        )
        self.declare_parameter('save_root_path', '/opt/ros2/shigure_core/events', save_path_descriptor)
        self.save_root_path: str = self.get_parameter("save_root_path").get_parameter_value().string_value

        frame_num_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_INTEGER,
            description='Number of save frames before and after.'
        )
        self.declare_parameter('frame_num', 60, frame_num_descriptor)
        self.frame_num: int = self.get_parameter("frame_num").get_parameter_value().integer_value

        camera_id_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_INTEGER,
            description='Number of save frames before and after.'
        )
        self.declare_parameter('camera_id', 1, camera_id_descriptor)
        self.camera_id: int = self.get_parameter("camera_id").get_parameter_value().integer_value

        is_recording_depth_info_descriptor = ParameterDescriptor(type=ParameterType.PARAMETER_BOOL,
                                                       description='If true, record depth images and point clouds.')
        self.declare_parameter('is_recording_depth_info', False, is_recording_depth_info_descriptor)
        self.is_recording_depth_info: bool = self.get_parameter("is_recording_depth_info").get_parameter_value().bool_value

        # ros subscriber
        contacted_subscriber = message_filters.Subscriber(
            self, 
            ContactedList, 
            '/shigure/contacted',
            qos_profile=shigure_qos
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
        color_subscriber = message_filters.Subscriber(
            self, 
            CompressedImage, 
            '/rs/color/compressed',
            qos_profile=shigure_qos
        )
        pose_id_subscriber = message_filters.Subscriber(
            self,
            HeaderString,
            'shigure/current_pose_id',
            qos_profile=shigure_qos
        )

        # 保存先ディレクトリ作成
        os.makedirs(self.save_root_path, exist_ok=True)

        self.time_synchronizer = message_filters.TimeSynchronizer(
            [contacted_subscriber, depth_subscriber, depth_camera_info_subscriber, color_subscriber, pose_id_subscriber], 3000)
        self.time_synchronizer.registerCallback(self.callback)

        self._color_img_buffer = []
        self._depth_img_buffer = []

        self._scene_list: List[Scene] = []

    def callback(self, contacted_list: ContactedList, depth_src: CompressedImage, camera_info: CameraInfo,
                 color_src: CompressedImage, current_pose_id: HeaderString):
        try:
            self.get_logger().info('Buffering start', once=True)

            # FPS計算
            self.frame_count_up()

            color_img: np.ndarray = self.bridge.compressed_imgmsg_to_cv2(color_src)
            depth_img: np.ndarray = compressed_depth_util.convert_compressed_depth_img_to_cv2(depth_src)

            contacted: Contacted
            for contacted in contacted_list.contacted_list:
                # 接触は弾く
                if ContactActionEnum.value_of(contacted.action) == ContactActionEnum.TOUCH:
                    continue

                people_bounding_box = self.convert_bounding_box(contacted.people_bounding_box)
                object_bounding_box = self.convert_bounding_box(contacted.object_bounding_box)
                event = Event(contacted.event_id, contacted.people_id, contacted.object_id, contacted.action, people_bounding_box,
                              object_bounding_box)
                scene = Scene(self.frame_num, camera_info.k.reshape((3, 3)), event,
                              self._color_img_buffer[-self.frame_num:], self._depth_img_buffer[-self.frame_num:])
                self._scene_list.append(scene)

                # json形式でcamera_infoを出力
                event_save_path = os.path.join(self.save_root_path, scene.event.event_id)
                os.makedirs(event_save_path, exist_ok=True)
                ConvertMsg.write_msg_as_json(event_save_path, "camera_info", camera_info)

                # db書き込み
                # ------
                EventRepository.insert_people(contacted.people_id, event_save_path, scene.event.people_bounding_box.width, scene.event.people_bounding_box.height)
                EventRepository.insert_object(
                    contacted.object_id, 
                    event_save_path, 
                    contacted.object_cube.x, 
                    contacted.object_cube.y, 
                    contacted.object_cube.z, 
                    contacted.object_cube.width, 
                    contacted.object_cube.height, 
                    contacted.object_cube.depth, 
                )

                frame_id = str(camera_info.header.frame_id)
                EventRepository.insert_camera(frame_id)

                person_id = EventRepository.select_autoincrement_person_id(contacted.people_id)
                object_id = EventRepository.select_autoincrement_object_id(contacted.object_id)
                camera_id = EventRepository.select_autoincrement_camera_id(frame_id)
                pose_id = current_pose_id.data
                EventRepository.insert_event(contacted.event_id, person_id, object_id, camera_id, pose_id, contacted.action)
                
                color_save_path = os.path.join(event_save_path, 'color')
                depth_save_path = os.path.join(event_save_path, 'depth')
                points_save_path = os.path.join(event_save_path, 'points')
                EventRepository.insert_frame(contacted.event_id, self.frame_num, color_save_path, depth_save_path, points_save_path)
                EventRepository.insert_event_meta(contacted.event_id, camera_info)
                # ------

            # 保存できる状態のシーンを取得
            new_scene_list: List[Scene] = []
            for scene in self._scene_list:
                if scene.is_full():
                    process = Process(target=self.save_scene, args=(scene, self.save_root_path))
                    process.start()
                else:
                    scene.add_frame(color_img, depth_img)
                    new_scene_list.append(scene)



            self._scene_list = new_scene_list

            self._color_img_buffer.append(color_img)
            self._depth_img_buffer.append(depth_img)

            print(f'[{datetime.datetime.now()}] fps : {self.fps}', end='\r')

        except Exception as err:
            self.get_logger().error(err)

    @staticmethod
    def convert_bounding_box(bounding_box: shigure_core_msgs.msg.BoundingBox) -> BoundingBox:
        x, y, width, height = bounding_box.x, bounding_box.y, bounding_box.width, bounding_box.height
        return BoundingBox(int(x), int(y), int(width), int(height))

    def save_scene(self, scene: Scene, save_root_path: str):
        print('保存を開始')

        color_save_path = os.path.join(save_root_path, scene.event.event_id, 'color')
        depth_save_path = os.path.join(save_root_path, scene.event.event_id, 'depth')
        points_save_path = os.path.join(save_root_path, scene.event.event_id, 'points')
        icon_save_path = os.path.join(save_root_path, scene.event.event_id)

        os.makedirs(color_save_path, exist_ok=True)
        os.makedirs(depth_save_path, exist_ok=True)
        os.makedirs(points_save_path, exist_ok=True)

        for i, color_img in enumerate(scene.color_img_list):
            file_path = os.path.join(color_save_path, str(i + 1) + '.png')
            cv2.imwrite(file_path, color_img)

        if self.is_recording_depth_info:
            for i, depth_img in enumerate(scene.depth_img_list):
                file_path = os.path.join(depth_save_path, str(i + 1) + '.png')
                cv2.imwrite(file_path, depth_img)

                height, width = depth_img.shape[:2]
                points = []
                for y in range(height):
                    for x in range(width):
                        s = np.asarray([[x, y, 1]]).T
                        depth = depth_img[y, x]
                        m = (depth * np.matmul(scene.k_inv, s)).T
                        points.append([m[0, 0], m[0, 1], depth])

                file_path = os.path.join(points_save_path, str(i + 1) + '.png')
                np.save(file_path, np.asarray(points))

        color_img_for_icon = scene.color_img_for_icon
        file_path = os.path.join(icon_save_path, 'people_icon.png')
        bb = scene.event.people_bounding_box
        cv2.imwrite(file_path, color_img_for_icon[bb.y:bb.y + bb.height, bb.x:bb.x + bb.width])
        file_path = os.path.join(icon_save_path, 'object_icon.png')
        bb = scene.event.object_bounding_box
        cv2.imwrite(file_path, color_img_for_icon[bb.y:bb.y + bb.height, bb.x:bb.x + bb.width])

        print('保存終了')


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
