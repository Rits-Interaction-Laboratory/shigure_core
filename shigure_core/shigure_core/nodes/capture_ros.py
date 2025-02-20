# coding: utf-8
from itertools import chain
import random
from itertools import chain
import random
import copy
# # カメラキャプチャ＆基本的なフィルタ処理サンプル
import string
import sys
import cv2

import inspect
print(inspect.getfile(cv2))
import numpy as np
# import cupy as np
# print(np.__version__)

import random
import time
import numpy as np
import rclpy
from typing import List

from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.qos import QoSProfile, ReliabilityPolicy,HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
import message_filters

from collections import Counter
from collections import defaultdict

import threading

import sys
# print(sys.path)
# sys.path.append("/home/azuma/ros2_ws/src/shigure_core/shigure_core/shigure_core/nodes")

from shigure_core.nodes.motpy import Detection, MultiObjectTracker

from shigure_core.nodes.motpy.testing_viz import draw_track

from shigure_core_msgs.msg import DetectedObjectList, DetectedObject, TrackedObjectList, TrackedObject, PoseKeyPointsList, Cube
from bboxes_ex_msgs.msg import BoundingBoxes ,Segments, Segment
from shigure_core.nodes.common_model.bounding_box import BoundingBox

from shigure_core.enum.detected_object_action_enum import DetectedObjectActionEnum
from shigure_core.nodes.common_model.timestamp import Timestamp
from shigure_core.nodes.node_image_preview import ImagePreviewNode
from shigure_core.nodes.yolox_object_detection.color_image_frame import ColorImageFrame
from shigure_core.nodes.yolox_object_detection.color_image_frames import ColorImageFrames
from shigure_core.nodes.yolox_object_detection.frame_object import FrameObject
from shigure_core.nodes.yolox_object_detection.judge_params import JudgeParams
from shigure_core.nodes.yolox_object_detection.logic import YoloxObjectDetectionLogic
from shigure_core.nodes.yolox_object_detection.Bbox_Object import BboxObject

from shigure_core.nodes.yolox_object_tracking.tracking_info import TrackingInfo

from shigure_core.nodes.yolox_object_detection.Bbox_Object import BboxObject

import pdb

from ultralytics import YOLO

import os

from datetime import datetime

from scipy.spatial.distance import euclidean
from ultralytics import YOLO

from pathlib import Path

class CaptureNode(ImagePreviewNode):
    def __init__(self):
        super().__init__("yolox_object_traking_node")
        # QoS Settings
        shigure_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        #shigure_qos2 = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history =HistoryPolicy.KEEP_ALL)
        
        #動画用の設定
        # self.video_path = 'qualification_hakkiri_1.mp4'
        self.video_path = './src/shigure_core/demo_movie/siro_kaiten_1.mp4'
        # self.video_path = 'qualification_back_stop_1.mp4'
        print("読み込み")

        
        print(self.video_path)
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            print("動画ファイルを開けませんでした。")
        else:
            print("開けた")
        
        # 出力動画の設定

        # self.output_video_path = "output_front_stop_1_RANSAC.mp4"
        # self.output_video_path = "output2_qualification_back_stop_1.mp4"
        # self.output_video_path = "output2_o6.mp4"
        self.output_video_path = "testo.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 動画のコーデック（例: mp4v）
        fps = self.cap.get(cv2.CAP_PROP_FPS)  # 元の動画のFPS
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # フレームの幅
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # フレームの高さ
        self.out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width, height))




        # publisher, subscriber
        self.detection_publisher = self.create_publisher(
        DetectedObjectList,
        '/shigure/object_detection',
        10
        )
        yolox_bbox_subscriber = message_filters.Subscriber(
            self,
            BoundingBoxes,
            '/bounding_boxes',
            qos_profile = shigure_qos
        )
        people_subscriber = message_filters.Subscriber(
            self,
             PoseKeyPointsList,
            '/shigure/people_detection',
            qos_profile=shigure_qos
        )
        color_subscriber = message_filters.Subscriber(
            self,
            CompressedImage,
            '/rs/color/compressed',
            qos_profile = shigure_qos
        )
        depth_camera_info_subscriber = message_filters.Subscriber(
            self,
            CameraInfo,
            '/rs/color/cameraInfo',
            qos_profile=shigure_qos
        )
        segment_subscriber = message_filters.Subscriber(
            self,
            Segments,
            '/Segments',
            qos_profile=shigure_qos
        )

        




        self.track_list = []
        self.lk_count :int = 0
        # self.revs: np.ndarray = np.ndarray
        self.lk_reset = True

        #ROS subscriber 
        # self.time_synchronizer = message_filters.TimeSynchronizer(
        #     [yolox_bbox_subscriber,color_subscriber,depth_camera_info_subscriber,segment_subscriber], 1000)
       

        self.frame_object_list: List[FrameObject] = []
        #self.start_item_list:List[BboxObject]= []
        self.bring_in_list:List[BboxObject] = []
        self.wait_item_list:List[BboxObject] = []
        self._color_img_buffer: List[np.ndarray] = []
        self._color_img_frames = ColorImageFrames()
        self._buffer_size = 90

        self.take_out_obj_class_id  = string
        self.take_out_people_id = string

        self._judge_params = JudgeParams(200, 5000, 5)
        self._count = 0
        self.check =0
        self.object_id_num:int =0
        self.trackable_objects = {}


        #tracking box
        self.curent_object_dict = {}
        self.feature_box_dict = {}

        self.tracked_objects = {}
        self.max_missing_frames = 10

        self.clipped_images: dict  = {}  # クリップ画像を保存する辞書

     #   self.bbox_item_list = []



        self.previous_object_dict = {}

        self._tracking_info = TrackingInfo()

        self.prvs_box = []
        self.ta_box: tuple[float,float,float,float]  = []

        self.bounding_box: tuple[float, float, float, float] = []

        self.frame_count = 0
        self.add_feature_every_n_frames = 5



        #SIFT setting
        self.FLANN_INDEX_KDTREE = 1
        self.index_params = dict(algorithm = self.FLANN_INDEX_KDTREE, trees = 8)
        self.search_params = dict(checks = 50)
        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)#マッチ検出器の定義
        self.sift = cv2.SIFT_create(nfeatures=0,contrastThreshold=0.01,nOctaveLayers=7,edgeThreshold=20,sigma= 1.4) #SIFT特徴点検出器の定義
        # self.sift = cv2.ORB_create() #SIFT特徴点検出器の定義

        cv2.setNumThreads(10)



        self.comand = "image"

        self.seg_mask= np.zeros((720, 1280))
        self.object_index = 0

        #motpy
        self.model_spec = {'order_pos': 1, 'dim_pos': 2,
                             'order_size': 1, 'dim_size': 2,
                             'q_var_pos': 100, 'r_var_pos': 0.1}
        self.matching_fn_kwargs={
         'min_iou': 0.1,
         'multi_match_min_iou': 0.50}

        self.dt = 0.125
        self.track_id_dict = {}

        self.flow_p =[]

        # self.fe_idelist = {} #特徴点の識別子

        self.first_time = 0


        self.mottracker = MultiObjectTracker(dt=self.dt, model_spec=self.model_spec,matching_fn_kwargs=self.matching_fn_kwargs)


        self._COLORS = np.array(
            [
                0.000, 0.447, 0.741,
                0.850, 0.325, 0.098,
                0.929, 0.694, 0.125,
                0.494, 0.184, 0.556,
                0.466, 0.674, 0.188,
                0.301, 0.745, 0.933,
                0.635, 0.078, 0.184,
                0.300, 0.300, 0.300,
                0.600, 0.600, 0.600,
                1.000, 0.000, 0.000,
                1.000, 0.500, 0.000,
                0.749, 0.749, 0.000,
                0.000, 1.000, 0.000,
                0.000, 0.000, 1.000,
                0.667, 0.000, 1.000,
                0.333, 0.333, 0.000,
                0.333, 0.667, 0.000,
                0.333, 1.000, 0.000,
                0.667, 0.333, 0.000,
                0.667, 0.667, 0.000,
                0.667, 1.000, 0.000,
                1.000, 0.333, 0.000,
                1.000, 0.667, 0.000,
                1.000, 1.000, 0.000,
                0.000, 0.333, 0.500,
                0.000, 0.667, 0.500,
                0.000, 1.000, 0.500,
                0.333, 0.000, 0.500,
                0.333, 0.333, 0.500,
                0.333, 0.667, 0.500,
                0.333, 1.000, 0.500,
                0.667, 0.000, 0.500,
                0.667, 0.333, 0.500,
                0.667, 0.667, 0.500,
                0.667, 1.000, 0.500,
                1.000, 0.000, 0.500,
                1.000, 0.333, 0.500,
                1.000, 0.667, 0.500,
                1.000, 1.000, 0.500,
                0.000, 0.333, 1.000,
                0.000, 0.667, 1.000,
                0.000, 1.000, 1.000,
                0.333, 0.000, 1.000,
                0.333, 0.333, 1.000,
                0.333, 0.667, 1.000,
                0.333, 1.000, 1.000,
                0.667, 0.000, 1.000,
                0.667, 0.333, 1.000,
                0.667, 0.667, 1.000,
                0.667, 1.000, 1.000,
                1.000, 0.000, 1.000,
                1.000, 0.333, 1.000,
                1.000, 0.667, 1.000,
                0.333, 0.000, 0.000,
                0.500, 0.000, 0.000,
                0.667, 0.000, 0.000,
                0.833, 0.000, 0.000,
                1.000, 0.000, 0.000,
                0.000, 0.167, 0.000,
                0.000, 0.333, 0.000,
                0.000, 0.500, 0.000,
                0.000, 0.667, 0.000,
                0.000, 0.833, 0.000,
                0.000, 1.000, 0.000,
                0.000, 0.000, 0.167,
                0.000, 0.000, 0.333,
                0.000, 0.000, 0.500,
                0.000, 0.000, 0.667,
                0.000, 0.000, 0.833,
                0.000, 0.000, 1.000,
                0.000, 0.000, 0.000,
                0.143, 0.143, 0.143,
                0.286, 0.286, 0.286,
                0.429, 0.429, 0.429,
                0.571, 0.571, 0.571,
                0.714, 0.714, 0.714,
                0.857, 0.857, 0.857,
                0.000, 0.447, 0.741,
                0.314, 0.717, 0.741,
                0.50, 0.5, 0
            ]
        ).astype(np.float32).reshape(-1, 3)
        # self.model = YOLO("yolov8n-seg.pt")
        #self.inactive_counter = {}
        # 全体画像の保存
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_folder = f"matched_images_{current_time}"


        current_time2 = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_folder2 = f"matched_images_{current_time}_second"
        

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        if not os.path.exists(self.output_folder2):
            os.makedirs(self.output_folder2)

        self.prv_des ={}

        self.prev_matchid =[]

        self.yolo11_model = YOLO("yolo11x-seg.pt")
    

    def process_video(self):
        try:

            while self.cap.isOpened():
                # 動画からフレームを取得
                ret, frame = self.cap.read()


                if not ret:
                    print("動画の最後に到達しました。")
                    break

                # try:

                    # フレームの処理（ここでカスタム処理を記述）
                processed_frame = self.callback(frame)
                # except Exception as e:

     

                # 処理結果を表示（必要に応じて保存なども可能）
                self.out.write(processed_frame)


                key = cv2.waitKey(1) & 0xFF

                # ESCキーで終了
                if  key ==ord('q'):  # 27 は ESC キー
                    print("処理を中断しました。途中までの動画を保存します...")

                    break
        except KeyboardInterrupt:
            print("\nCtrl+C が押されました。処理を中断し、途中までの動画を保存します...")
        finally:

            # 終了処理
            self.cap.release()
            self.out.release()
            cv2.destroyAllWindows()
            print(f"処理後の動画を保存しました: {self.output_video_path}")

    #rosノードとするときにのコールバック
    #def callback(self, yolox_bbox_src: BoundingBoxes,color_img_src: CompressedImage, camera_info: CameraInfo, segment_src:Segments):
    def callback(self, color_img_src):
        self.get_logger().info('Buffering start', once=True)
        self.frame_count_up()

        # print(color_img_src.shape[:2])

        color_img = color_img_src

        # color_img: np.ndarray = self.bridge.compressed_imgmsg_to_cv2(color_img_src)
        height, width = color_img.shape[:2]

        self.img_size = (height,width)
        

        results = self.yolo11_model(color_img)

        # result_img = results[0].plot()
        self.detect = color_img.copy()

        masks = results[0].masks.cpu().numpy().data #マスク領域抽出

        self.masks_shaped = []
            
        for mask in masks:
            self.masks_shaped.append(cv2.resize(mask,(int(self.detect.shape[1]),int(self.detect.shape[0]))))

        for mask in self.masks_shaped:
            the_mask = mask.copy()
            the_mask = np.stack([the_mask] * 3,axis=-1)
            color = (255,0,0)
            self.detect[the_mask[:, :, 0] > 0.5] = self.detect[the_mask[:, :, 0] > 0.5] * 0.5 + np.array(color) * 0.5

    
        #動画デバック用YOLO11モデル、ROSデバックのときはいらない
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        class_names = self.yolo11_model.names
        img_header = ""

        masks = result.masks

        def create_msg(bboxes, scores, cls, cls_names, img_header, mask_data):
            segments = Segments()
            
            i = 0
            brack_img = np.zeros((720,1280))
            for bbox in bboxes:
                one_box = Segment()
                # if < 0
                if bbox[0] < 0:
                    bbox[0] = 0
                if bbox[1] < 0:
                    bbox[1] = 0
                if bbox[2] < 0:
                    bbox[2] = 0
                if bbox[3] < 0:
                    bbox[3] = 0
                one_box.xmin = int(bbox[0])
                one_box.ymin = int(bbox[1])
                one_box.xmax = int(bbox[2])
                one_box.ymax = int(bbox[3])

                mask = mask_data[i].xy[0]
    
                mask =  np.array(mask, dtype=np.int32)

                one_box.x_masks = mask[:, 1].astype(np.int32).tolist()  # x座標
                one_box.y_masks = mask[:, 0].astype(np.int32).tolist()  # y座標

                one_box.probability = float(scores[i])
                one_box.class_id = str(cls_names[int(cls[i])])
                segments.segments.append(one_box)

                i = i+1

            return segments

        
        #検出BOX情報
        
        segment_src = create_msg(boxes, scores, class_ids, class_names, img_header,masks)

        # print("boxes",len(results))

        #ROS2用メッセージのアイコン画像設定
        if not hasattr(self, 'object_list'):
            self.object_list = []
            black_img = np.zeros_like(color_img)
            for i in range(4):
                self.object_list.append(cv2.resize(black_img.copy(), (width // 2, height // 2)))

        self._color_img_buffer.append(color_img)
        if len(self._color_img_buffer)> 1:
            self._color_img_buffer.pop(-1)
        
        #動画デバック用タイムスタンプ
        timestamp = Timestamp(0,0)

        ##ROS2用##
        # timestamp = Timestamp(color_img_src.header.stamp.sec, color_img_src.header.stamp.nanosec)


        # frame = ColorImageFrame(timestamp, self._color_img_buffer[0], color_img) #bufferの先頭の画像と新しい画像
        # # self._color_img_frames.add(frame) #ColorImageFrameslistの更新して、listに追加

        #これは検出ノードの名残です

        # frame_object_dict,bring_in_list,wait_item_list,people_item_list,take_out_people_id,take_out_obj_class_id = self.yolox_object_detection_logic.execute(yolox_bbox_src, timestamp,people,color_img,self.frame_object_list,self._judge_params,self.take_out_people_id ,self.take_out_obj_class_id ,self.bring_in_list,self.wait_item_list)

        # #if self._count == 0:
        #     #self.start_item_list = start_item_list
        # self.bring_in_list = bring_in_list
        # self.wait_item_list = wait_item_list
        # self.people_item_list = people_item_list
        # self.take_out_people_id = take_out_people_id
        # self.take_out_obj_class_id = take_out_obj_class_id
        # #count = 1
        # #self._count = count
        # self.frame_object_list = list(chain.from_iterable(frame_object_dict.values())) #frame_object_dictをすべて取り出し

        # sec, nano_sec = frame.timestamp.timestamp
        # detected_object_list = DetectedObjectList()
        # detected_object_list.header.stamp.sec = sec
        # detected_object_list.header.stamp.nanosec = nano_sec
        # detected_object_list.header.frame_id = camera_info.header.frame_id
        ##ROS2用##


        # if self.frame_object_list:
        #     detected_object_list = self.create_msg(self.frame_object_list, detected_object_list, frame)

        # self.detection_publisher.publish(detected_object_list)
        #result = color_img.copy()


        self.get_logger().info('Buffering end', once=True)




        # lk勾配法の特徴点の更新間隔

        # if self.lk_count > 2:
        #     self.lk_reset = True
        #     self.lk_count =0

        # elif self.lk_count == 0 :
        #     self.lk_reset = True

        # else:
        #     self.lk_reset = False


        #self.lk_count = self.lk_count + 1
        
        #実行用関数設定
        proc = "lktracking"
        # proc = "color"
        # proc = "sift"
        ndi_send = None

        # ret,img = cap.read()
        ret = True

        img = color_img
        
        #特徴点が少なくなったときのフラグ
        self.point_relia = False
        uncertain_flag = False



        # height = img.shape[0]
        # width = img.shape[1]
        # img = cv2.resize(img,((int)(width/2),(int)(height/2)))

        #self.prvs = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)



        profile_face = True

        #
        ## for LK tracking
        #
        #ShiTomasiコーナー検出器のためのパラメータ
        lk_fnum = 1000
        lk_fnum2 = 3000
        feature_params = dict( maxCorners = lk_fnum,
                            qualityLevel = 0.001,
                            minDistance = 3,
                            blockSize = 8 )
        feature_params2 = dict( maxCorners = lk_fnum2,
                    qualityLevel = 0.001,
                    minDistance = 2,
                    blockSize = 7 )
        # Lucas-Kanade法によるオプティカル・フローのためのパラメータ
        lk_params = dict( winSize  = (31,31),
                        maxLevel = 6,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))



        #
        ## for histogram equalization
        #
        self.new_id = 0
        self.brack_img = self.seg_mask.copy()



        #
        ## main loop
        #

    

        #デバック用コマンド切り替え
        self.key = cv2.waitKey(1) & 0xFF

        
        if self.key == ord('a'):
            self.comand = "flow"
        elif self.key == ord('s') :
            self.comand = "sift"
        elif self.key == ord('r') :
            self.comand = "image"
        elif self.key == ord('n') :
            self.comand = "new_point"
        elif self.key == ord('m') :
            self.comand = "old_point"
        elif self.key == ord('l') :
            self.comand == "line"
        elif self.key == ord('t') :
            self.comand == "add_point"
        


        # if(ret==True):
        #     if key == ord('q') :
        #         print("a")
        #         # break
        #     elif key == ord('g') : # Monocolor
        #         proc = "gray"
        #     elif key == ord('c') : # Color
        #         proc = "color"


        # image processing
        if proc == "color" :
            result = img
            
        elif proc == "lktracking" : #追跡処理
        
            print("フレーム",self.frame_count)

            self.fe_idelist = {} #特徴点の識別子
            # tracemalloc.start()
            
            mot = MOT()
            
            #グレースケール画像
            next = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # result = img.copy()
            #self.frame_count += 1
            #print(self.frame_count)

            bbox_item_list = []
            self.mot_item_list = {}

            self.id  = []
            self.match = {}
            box_id =[]

            self.testmask = img.copy()
            # self.detect = img.copy()

            self.flow = img.copy()
            self.matchkline = img.copy()
 
            if( self.lk_reset == True ):#最初のフレーム特徴抽出
                if self._count == 0:
                    self.prvs = next

                    self.matchkline2 = self.matchkline


                    self._count = self._count + 1

                # if len( self.track_list)> 10:
                #     self.track_list = self.track_list[5:]
                yolo_segments = segment_src.segments
                # yolo_segments = filter_boxes(yolo_segments)
                self.bbox_count = 0

                mask = np.zeros_like(next)


                for id, bbox in enumerate(yolo_segments):
                    probability = bbox.probability
                    x = bbox.xmin
                    y = bbox.ymin
                    xmax = bbox.xmax
                    ymax = bbox.ymax
                    height = ymax - y
                    width = xmax - x
                    class_id = bbox.class_id
                    x_masks = bbox.x_masks
                    y_masks = bbox.y_masks

                    mask_pairs = list(zip(y_masks, x_masks))

                    m_points = np.array(mask_pairs, dtype=np.int32)

                    box_info= [x,y,xmax,ymax]









                    object_id = ""
                    
                    #検出物体のソート処理
                    if is_unknown_object(class_id, probability,box_info,img_size=self.img_size) and height < 700:
                        print("1frame")


                        mask = cv2.drawContours(mask, [m_points], -1, 255, thickness=cv2.FILLED)

                        brack_img = self.seg_mask.copy()

                        mask_img = cv2.fillPoly(brack_img, [m_points], 255)  # 255は白の値

                        # BBOX(左上端座標, 幅, 高さ)
                        bounding_box = BoundingBox(x, y, width, height)
                        area = width*height # BBOXの面積

                        object_id = self.bbox_count

                        test_item = [x,y, width, height]

                        box_id.append(id)



                        bbox_item = BboxObject(bounding_box, area, mask_img, timestamp,class_id,object_id)
                        bbox_item_list.append(bbox_item)


                        bbox_item2 = [x,y, width, height, probability,class_id]
                        self.mot_item_list[id]=bbox_item2

                        self.match[object_id] = str(object_id)



                        #Track用box辞書
                        self.curent_object_dict[object_id] = bbox_item
                        self.bbox_count += 1

                        #print(object_id)


                        #test_item = [x,y,xmax,ymax]


                self.first_mask = mask #cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)


                distance = 20  # 削る距離（ピクセル単位）
                kernel = np.ones((distance, distance), np.uint8)  # 距離に基づくカーネル

                self.first_mask= cv2.erode(self.first_mask, kernel, iterations=1)

                self.keypoints, self.descriptors = self.sift.detectAndCompute(img, mask = self.first_mask)
                # cv2.imwrite("test_mask.png", self.first_mask)
                p0 = np.array([kp.pt for kp in self.keypoints], dtype=np.float32).reshape(-1, 1, 2)
                # print(p0)



                self.lk_reset = False

            else :
                p0 = self.good_new.reshape(-1,1,2)
                #print(p0)


            # オプティカル・フローを計算

            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prvs, next, p0, None, **lk_params)

            #print(p1)

            print(len(p1))


            if self.frame_count == 1 :
                self.result = img.copy()
                # 良い特徴点を選択
                self.good_new = p1[st==1]
                self.good_old = p0[st==1]
                track = np.zeros_like(img)

                def lktracking():#オプティカルフローを使った特徴点記録


                    # # 良い特徴点を選択
                    # self.good_new = p1[st==1]
                    # self.good_old = p0[st==1]

                    # yolox_bboxes = yolox_bbox_src.bounding_boxes #yolox-rosから受け取った物体集合から物体一つずつ取り出す
                    # yolox_bboxes = filter_boxes(yolox_bboxes)
                    # self.bbox_count = 0

                    yolo_segments = segment_src.segments


                    feature_list = []
                    box_flow = []
                    box_flow2 = []
                    points_in_mask = []
                    fe_ide = []

                    self.addd = img.copy()

                    

                    for i,(new,old) in enumerate(zip(self.good_new,self.good_old)):
                        total_overlap_count = 0
                        track_id = []

                        a,b = new.ravel()
                        new_point = a,b #+211
                        c,d = old.ravel()

                        old_point = c,d #+211
                        colar =(0,255,0)




                        for key, b_item in self.curent_object_dict.items() :
                            #print(b_item)
                            bounding_box_src = b_item._bounding_box
                            x, y, width, height = bounding_box_src.items
                            reactangle = [x,y, width, height]


                            brack_img = b_item._mask


                            # if is_point_inside_bounding_box(new_point, reactangle):
                            #特徴点抽出処理
                            if brack_img[int(b), int(a)] == 255:

                                track_id.append(key)

                                # if len(box_flow) < 20:
                                #     box_flow.append(new_point)

                                # box_flow.append(new_point)

                        






                        if len(track_id) > 0 :
                            # cv2.line(track, (int(a),int(b)),(int(c),int(d)), colar, 2)
                            # result = cv2.circle(result,(int(a),int(b)),5,colar,-1)
                            feature_list.append([new_point,old_point,track_id])


                            if track_id[0] in self.fe_idelist:#カルマンフィルタの計算負荷軽減のため、特徴点を20個選択

                                if len(self.fe_idelist[track_id[0]]) < 20:
                                    self.fe_idelist[track_id[0]].append([new_point])

                            else:
                                self.fe_idelist.setdefault(track_id[0], [[new_point]])


                        else:#物体領域外の特徴
                            fe_ide.append(i)


                    # print("feature_list")
                    # print(feature_list)


                    feature = {"feature": feature_list}

                    bbox = {"bbox": self.curent_object_dict}

                    self.feature_box_dict.setdefault(self.frame_count, [feature])


                    self.feature_box_dict[self.frame_count].append(bbox)

                    self.inactive_counter =[0] * len(self.feature_box_dict[self.frame_count][0]["feature"])  # 各特徴点のカウンターをリストで初期化


                    print(len(self.fe_idelist))
                    print(len(self.mot_item_list))
                    # cv2.imwrite("rrrrrr.png",self.addd)


                    for bo_id, mot_item in enumerate(self.mot_item_list.values()):
                        if bo_id in self.fe_idelist:

                            box_flow = self.fe_idelist[int(bo_id)]
                            mot_item.insert(5, box_flow)
                        else:
                            box_flow = []
                            mot_item.insert(5, box_flow)


                    #物体領域外の特徴点を削除
                    for del_fe in sorted(fe_ide,reverse=True):
                        self.good_new= np.delete(self.good_new, del_fe, axis=0)
                        self.good_old= np.delete(self.good_old, del_fe, axis=0)





                    # mot

                def sift1(): #SIFTを使った特徴点記録
                    yolo_segments = segment_src.segments
                 
                    self.mask =self.first_mask

                    result = img

                    boxes = []
                    self.countf= 0

                    self.match_sift ={}

                    self.matched_images = []

                    self.matched_images2 = []

                    alive_frame = []

                    #検出Boxごとの特徴を記録
                    for id, bbox in enumerate(yolo_segments ):
                        probability = bbox.probability
                        x = bbox.xmin
                        y = bbox.ymin
                        xmax = bbox.xmax
                        ymax = bbox.ymax
                        height = ymax - y
                        width = xmax - x
                        class_id = bbox.class_id
                        # object_id = ""
                        points_des = []
                        box_point = []
                        box_des = []
                        box_info= [x,y,xmax,ymax]

                        if is_unknown_object(class_id, probability,box_info,img_size=self.img_size) and height < 700 and id in box_id:
                            boxes.append([x, y, xmax, ymax])
                            clipped_image = img[y:ymax, x:xmax]


                            for i, val in enumerate(zip(self.keypoints,self.descriptors)):
                                kp, des = val
                                # 特徴点がボックス内にあるか確認
                                if x <= kp.pt[0] <= xmax and y <= kp.pt[1] <= ymax:
                                    colar =get_id_color(self.countf)

                                    kx, ky = int(kp.pt[0]), int(kp.pt[1]) 
                                    # result = cv2.circle(result,(int(kp.pt[0]),int(kp.pt[1])),5,colar,3)

                                    if self.descriptors is not None:#特徴点をBoxの相対座標で記録
                                        new_kp = cv2.KeyPoint(kx - x, ky - y, kp.size)
                                        box_point.append(new_kp)
                                        box_des.append(self.descriptors[i])
                                        alive_frame.append(1)






                            # self.clipped_images[self.countf] = {
                            #     "keypoints": keypoints_in_box,
                            #     "descriptors": np.array(descriptors_in_box) if descriptors_in_box else None,
                            #     "box": (x, y, width, height),
                            # }
                            # print("nani",len(points_des))

                            if not (box_point ==[] or box_des ==[]):

                                points_des =(box_point,box_des,class_id,box_info,alive_frame)

                                self.clipped_images.setdefault(self.countf, points_des)

                                self.match_sift[id] = self.countf



                            self.countf +=1
                    print("self.sift_match",self.match_sift )

            


                #SIFTとオプティカルフローのスレット別処理
                thread1 = threading.Thread(target= lktracking)
                thread2 = threading.Thread(target= sift1)
                thread1.start()
                thread2.start()


                thread1.join()
                thread2.join()






                print(self.mot_item_list)
                #カルマンフィルタの入力用のデータ作成
                motdetections = mot.bboxes2out_detections(self.mot_item_list)
                #print(motdetections)

                #カルマンフィルタの処理
                self.mottracker.step(motdetections, self.match)
                mottracks = self.mottracker.active_tracks(min_steps_alive=1)
                #print(len(mottracks))






                #カルマンフィルタの結果
                for track_result in mottracks:
                    if track_result.id not in self.track_id_dict:
                        new_id = len(self.track_id_dict)
                        self.track_id_dict[track_result.id]= new_id
                result = mot.draw_debug(self.result,mottracks,self.track_id_dict)

                # print(fe_ide)
                # print(self.good_new.shape[0])

                # for del_fe in sorted(fe_ide,reverse=True):
                #     self.good_new= np.delete(self.good_new, del_fe, axis=0)
                #     self.good_old= np.delete(self.good_old, del_fe, axis=0)








                #print(self.feature_box_dict)

            else:#2フレーム以降の処理
        
                print("開始")
                # print(img.shape)
                # result = img.copy()


                # 良い特徴点を選択

                # print(len(self.good_new) )
                # print(len(self.feature_box_dict[self.frame_count -1][0]["feature"]))

                self.good_new = p1#[st==1]
                self.good_old = p0#[st==1]

                track_list = []
                feature_list = []



                # print(len(self.good_new))

                #print(self.feature_box_dict[self.frame_count -1][0]["feature"])
                # print(len(self.feature_box_dict[self.frame_count -1][0]["feature"]))
                yolo_segments = segment_src.segments
                track = np.zeros_like(img)


                #オプティカルフローによる特徴点登録処理
                def lktracking2():
                    self.result = img.copy()
                    if len(self.good_new) != len(self.feature_box_dict[self.frame_count -1][0]["feature"]):
                        print("特徴の数が違う")
                        print(len(self.good_new))
                        #print(len(self.good_old))
                        print(len(self.feature_box_dict[self.frame_count -1][0]["feature"]))



                    #前のフレームと情報が間違っていないかの確認
                    for i,(new,old,feature) in enumerate(zip(self.good_new,self.good_old, self.feature_box_dict[self.frame_count -1][0]["feature"])):

                        total_overlap_count = 0



                        #print(new)

                        a,b = new.ravel()
                        point = a,b #+211

                        #print(a)
                        c,d = old.ravel()

                        point2 = c,d #+211
                        colar =(0,255,0)

                    

                        if int(feature[0][0]) == int(c) and int(feature[0][1]) == int(d) :
                            track_list = feature[2]
                        else:
                            print("not kaman match")


                            print(int(feature[0][0]))
                            print(int(c))
                            print(int(feature[0][1]))
                            print(int(d))

                        # if int(feature[0][0]) != int(c):

                        #     print("違う特徴")
                        #     print(int(feature[0][0]))
                        #     print(int(c))

                        # if point in p1[st==0]:
                        #     print("含まれる")
                        #     # feature_list.append([point,point2,track_list])
                        # else:

                        feature_list.append([point,point2,track_list])



                    self.feature_box_dict.setdefault(self.frame_count, [{"feature": feature_list}])



                    self.ID_point_count_dict = {}#物体領域内の特徴数を記録する辞書
                    # track = np.zeros_like(img)
                    #print(yolox_bboxes)

                    lost_box = {}


                    print(color_img.shape[:2])
                    mask_img = np.zeros(img.shape[:2])


                    self.mask_box= {}

                    print("detect_count",len(yolo_segments))


                    for num, bbox in enumerate(yolo_segments):#yolox_bboxes
                        probability = bbox.probability
                        x = bbox.xmin
                        y = bbox.ymin
                        xmax = bbox.xmax
                        ymax = bbox.ymax
                        height = ymax - y
                        width = xmax - x
                        class_id = bbox.class_id
                        #object_id = ""
                        object_id = num

                        x_masks = bbox.x_masks
                        y_masks = bbox.y_masks
                        area = width*height # BBOXの面積


                        point_counts = {}
                        total_counts = {}

                        reactangle = [x,y, width, height]

                        count = 0

                        box_flow =[]
                        points_in_mask = []

                        box_info= [x,y,xmax,ymax]






                        if is_unknown_object(class_id, probability,box_info,img_size=self.img_size) and height < 700 :
                            # count +=1
                            print(class_id)
                            #print(height)
                            mask_pairs = list(zip(y_masks,x_masks))


                            brack_img =  self.seg_mask.copy()


                            bounding_box = BoundingBox(x, y, width, height) # BBOX(左上端座標, 幅, 高さ)

                            #print(self.feature_box_dict[self.frame_count][0]["feature"])
                            m_points = np.array(mask_pairs, dtype=np.int32)


                            # 輪郭を描いてその範囲を白塗り
                            cv2.fillPoly(brack_img, [m_points], 255)  # 255は白の値

                            # distance = 10  # 削る距離（ピクセル単位）
                            # kernel = np.ones((distance, distance), np.uint8)  # 距離に基づくカーネル


                            # brack_img= cv2.erode(brack_img, kernel, iterations=1)

                            # cv2.fillPoly(self.brack_img, [m_points], 255)

                            self.mask_box[num] = brack_img



                            # cv2.imshow("demo",self.brack_img)

                            #検出領域内の特徴点を数える
                            for values in self.feature_box_dict[self.frame_count][0]["feature"] :
                                #print(values)

                                new_point, old_point, trackid = values

                                #print(trackid)
                                a,b = new_point
                                c,d = old_point


                                contour = m_points.reshape((-1, 1, 2)).astype(np.int32)



                                if 0 <= a < color_img.shape[1] and 0 <= b < color_img.shape[0]:  # 範囲チェック

                                    if brack_img[int(b), int(a)] == 255:
                                        # print("マスク内")
                                        
                                        # for id in trackid:
                                        #マスク領域内なら、１カウント
                                        if trackid[0] in  total_counts:
                                            total_counts[trackid[0]] += 1
                                            #print(id)

                                        else:

                                            total_counts.setdefault(trackid[0], 0)
                                            total_counts[trackid[0]] += 1


    

                            # print("各detectboxごとのTack_featureの数")
                            print(total_counts)
                            if  total_counts == {}:
                                self.point_relia = True
                           
                            self.ID_point_count_dict.setdefault(num, []).append(total_counts)

                            total = sum(total_counts.values())


                            bounding_box = BoundingBox(x, y, width, height) # BBOX(左上端座標, 幅, 高さ)


                            bbox_item = BboxObject(bounding_box, area, brack_img, timestamp,class_id,object_id)

                            bbox_item2 = [x,y, width, height, probability,class_id]
                            
                            self.mot_item_list[num]=bbox_item2

                            bbox_item_list.append(bbox_item)

                    # brack_img = np.zeros(img.shape[:2])
                    print("aho")
                    print(len(self.feature_box_dict[self.frame_count][0]["feature"]))
                    print(len(self.good_new))
                    # cv2.imshow("demo5",self.addd)

                    # cv2.imwrite("test_5.png", self.ad2)
                    # cv2.imwrite("test_5.png", self.addd)
                    # cv2.imshow("demo6",self.ad2)
                    # cv2.imshow("demo",self.brack_img)

                    

                    # 削除対象のインデックスを記録するリスト
                    indices_to_remove = []



                    fe_ide = []




                    print("各detectboxごとのTack_featureの数")
                    print(self.ID_point_count_dict)
                    print(self.mot_item_list)


                    # 空の辞書を持つキーを取り出す
                    empty_keys = [key for key, value in self.ID_point_count_dict.items() if value == [{}]]



                    self.match = {}

                    print("結果")
                    self.new_key = empty_keys



                    #物体領域内の特徴点数でID割当て
                    # Step 1: Calculate the total for each inner_key
                    totals = {}
                    for outer_key, inner_list in self.ID_point_count_dict.items():

                        for inner_dict in inner_list:

                            for inner_key, inner_value in inner_dict.items():
                                if inner_key not in totals:
                                    totals[inner_key] = 0
                                totals[inner_key] += inner_value


                    # Step 2: Collect all potential assignments with their ratios
                    potential_assignments = {}

                    for outer_key, inner_list in self.ID_point_count_dict.items():

                        if inner_list == [{}] :
                            potential_assignments[outer_key] = []
                            potential_assignments[outer_key].append("None")

                        for inner_dict in inner_list:
                            for inner_key, inner_value in inner_dict.items():
                                if outer_key not in potential_assignments:
                                    potential_assignments[outer_key] = []

                                #特徴点を追加するかの判定
                                if (totals[inner_key] < 50 and  max(inner_dict.values()) == inner_value) :
                                    self.new_key.append(outer_key)
                                    self.point_relia = True

                                if totals.get(inner_key, 0) < 1:
                                    continue


                                ratio = inner_value / totals[inner_key]

                                potential_assignments[outer_key].append((ratio, inner_key, inner_value))

  

                    # Step 3: Select the best assignments based on ratios
                    assignments = {}
                    # print(potential_assignments.items())

                    for outer_key, values in potential_assignments.items():
                        assignments[outer_key] = []
                        if values == ["None"]:
                            assignments[outer_key].append((0, "None", "None", 0))
                        else:
                            for ratio, inner_key, inner_value in values:
                                total_inner_values = sum(inner_dict[inner_key] for inner_dict in self.ID_point_count_dict[outer_key])
                                ratio2 = inner_value / total_inner_values
                                assignments[outer_key].append((ratio, inner_key, inner_value, ratio2))




                    # Function to find the best assignment

                    def find_best_assignment(values):
                        # print(values)
                        ratio, inner_key, inner_value, ratio2 = values
                        candidates = [inner_key]

                        # candidates = [inner_key for ratio, inner_key, inner_value, ratio2 in values ]#if ratio >= 0.1 and ratio2 >= 0.1
                        # print(candidates)

                        if len(candidates) > 1:


                            return " or ".join(map(str, candidates))
                        if candidates:
                            return str(candidates[0])
                        else:
                            candidates = [inner_key for ratio, inner_key, inner_value, ratio2 in values ]
                            if candidates:
                                return str(candidates[0])
                            else:

                                return None


                    # 1. 各キーに対して最大値を選択
                    max_assignments = {
                        key: max(values, key=lambda x: x[2])
                        for key, values in assignments.items()
                    }

                    # 2. 同じ2番目の要素を持つタプルを比較
                    # 2番目の要素でグループ化
                    grouped = {}
                    for key, value in max_assignments.items():
                        second_elem = value[1]
                        if second_elem not in grouped:
                            grouped[second_elem] = []
                        grouped[second_elem].append((key, value))

                    # 3. 変更の適用
                    for group in grouped.values():
                        if len(group) > 1:
                            # 3番目の要素を比較して最大のキーを決定
                            max_key = max(group, key=lambda x: x[1][2])[0]  # 3番目の要素で比較
                            # 他のキーの要素を変更
                            for key, value in group:
                                if key != max_key:
                                    max_assignments[key] = (0, 'None', 'None', 0)
                                else:
                                    # 最大のものを保持
                                    max_assignments[key] = value

                    # 最終的な結果に最大のタプルと[(0, 'None', 'None', 0)]を持つキーを保持
                    final_assignments = {key: value for key, value in max_assignments.items()}

        

                    for outer_key, values in final_assignments.items():
                        assigned_inner_key = find_best_assignment(values)
                        # print( assigned_inner_key)
                        if assigned_inner_key is not None:
                            self.match[outer_key] = assigned_inner_key

                    #print(self.match)

                    self.match = resolve_conflicts(self.match)





                    #print(self.match)

                    self.id = list(self.match.values())
                    #print( self.id)


                    counta = 0

                    # # 割り当てられていないキーを探し出して結果の辞書に追加
                    # for key in self.ID_point_count_dict.keys():
                    #     if key not in self.match:
                    #         print("ssssssssssssssssssssd")
                    #         point_relia = True
                    #         counta += 1
                    #         #print( counta)
                    #         #print(self.curent_object_dict)
                    #         n_id =len(self.curent_object_dict)+ counta
                    #         self.match[key] = str(n_id)
                    #         self.new_key.append(key)
                    #         print(key)
                    #          # ここでは値をNoneに設定


                    # print(self.match)

                    # self.match = dict(sorted(self.match.items(), key=lambda item: item[1]))


                    print(self.match)

                    #self.mot_item_list をId順に並べる　

                    print("数")

                    print(len(self.feature_box_dict[self.frame_count][0]["feature"]))
                    mean = []

                    distance = 10  # 削る距離（ピクセル単位）
                    kernel = np.ones((distance, distance), np.uint8)  # 距離に基づくカーネル


                    self.brack_img= cv2.erode(self.brack_img, kernel, iterations=1)




                    # カルマンフィルタの入力用の特徴点と対応がつかない特徴点のメンテナンス
                    for idx,values in enumerate(self.feature_box_dict[self.frame_count][0]["feature"]):
                    
                        new_point, old_point, trackid = values



                        #print(trackid)
                        a,b = new_point
                        c,d = old_point


             

                        box_flow2 = []

                        if 0 <= int(a) < self.brack_img.shape[1] and 0 <= int(b) < self.brack_img.shape[0]:  # 範囲チェック


                            if self.brack_img[int(b), int(a)] == 255:

                                # distances = abs(cv2.pointPolygonTest(contour, (a,b), True))


                                if len(trackid) > 0:

                                    if trackid[0] in self.fe_idelist:
                                        if len(self.fe_idelist[trackid[0]]) < 20:
                                            self.fe_idelist[trackid[0]].append([new_point])

                                    else:
                                        self.fe_idelist.setdefault(trackid[0], [[new_point]])

                                    txt_bk_color = get_id_color(int(trackid[0]))
                                    self.flow = cv2.circle(self.flow,(int(a),int(b)),5,txt_bk_color,2)
                                    # cv2.line(track, (int(a),int(b)),(int(c),int(d)), txt_bk_color, 2)



                                if len(self.inactive_counter) < idx+1:
                                    self.inactive_counter.append(0)
                                else:
                                    self.inactive_counter[idx] = 0

                                # print(self.feature_box_dict[self.frame_count][0]["feature"][idx])



                            elif len(trackid) > 0:
                                # print(len(self.inactive_counter))
                                # print(idx)

                                if len(self.inactive_counter) < idx+1:
                                    self.inactive_counter.append(0)

                                self.inactive_counter[idx] += 1





                                # # 20フレーム連続してマスク外にいる場合は辞書から削除 今は1フレーム
                                if self.inactive_counter[idx] > 1  :
                                    self.inactive_counter[idx] =0

                                    if trackid[0] in self.fe_idelist:
                                        if len(self.fe_idelist[trackid[0]]) < 20:
                                            # print(trackid[0])
                                            self.fe_idelist[trackid[0]].append("None")


                                        
                                            mean.append(trackid[0])


                                            indices_to_remove.append(idx)


                                    else:
                                        self.fe_idelist.setdefault(trackid[0], ["None"])
                                        mean.append(trackid[0])


                                        indices_to_remove.append(idx)

                                else:
                                    if trackid[0] in self.fe_idelist:
                                        if len(self.fe_idelist[trackid[0]]) < 20:
                                            self.fe_idelist[trackid[0]].append([new_point])
                                    else:
                                        self.fe_idelist.setdefault(trackid[0], [[new_point]])


                                # txt_bk_color = get_id_color(int(trackid[0]))
                                # self.flow = cv2.circle(self.flow,(int(a),int(b)),5,txt_bk_color,-1)


          

                        elif len(trackid) > 0:
            

                            if len(self.inactive_counter) < idx+1:
                                self.inactive_counter.append(0)

                            self.inactive_counter[idx] = 0

                            if trackid[0] in self.fe_idelist:

                                if len(self.fe_idelist[trackid[0]]) < 20:
                                    self.fe_idelist[trackid[0]].append("None")
                            else:
                                self.fe_idelist.setdefault(trackid[0], ["None"])

                            mean.append(trackid[0])

                            # print(trackid[0])


                            indices_to_remove.append(idx)




                    # print(indices_to_remove)
                    indices_to_remove = list(set(indices_to_remove))
                    indices_to_remove = sorted(indices_to_remove, reverse=True)
                    print(indices_to_remove)

                    #マスク外だった特徴点を辞書から削除
                    for idx in indices_to_remove:
                        del self.feature_box_dict[self.frame_count][0]["feature"][idx]  # 特徴点を削除
                        del self.inactive_counter[idx]  # カウンターも削除
                        self.good_new= np.delete(self.good_new, idx, axis=0)
                        self.good_old= np.delete(self.good_old, idx, axis=0)

                    print("aho2")
                    print(len(self.feature_box_dict[self.frame_count][0]["feature"]))
                    print(len(self.good_new))



                #SIFTを使った特徴点登録
                def sift2():

                    self.result = img.copy()
                    result_images = []

                    #グレー画像
                    next = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    img2 =  img.copy() #入力画像

                    good = [] #良い特徴点マッチのリスト

                    self.assigned_template_ids = {}

                    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                    self.mask = np.zeros_like(gray_img2)



                    boxes = []

                    matched_ids = {}


                    fm_count= 0

                    #yolo11のセグメント情報からマスク画像を作成
                    for id, bbox in enumerate(yolo_segments):
                        probability = bbox.probability
                        x = bbox.xmin
                        y = bbox.ymin
                        xmax = bbox.xmax
                        ymax = bbox.ymax
                        height = ymax - y
                        width = xmax - x
                        class_id = bbox.class_id
                        x_masks = bbox.x_masks
                        y_masks = bbox.y_masks

                        box_info= [x,y,xmax,ymax]
                        mask_pairs = list(zip(y_masks, x_masks))

                        object_id = ""

                        if is_unknown_object(class_id, probability,box_info,img_size=self.img_size) and height < 700:
                            m_points = np.array(mask_pairs, dtype=np.int32)

                            self.mask = cv2.drawContours(self.mask, [m_points], -1, 255, thickness=cv2.FILLED)




                    #マスク画像を使ったSIFT抽出
                    kp2, des2 = self.sift.detectAndCompute(gray_img2,mask= self.mask)  #入力画像の特徴点算出





                    print("kp2の数",len(kp2))

                    # for kp in kp2:
                    #     cv2.circle(self.result, (int(kp.pt[0]),int(kp.pt[1])), 5, (255,0,0), -1)



                    

                    self.overlap_dict =[]




                    for n_id, bbox in enumerate(yolo_segments):
                        probability = bbox.probability
                        x = bbox.xmin
                        y = bbox.ymin
                        xmax = bbox.xmax
                        ymax = bbox.ymax
                        height = ymax - y
                        width = xmax - x
                        class_id = bbox.class_id
                        roi = img2[y:ymax, x:xmax]

                        x_masks = bbox.x_masks
                        y_masks = bbox.y_masks

                        box_point = []

                        box_des = []
                        reactangle =[x,y,xmax,ymax]

                        

                        if is_unknown_object(class_id, probability,reactangle,img_size=self.img_size) and height < 700:
                            print("nuer",n_id)

                            print("class",class_id)

                            mask_pairs = list(zip(y_masks, x_masks))
                            m_points = np.array(mask_pairs, dtype=np.int32)
                        
                            brack_img = self.seg_mask.copy()
                            brack_img = cv2.drawContours(brack_img, [m_points], -1, 255, thickness=cv2.FILLED)
                      
                            #物体領域をモルフロジー変換で収縮処理する
                            if class_id == "person":

                                distance = 10  # 削る距離（ピクセル単位）
                                kernel = np.ones((distance, distance), np.uint8)  # 距離に基づくカーネル


                                brack_img= cv2.erode(brack_img, kernel, iterations=1)
                            else:

                                distance = 5  # 削る距離（ピクセル単位）
                                kernel = np.ones((distance, distance), np.uint8)  # 距離に基づくカーネル


                                brack_img= cv2.erode(brack_img, kernel, iterations=1)



                            

                            #物体の２重検出を除く処理
                            for j_num, r_item in enumerate(yolo_segments):

                                xr = r_item.xmin
                                yr = r_item.ymin
                                xmaxr = r_item.xmax
                                ymaxr = r_item.ymax
                                heightr = ymaxr - yr
                                # print(height)
                                widthr = xmaxr - xr



                                reactangler = [xr,yr, xmaxr , ymaxr]


                                if is_unknown_object(class_id, probability,reactangler,img_size=self.img_size) and height < 700 :
                                    if n_id != j_num and iou(reactangle, reactangler):
                                        if not j_num  in self.overlap_dict:
                                            self.overlap_dict.append(j_num)  # 重なっているボックスのインデックスを保存
                                        if  not n_id  in self.overlap_dict:
                                            self.overlap_dict.append(n_id)




                            updated_keypoints = []

                            #入力画像の特徴点のサイズ(kp.size)がマスク内のものを抽出する
                            for i, val in enumerate(zip(kp2,des2)):
                                kp, des = val
                                # cv2.imshow("c",brack_img)

                                kx, ky = int(kp.pt[0]), int(kp.pt[1])  # 特徴点の座標
                                radius = int(kp.size/2)

                                 # 特徴点の周囲領域をクロップ
                                x_min = int(max(0, kx - kp.size))
                                x_max = int(min(brack_img.shape[1], int(kx + kp.size)))
                                y_min = int(max(0, ky - kp.size))
                                y_max = int(min(brack_img.shape[0], int(ky + kp.size)))

                                region = brack_img[y_min:y_max, x_min:x_max]

                                

                                # if brack_img[int(kp.pt[1]),int( kp.pt[0])] == 255 and brack_img[int(kp.pt[1])-1,int( kp.pt[0])] == 255 and brack_img[int(kp.pt[1]),int( kp.pt[0])-1] == 255 and brack_img[int(kp.pt[1])+1,int( kp.pt[0])]and brack_img[int(kp.pt[1]),int( kp.pt[0])+1] == 255 and brack_img[int(kp.pt[1]+1),int( kp.pt[0])+1] == 255 and brack_img[int(kp.pt[1]-1),int( kp.pt[0])-1] == 255: #x <= kp.pt[0] <= xmax and y <= kp.pt[1] <= ymax:
                                # cv2.circle(self.result, (int(kp.pt[0]),int(kp.pt[1])), int(kp.size / 2), (255,125,0), 1)
                                    # print("特徴あった")

                                if np.all(region ==255):  # 周辺が完全にマスク内

                                    # cv2.circle(self.matchkline, (int(kp.pt[0]),int(kp.pt[1])), int(kp.size / 2), (255,125,0), 2)
                                    new_kp = cv2.KeyPoint(kx - x, ky - y, kp.size)
                           

                                    # nm_x = int(kp.pt[0]-x)
                                    # nm_y = int(kp.pt[1]-y) 

                                    # kp2[i]= (nm_x,nm_y)


                                    box_point.append(new_kp)
                                    box_des.append(des2[i])


                            if not (box_point ==[] or box_des ==[]):

                                points_des =(box_point,box_des,class_id,reactangle)


                                self.assigned_template_ids.setdefault(n_id, points_des)

                            else:
                                
                                print("特徴点が見つからない")

                                points_des =(box_point,box_des,class_id,reactangle)


                                self.assigned_template_ids.setdefault(n_id, points_des)

        
                                # pdb.set_trace()








                    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

                    # kp2, des2 = self.sift.detectAndCompute(gray_img2,brack_img)  #入力画像の特徴点算出

                    
                    self.match_sift = {}


                    self.siftcount = {}
                    self.c_siftcount = {}

                    self.savepoint ={}

                    self.matched_images = []
                    self.matched_images2 = []
                    
                    distance_threshold =300

                    c_threshold= 100

                    self.good_dict = {}






                    


                    #siftのマッチング処理
                    for kt,feature  in self.assigned_template_ids.items():
                        print("確認")

                        kp2, des2, a_class_id,det_box = feature
                        des2 = np.asarray(des2, dtype=np.float32)

                        self.best_match_template_id = ""
                        self.second_best_match_template_id = ""


                        self.best_match_count = 0
                        self.second_best_match_count = 0

                        self.best_match_ratio= 0.0
                        # self.savepoint =[]
                        self.f_count = {}

                        c_fcount ={}

                        h_flag = False
                        # self.region_matched_img =cv2.hconcat(self.matchkline2,self.matchkline)


                        se_flag = False

                        self.region_matched_img =None

                        good_src ={}

                        no_matchid = {}

                        self.rrry= True

                        self.matchkline = img.copy()

                        


                      

                        

                        
                        for template_id, val in self.clipped_images.items():
                            # print(len(self.clipped_images))
                            # colar = template_colors[template_id % len(template_colors)]
                            # print(len(val))
                            print("確認２")

                            self.good = []
                            kp1 , self.des1, c_class_id,track_box, notmatch_frame = val

                            # print("box_n", track_box)

                            self.filtered_matches =[]





                            #ユークリッド距離による候補の選定
                            def is_within_distance(box1, box2,class_v, threshold,threshold2):
                                """
                                ボックスの中心間距離が閾値以内か確認する関数
                                """
                                center1 = np.array([(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2])  # 追跡ボックスの中心
                                center2 = np.array([(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2])  # 検出ボックスの中心
                                distance = np.linalg.norm(center1 - center2)  # ユークリッド距離


                                if class_v =="chair":

                                    return distance <= threshold2

                                if not distance <= threshold:
                                    print("ユークリッド距離ミス", distance)
                                return distance <= threshold


                            if is_within_distance(track_box, det_box,a_class_id, distance_threshold,c_threshold):
                                print("siftマッチング")



                                if not  template_id in self.prev_matchid and a_class_id !="chair":


                                    # self.kp1, self.des1 = self.sift.detectAndCompute(self.img1,None) #テンプレート画像の特徴点算出
                        

                                    self.des1 = np.asarray(self.des1, dtype=np.float32)

                                    # print("des1",self.des1)


                                    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

                                    



                                    # matches = bf.match(self.des1,des2)

                                    if des2 is not None and len(des2) >= 2:

                                        matches = self.flann.knnMatch(self.des1,des2,k=2) #特徴点マッチを行う

                                        for m,n in matches:
                                            if m.distance < 0.70*n.distance:
                                                self.good.append(m)

                                        match_ratio = len(self.good) / len(kp1) if len(kp1) > 0 else 0
                                        print("マッチの個数",len(matches))

                               



                                        threshold = 5  # 距離の閾値（ピクセル単位）
                                        fe_y = []
                                        old_f =[]
                                        max_distance3 = 52

                                        #マッチングした特徴点が近くにあるなら１つの特徴点に統合、過去のマッチングした特徴点の相対位置と離れすぎているもの除く
                                        for match in  self.good:
                                            boxt =self.mot_item_list[kt]
                                            new_kp = cv2.KeyPoint(int(kp2[match.trainIdx].pt[0]) + boxt[0], int(kp2[match.trainIdx].pt[1]) + boxt[1], kp.size)
                                            old_kp = cv2.KeyPoint(int(kp1[match.queryIdx].pt[0]) + boxt[0], int(kp1[match.queryIdx].pt[1]) + boxt[1], kp.size)
                                            train_point = new_kp.pt 

                                            is_duplicate = False
                                            for filtered_match in self.filtered_matches:
                                                existing_point = kp2[filtered_match.trainIdx].pt
                                                if euclidean(train_point, existing_point) < threshold:
                                                    is_duplicate = True
                                                    break
                                            
                                            if not is_duplicate:
                                                pot1 = np.array(kp1[match.queryIdx].pt)  # 画像1の対応点
                                                pot2 = np.array(kp2[match.trainIdx].pt)
                                                distance = np.linalg.norm(pot1 - pot2)
                                                if distance < max_distance3:
                                                    self.filtered_matches.append(match)

                                            fe_y.append(new_kp)
                                            old_f.append(old_kp)





                                    
                                    else:
                                    
                                       
                                        print("Not enough features in des2 to perform knnMatch.")


                                    
                                    # print("マッチの個数",len(matches))
                            


                                    # for m,n in matches:
                                    #     if m.distance < 0.70*n.distance:
                                    #         self.good.append(m)

                                    # match_ratio = len(self.good) / len(kp1) if len(kp1) > 0 else 0

                                    

                                    # f_count[template_id] = len(self.good)
                                    #track_idごとの対応点情報
                                    # good_src[template_id] = self.good

                                    new_matches = [
                                        cv2.DMatch(idx, idx, match.distance)
                                        for idx, match in enumerate( self.filtered_matches)
                                    ]

                                    


                                    MIN_MATCH_COUNT = 5 #最低限マッチしてほしい数
                                    count_cost = len(self.filtered_matches) #+int(match_ratio*100)
                                

                                    if len(self.filtered_matches)>MIN_MATCH_COUNT and a_class_id == c_class_id:
                                        h_flag = True

                                        

                                    
                                        # self.f_count[template_id] = count_cost
                                

                                        # c_fcount[template_id] = str(count_cost)+str("/")+str(len(kp1))
                                        print("kp1",len(kp1))
                                        print("eds1",self.des1.shape)
                                        print("m.queryIdx",m.queryIdx)

                                        src_pts = np.float32([ kp1[m.queryIdx].pt for m in self.filtered_matches ]).reshape(-1,1,2)
                                        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in self.filtered_matches ]).reshape(-1,1,2)
                                        
                                        
                                    
                                       

                                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,3.0)
                                        self.matchesMask = mask.ravel().tolist()



                                        det_pts2 = [ self.filtered_matches[i] for i in range(len(self.filtered_matches))  if self.matchesMask[i] == 1]

                                        # if len(det_pts2) > 12 :
                                        #     self.f_count[template_id] = len(det_pts2)


                                        count_cost = len(det_pts2)

                                        good_src[template_id] = det_pts2

                                        c_fcount[template_id] = str(count_cost)+str("/")+str(self.des1.shape[0])
                                        


                                        # インライヤーの数を計算
                                        inliers = mask.ravel().tolist()
                                        num_inliers = sum(inliers)
                                        num_matches = len(self.filtered_matches)


                                        # 確からしさの確率を計算
                                        confidence = num_inliers / num_matches


                                        # count_cost = count_cost*confidence

                                        # f_count[template_id] = count_cost


                                        # c_fcount[template_id] = str(count_cost)+str("/")+str(confidence)

                                        if len(self.f_count)>0:

                                            dis = count_cost - self.best_match_count

                                            print("dis",dis)
                                            #特徴点数をカウント

                                            if len(det_pts2) > 3 :
                                                c_fcount[template_id] = str(count_cost)+str("/")+str(self.des1.shape[0])
                                                self.f_count[template_id] = len(det_pts2)
                                            
                                        
                                        
                                        else:
                                     
                                            c_fcount[template_id] = str(count_cost)+str("/")+str(self.des1.shape[0])
                                            if len(det_pts2) > 3 :
                                                self.f_count[template_id] = len(det_pts2)




                                        if count_cost > self.best_match_count :
                                            self.second_best_match_template_id = self.second_best_match_template_id
                                            self.second_best_match_count =  self.best_match_count
                                            self.best_match_count = count_cost
                                            self.best_match_template_id = template_id

                                            # if len(self.f_count)>0:

                                            #     dis = self.best_match_count - self.second_best_match_count

                                            #     print("dis",dis)

                                            #     if len(det_pts2) > 10 and dis > 22:
                                            #         c_fcount[template_id] = str(count_cost)+str("/")+str(self.des1.shape[0])
                                            #         self.f_count[template_id] = len(det_pts2)
                                               
                                            #     else:
                                            #         self.f_count = {}
                                            #         self.rrry = False
                                            #         del self.mot_item_list[kt]
                                            #         break
                                            # else:
                                            #     c_fcount[template_id] = str(count_cost)+str("/")+str(self.des1.shape[0])
                                            #     if len(det_pts2) > 10 :
                                            #         self.f_count[template_id] = len(det_pts2)



                                            # src_pts = np.float32([ kp1[m.queryIdx].pt for m in self.good ]).reshape(-1,1,2)
                                            # dst_pts = np.float32([ kp2[m.trainIdx].pt for m in self.good ]).reshape(-1,1,2)

                                            self.savepoint[kt] = dst_pts

                                            # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                                            # self.matchesMask = mask.ravel().tolist()

                                            # kp2[m.trainIdx].pt = ()
                                            # absolute_keypoints = [(x + region_origin[0], y + region_origin[1]) for x, y in relative_keypoints]
                                            #マスク描画
                                            mask = self.masks_shaped[kt]
                                            the_mask = mask.copy()
                                            the_mask = np.stack([the_mask] * 3,axis=-1)
                                            color = (255,0,0)
                                            self.matchkline[the_mask[:, :, 0] > 0.5] =  self.matchkline[the_mask[:, :, 0] > 0.5] * 0.5 + np.array(color) * 0.5

                                        

                                            # マッチング結果を線で描画
                                            for idx1,idx2 in zip(src_pts,dst_pts):
                                                # print("nanid",idx1.flatten())
                                                
  

                                                pt1 = tuple(map(int, np.array([det_box[0], det_box[1]]).flatten()+idx2.flatten()))  # 旧フレームの特徴点
                                                pt2 = tuple(map(int,  np.array([track_box[0], track_box[1]]).flatten()+idx1.flatten()))   # 新フレームの特徴点

                                                cv2.circle(self.result, pt1, 2,(0, 255, 0), 2) 
                                                # cv2.line(track, pt1, pt2, (0, 0, 255), 2)  # 赤色のフロー線

                                            


                                            # 対応する特徴点を線で可視化
                                            self.region_matched_img = cv2.drawMatches( 
                                                self.matchkline2, old_f, self.matchkline,fe_y,  new_matches, None,matchColor=(0, 0, 255),singlePointColor=(0, 255, 0)
                                            )




                                            

                                            h_flag = True


                                        elif count_cost > self.second_best_match_count :
                                            self.second_best_match_count = count_cost 
                                            self.second_best_match_template_id = template_id

                                            # 対応する特徴点を線で可視化
                                            region_matched_img2 = cv2.drawMatches( 
                                                self.matchkline2, kp1, self.matchkline, kp2, self.filtered_matches, None,matchColor=(0, 255, 0),singlePointColor=(0,0,255)
                                            )

                                            se_flag = True
                                    




                                #椅子のSIFTマッチ処理(特徴が取れにくいので、別設定)
                                else: 
                                    # kp1 , self.des1 = self.prv_des[template_id]

                                    self.des1 = np.asarray(self.des1, dtype=np.float32)

                                    # print("des1",self.des1)


                                    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)





                                    # matches = bf.match(self.des1,des2)

                                    if des2 is not None and len(des2) >= 2:

                                        matches = self.flann.knnMatch(self.des1,des2,k=2) #特徴点マッチを行う
                                        for m,n in matches:
                                            if m.distance < 0.75*n.distance:
                                                self.good.append(m)

                                        match_ratio = len(self.good) / len(kp1) if len(kp1) > 0 else 0


                                        threshold = 5  # 距離の閾値（ピクセル単位）

                                    else:

                                        # for m,n in matches:
                                        #     if m.distance < 0.75*n.distance:
                                        #         self.good.append(m)

                                        # match_ratio = len(self.good) / len(kp1) if len(kp1) > 0 else 0

                                        print("Not enough features in des2 to perform knnMatch.")



                                    # match_ratio = len(self.good) / len(kp1) if len(kp1) > 0 else 0

                                    # good_src[template_id] = self.good

                                    # f_count[template_id] = len(self.good)

                                    # count_cost = len(self.good) #+int(match_ratio*100)




                                    MIN_MATCH_COUNT = 4 #最低限マッチしてほしい数

                                    if len(self.good)>MIN_MATCH_COUNT and a_class_id == c_class_id:

                                        h_flag = True
                                        # f_count[template_id] = count_cost #len(self.good)


                                        src_pts = np.float32([ kp1[m.queryIdx].pt for m in self.good ]).reshape(-1,1,2)
                                        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in self.good ]).reshape(-1,1,2)

                                    

                                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,10.0)
                                        self.matchesMask = mask.ravel().tolist()

                                        num_inliers = sum(self.matchesMask)

                                        num_matches = len(self.good)

                                        det_pts2 = [ self.good[i] for i in range(len(self.good))  if self.matchesMask[i] == 1]

                                        good_src[template_id] = det_pts2





                                        count_cost = len(det_pts2)

                                        # 確からしさの確率を計算
                                        confidence = num_inliers / num_matches
                                        # print(f"確からしさの確率: {confidence:.2f} ({num_inliers}/{num_matches})")
                                        # count_cost = count_cost * confidence

                                        if count_cost > 5:

                                            self.f_count[template_id] = len(det_pts2)#len(self.good)

                                        

                                        c_fcount[template_id] = str(len(self.good))+str("/")+str(len(kp1))


                                        # c_fcount[template_id] = str(count_cost)+str("/")+str(confidence)




                                        if count_cost > self.best_match_count :
                                            self.second_best_match_template_id = self.second_best_match_template_id
                                            self.second_best_match_count =  self.best_match_count
                                            self.best_match_count = count_cost
                                            self.best_match_template_id = template_id


                                            # src_pts = np.float32([ kp1[m.queryIdx].pt for m in self.good ]).reshape(-1,1,2)
                                            # dst_pts = np.float32([ kp2[m.trainIdx].pt for m in self.good ]).reshape(-1,1,2)

                                            self.savepoint[kt] = dst_pts

                                            # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                                            # self.matchesMask = mask.ravel().tolist()

                                            # mask = self.masks_shaped[kt]
                                            # the_mask = mask.copy()
                                            # the_mask = np.stack([the_mask] * 3,axis=-1)
                                            # color = (255,0,0)
                                            # self.matchkline[the_mask[:, :, 0] > 0.5] =  self.matchkline[the_mask[:, :, 0] > 0.5] * 0.5 + np.array(color) * 0.5





                                            # 対応する特徴点を線で可視化
                                            self.region_matched_img = cv2.drawMatches( 
                                                self.matchkline2, kp1, self.matchkline, kp2, det_pts2, None,matchColor=(0, 255, 0),singlePointColor=(0, 0, 255)
                                            )
                                     

                                            

                                            


                                        elif count_cost > self.second_best_match_count :
                                            self.second_best_match_count = count_cost
                                            self.second_best_match_template_id = template_id
                                            se_flag = True

                                            # 対応する特徴点を線で可視化
                                            region_matched_img2 = cv2.drawMatches( 
                                                self.matchkline2, kp1, self.matchkline, kp2, det_pts2, None,matchColor=(0, 255, 0),singlePointColor=(0, 0, 255)
                                            )
                            else:
                                print("ユークリッド距離判定外")

                        
                        dis =  self.best_match_count -self.second_best_match_count

                        print("dis",dis)

                        # if not  abs(dis) > 25:
                    
                        #     self.rrry = False
                        #     del self.mot_item_list[kt]
                            
     

                        

                                
                        self.good_dict[kt] = good_src
                                

                            

                            # 画像左上にどの領域のマッチング結果かを表示


                            # テキストの表示位置 (右上)

                        if h_flag and self.region_matched_img is not None :
                            print("kaku",len(self.good))
                       
                            y_start = 50  # 開始y座標
                
                            x_right = self.region_matched_img.shape[1] - 15  # 右端からの位置
                            text = f"{kt}"

                            font_scale = 2
                            thickness = 3

                            font = cv2.FONT_HERSHEY_SIMPLEX

                            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                            x_text = x_right - text_size[0]  # テキストの右端を画像右端に合わせる
    
                            text_color = (255, 255, 255)  # 青色
                            cv2.putText(
                                self.region_matched_img, text, (x_text, y_start), cv2.FONT_HERSHEY_SIMPLEX, 
                                font_scale, text_color, thickness, lineType=cv2.LINE_AA
                            )
                            

                            if se_flag :
                                x_right = region_matched_img2.shape[1] - 15

                                text = f" SECOOND Region {kt}:)"
                    

                                cv2.putText(
                                    region_matched_img2, text, (x_text, y_start), cv2.FONT_HERSHEY_SIMPLEX, 
                                    font_scale, text_color, thickness, lineType=cv2.LINE_AA
                                )
                                self.matched_images2.append(region_matched_img2)

                            



                                
                            

                            self.matched_images.append(self.region_matched_img)



                                # src_pts = np.float32([ kp1[m.queryIdx].pt for m in self.good ]).reshape(-1,1,2)
                                # dst_pts = np.float32([ kp2[m.trainIdx].pt for m in self.good ]).reshape(-1,1,2)

                                # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                                # self.matchesMask = mask.ravel().tolist()



                                # マッチングの対応点を全体画像に描画




                                # for i, m in enumerate(self.savepoint):
                                #     if self.matchesMask[i]:  # マッチした点だけ描画

                                #         # pt1 = tuple(np.int32(src_pts[i][0]))  # テンプレート画像の点
                                #         pt2 = tuple(np.int32(self.dst_pts[i][0]))  # 1フレーム目の全体画像の対応する点


                                #         # pt1_adjusted = (pt1[0] + x_offset, pt1[1] + y_offset)
                                #         # print("id",self.best_match_template_id )
                                #         colar =get_id_color(self.best_match_template_id)
                                #         # pt1_adjusted2 = (pt11[0] + x_offset, pt11[1] + y_offset)

                                #         # cv2.circle(result, pt1_adjusted, 5, colar, 2)

                                #         # cv2.circle(result, pt1_adjusted, 5, colar,-1)  # 対応点を青で描画

                                #         # 対応する特徴点を線で結び、点も描画
                                #         # cv2.line(result, pt1_adjusted, pt2, colar, 1, cv2.LINE_AA)
                                #         cv2.circle(self.result, pt2, 5, colar, -1)  # 対応点を青で描画


                        # self.match_sift[kt] = self.best_match_template_id

                        if  self.rrry:

                            self.siftcount.setdefault(kt, []).append(self.f_count)
                            self.c_siftcount.setdefault(kt, []).append(c_fcount)




                        # if self.best_match_template_id != "" and not self.best_match_template_id in self.match_sift.values():
                        #     # self.clipped_images[self.best_match_template_id] = self.assigned_template_ids[kt]
                        #     # print("",)
                        #     self.match_sift[kt] = self.best_match_template_id
                        #     self.siftcount.setdefault(kt, []).append(f_count)

                        # elif self.second_best_match_template_id != "" and not self.best_match_template_id in self.match_sift.values():
                        #     # self.clipped_images[self.second_best_match_template_id] = self.assigned_template_ids[kt]
                        #     self.match_sift[kt] = self.second_best_match_template_id

                        #     self.siftcount.setdefault(kt, []).append(f_count)

                        # else:

                        #     # self.clipped_images[self.countf] = self.assigned_template_ids[kt]

                        #     self.siftcount.setdefault(kt, []).append(f_count)

                        #     self.match_sift[kt] = self.countf
                        #     self.countf += 1


                    # print("self.match_sift",self.match_sift )

                    print("old_sif_mt",self.siftcount)



                    # # フォント設定
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    color = (255, 255, 255)  # 黒色
                    thickness = 2

                    # テキストの表示位置 (右上)
                    margin = 15
                    y_start = margin  # 開始y座標
                    line_spacing = 25  # 行間
                    x_right = img.shape[1] - margin  # 右端からの位置


                    # # 辞書の内容を描画
                    for key, value_list in self.c_siftcount.items():
                        for value_dict in value_list:
                            # テキスト内容
                            text = f"Key {key}: " + ", ".join([f"{k}:{v}" for k, v in value_dict.items()])

                            # テキストサイズを計算 (右端揃え用)
                            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                            x_text = x_right - text_size[0]  # テキストの右端を画像右端に合わせる

                            # 画像上に描画
                            cv2.putText(self.result, text, (x_text, y_start), font, font_scale, color, thickness)
                            y_start += line_spacing  # 次の行に移動

                    



                    assigned_keys = {}  # 内側キー -> 割当済み (値, 外側キー)
                    self.sift_match2 = {}

                    # # 空のリストや辞書がないデータだけをフィルタリング
                    # self.siftcount = {k: v for k, v in self.siftcount.items() if v and isinstance(v[0], dict) and v[0]}

                    # sorted_keys = sorted(self.siftcount.keys(), key=lambda k: max(self.siftcount[k][0].values()), reverse=True)







                thread2 = threading.Thread(target= sift2)
                thread1 = threading.Thread(target= lktracking2)

                thread1.start()
                thread1.join()

                thread2.start()
                thread2.join()


                # print("self.assigned_template_ids",self.assigned_template_ids)



                #siftとopt_flowの特徴点の割当結果を組み合わせる
                def merge_dicts(d1, d2):
                    merged = {}

                    # d1 と d2 の順序を維持する
                    for key in d1:
                        if key in d2:
                            combined = {}

                            # 空でない辞書をスキップ
                            d1_filtered = [d for d in d1[key] if d]  # 空でない辞書のみ残す
                            d2_filtered = [d for d in d2[key] if d]  # 空でない辞書のみ残す

                            # キーと値をマージ
                            d1_dict = {k: v for d in d1_filtered for k, v in d.items()}
                            d2_dict = {k: v for d in d2_filtered for k, v in d.items()}

                            # キーを統合して値を足す
                            all_subkeys = set(d1_dict.keys()).union(d2_dict.keys())
                            for subkey in all_subkeys:
                                combined[subkey] = (d2_dict.get(subkey, 0))#int(0.5*d1_dict.get(subkey, 0)) d1_dict.get(subkey, 0) + 1.1*d1_dict.get(subkey, 0) +

                            merged[key] = [combined]
                        # else:
                            # merged[key] = int(0.7*d1[key])  # d1 のみが存在する場合

                    # d2 に存在して、d1 にないキーも追加
                    for key in d2:
                        if key not in d1:
                            merged[key] = d2[key]  # d2 のみが存在する場合

                    return merged


                # self.siftcount = merge_dicts(self.ID_point_count_dict, self.siftcount)

                # self.siftcount = dict(sorted(self.siftcount.items()))

                print("new_mt",self.siftcount)











                assigned_keys = {}  # 内側キー -> 割当済み (値, 外側キー)
                self.sift_match = {}
                self.sift_maxcont = {}

                # 空のリストや辞書がないデータだけをフィルタリング
                # self.siftcount = {k: v for k, v in self.siftcount.items() if v and isinstance(v[0], dict) and v[0]}

                # sorted_keys = sorted(self.siftcount.keys(), key=lambda k: max(self.siftcount[k][0].values()), reverse=True)


                self.sift_match = {key: 'None' for key in self.siftcount.keys()}  # すべてのキーに初期値として None を設定
                self.sift_maxcont = {key: 0 for key in self.siftcount.keys()}

                # 割り当て処理
                for outer_key in self.siftcount.keys():
                    # データが空の場合はスキップ（None は既に設定済み）
                    if not self.siftcount[outer_key] or not self.siftcount[outer_key][0]:
                        # self.sift_match[outer_key] ='None'
                        continue

                    inner_dict = self.siftcount[outer_key][0]
                    sorted_inner = sorted(inner_dict.items(), key=lambda x: x[1], reverse=True)

                    for inner_key, value in sorted_inner:
                        # 内側キーがまだ割り当てられていない場合
                        if inner_key not in assigned_keys:
                            assigned_keys[inner_key] = (value, outer_key)
                            self.sift_match[outer_key] = str(inner_key)
                            self.sift_maxcont[inner_key] = value

                            break
                        else:
                            # 内側キーが既に割り当て済みの場合
                            existing_value, existing_outer = assigned_keys[inner_key]
                            if value > existing_value:
                                # 割り当て直し：新しい外側キーに割り当て
                                assigned_keys[inner_key] = (value, outer_key)
                                self.sift_match[outer_key] = str(inner_key)
                                self.sift_maxcont[inner_key] = value

                                # 元の外側キーを次の候補に再割り当て
                                del self.sift_match[existing_outer]
                                
                                self.sift_match[existing_outer] = 'None'  # 元のキーに再割り当てできない場合に備えて None を設定
                                if existing_outer in self.siftcount and self.siftcount[existing_outer] and self.siftcount[existing_outer][0]:
                                    for next_inner_key, next_value in sorted(self.siftcount[existing_outer][0].items(), key=lambda x: x[1], reverse=True):
                                        if next_inner_key not in assigned_keys or next_value > assigned_keys[next_inner_key][0]:
                                            assigned_keys[next_inner_key] = (next_value, existing_outer)
                                            self.sift_match[existing_outer] = str(next_inner_key)
                                            self.sift_maxcont[next_inner_key] = next_value
                                            break
                                break



                self.sift_match = dict(sorted(self.sift_match.items()))

                print("self.sift_match",self.sift_match)

                # self.sift_match2 = dict(sorted(self.sift_match2.items()))



            # # フォント設定
            #     font = cv2.FONT_HERSHEY_SIMPLEX
            #     font_scale = 0.6
            #     color = (255, 255, 255)  # 黒色
            #     thickness = 2

            #     # テキストの表示位置 (右上)
            #     margin = 15
            #     y_start = margin  # 開始y座標
            #     line_spacing = 25  # 行間
            #     x_right = img.shape[1] - margin  # 右端からの位置


            #     # 辞書の内容を描画
            #     for key, value_list in self.siftcount.items():
            #         for value_dict in value_list:
            #             # テキスト内容
            #             text = f"Key {key}: " + ", ".join([f"{k}:{v}" for k, v in value_dict.items()])

            #             # テキストサイズを計算 (右端揃え用)
            #             text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            #             x_text = x_right - text_size[0]  # テキストの右端を画像右端に合わせる

            #             # 画像上に描画
            #             cv2.putText(self.flow, text, (x_text, y_start), font, font_scale, color, thickness)
            #             y_start += line_spacing  # 次の行に移動



                for mot_item, bo_id  in zip(self.mot_item_list.values(),self.sift_match.values()):

                        if bo_id != 'None':
                            if int(bo_id) in self.fe_idelist:
                                box_flow = self.fe_idelist[int(bo_id)]
                                mot_item.insert(5, box_flow)
                            else:
                                box_flow = []
                                mot_item.insert(5, box_flow)
                        else:
                            box_flow = []
                            mot_item.insert(5, box_flow)








                # #mot
                motdetections = mot.bboxes2out_detections(self.mot_item_list)
                # print("box_view")
                # print(motdetections)
                _, d_trac,d_ind, lost_f_id , self.re_match = self.mottracker.step(motdetections, self.sift_match)
                mottracks = self.mottracker.active_tracks(min_steps_alive=3)
                print("count_box")
                # print((d_trac))

                print("len(mot)",len( mottracks))







                for track_result in mottracks:
                    self.new_id = self.bbox_count

                    print("class_id_f",track_result.class_id)

                    if track_result.id not in self.track_id_dict:

                        self.track_id_dict[track_result.id]= self.new_id
                        self.bbox_count += 1

                    tracker_id = self.track_id_dict[track_result.id]

                self.result = mot.draw_debug(self.result,mottracks,self.track_id_dict)



                #検出Boxデバック用

                # 重なり情報を保存する辞書
                # overlap_dict = []

                none_count2 = sum(1 for v in self.sift_match.values() if v == 'None')




                for num, w_item in enumerate(yolo_segments):

                    probability = w_item.probability
                    x = w_item.xmin
                    y = w_item.ymin
                    xmax = w_item.xmax
                    ymax = w_item.ymax
                    height = ymax - y
                    # print(height)
                    width = xmax - x
                    class_id = w_item.class_id
                    object_id = ""
                    x_masks = w_item.x_masks
                    y_masks = w_item.y_masks


                    point_counts = {}
                    total_counts = {}

                    reactangle = [x,y, xmax, ymax]

                    # count = 0

                    if is_unknown_object(class_id, probability,reactangle,img_size=self.img_size) and height < 700 :



                        # for j_num, r_item in enumerate(yolo_segments):

                        #     xr = r_item.xmin
                        #     yr = r_item.ymin
                        #     xmaxr = r_item.xmax
                        #     ymaxr = r_item.ymax
                        #     heightr = ymaxr - yr
                        #     # print(height)
                        #     widthr = xmaxr - xr



                        #     reactangler = [xr,yr, xmaxr , ymaxr]


                        #     if is_unknown_object(class_id, probability) and height < 700 :
                        #         if num != j_num and iou(reactangle, reactangler):
                        #             if not j_num  in overlap_dict:
                        #                 overlap_dict.append(j_num)  # 重なっているボックスのインデックスを保存
                        #             if  not num  in overlap_dict:
                        #                 overlap_dict.append(num)




                        # mask_pairs = list(zip(y_masks, x_masks))

                        # m_points = np.array(mask_pairs, dtype=np.int32)




                        if num in self.sift_match:
                            id = self.sift_match[num]


                            if id =="None":

                                if none_count2 !=0 and not none_count2 < 0 :

                                    print("self.track_id_dict", self.track_id_dict)

                                    id = self.new_id-none_count2+1


                                    # self.clipped_images[self.new_id-none_count+1] = self.assigned_template_ids[kt]

                                    # self.match_sift[kt] = self.new_id-none_count+1

                                    colar =get_id_color(int(id))

                                    none_count2 -=1


                                    cv2.putText(self.detect, f'Class : {class_id}', (x,y-5),cv2.FONT_HERSHEY_PLAIN, 1.5, colar , thickness=2)
                                    self.detect = cv2.rectangle(self.detect, (x, y), (x + width, y + height), colar , thickness=3)
                            else:
                                id = self.sift_match[num]


                                colar =get_id_color(int(id))
                                cv2.putText(self.detect, f'Class : {class_id}', (x,y-5),cv2.FONT_HERSHEY_PLAIN, 1.5, colar , thickness=2)
                                self.detect = cv2.rectangle(self.detect, (x, y), (x + width, y + height), colar , thickness=3)






                #             self.detect = cv2.drawContours(self.detect, [m_points], -1, colar, thickness=cv2.FILLED)
    

                #siftの更新

                print("assdd",len(self.assigned_template_ids))
                print("mottracks",len(mottracks))

                none_count3 = sum(1 for v in self.sift_match.values() if v == 'None')

                # self.prev_matchid =[]



                for kt,feature in self.assigned_template_ids.items():

                    # kt,feature = assigned_template




                    kp2, des2,class_id,det_box = feature

                    self.detec_box = det_box
                    # des2 = np.asarray(des2, dtype=np.float32)







                    if kt  in self.sift_match and self.sift_match[kt] !='None' :

                        self.best_match_template_id = int(self.sift_match[kt])


                        print("self.assigned_template_ids[kt]",self.best_match_template_id)
                        print("self.f_count",self.sift_maxcont)

                        # if self.sift_maxcont != {}:
                        #     if int(self.sift_maxcont[self.best_match_template_id]) >5:
                        
                

                        kp1, des1,class_id,trac_box, frame_alive = self.clipped_images[self.best_match_template_id] 
                        # if isinstance(des1, list):
                        self.clipped_images[self.best_match_template_id] = (kp1, des1,class_id,self.detec_box, frame_alive)

                        #     des1 = np.asarray(des1, dtype=np.float32)


                        self.match_sift[kt] = self.best_match_template_id

                        colar =get_id_color(self.best_match_template_id)



                        # if self.siftcount[kt][0][int(self.sift_match[kt])] > 5:

                        #     self.best_match_template_id = int(self.sift_match[kt])

                        #     # if len(self.clipped_images[self.best_match_template_id]) > len(self.assigned_template_ids[kt]):

                        #         self.clipped_images[self.best_match_template_id] = self.assigned_template_ids[kt]
                        #         # print("",)
                        #         self.match_sift[kt] = self.best_match_template_id
                        #     colar =get_id_color(self.best_match_template_id)
                   

                        pt_src =self.good_dict[kt]
                        point_data = pt_src[self.best_match_template_id]
                        new_pts =[]
                        old_pts = []

                        self.del_kp2= []

                        counts = 0

                        counts2 = 0

                        frame_le = self.frame_count

                        print("self.frame_count",self.frame_count)

                        

                        
                        #マッチングした特徴点の情報を更新
                        for m in point_data :
                            pts = kp2[m.trainIdx].pt

                            # cv2.drawMarker(self.result, tuple(np.int32((pts))), colar, markerType=cv2.MARKER_TRIANGLE_UP, markerSize=9, thickness=2, line_type=cv2.LINE_8)
                            frae = self.frame_count - frame_alive[m.queryIdx]

                            if frae < 10 :

                                # print("kp2",len(kp2))

                                # print("m.trainIdx",m.trainIdx)

                                # new_pts = np.float32( kp2[m.trainIdx].pt).reshape(-1,1,2)
                                new_pts = kp2[m.trainIdx].pt
                                new_des = des2[m.trainIdx]
                                kp1[m.queryIdx] = kp2[m.trainIdx]
                                des1[m.queryIdx] = new_des

                                # kp1 = np.vstack((kp1, new_pts))
                                

                                # des1 = np.vstack(des1,new_des)

                                frame_alive[m.queryIdx] = self.frame_count
                                

                                self.del_kp2.append(m.trainIdx)

                                counts += 1
                                # del kp2[m.trainIdx]
                                if len(new_pts) > 0:
                                    if  (self.comand == "sift" or self.comand == "new_point" or self.comand == "line") :
                                        cv2.drawMarker(self.result, tuple(np.int32((new_pts))), (0,0,255), markerType=cv2.MARKER_TRIANGLE_UP, markerSize=9, thickness=2, line_type=cv2.LINE_8)
                                        # cv2.drawMarker(self.result, tuple(np.int32((new_pts))), colar, markerType=cv2.MARKER_TRIANGLE_UP, markerSize=9, thickness=2, line_type=cv2.LINE_8)

                            

                            else:
                                # old_pts = np.float32(kp2[m.trainIdx].pt).reshape(-1,1,2)
                                old_pts = kp2[m.trainIdx].pt
                                old_des = des2[m.trainIdx]

                                before_point = kp1[m.queryIdx].pt
                                kp1[m.queryIdx] = kp2[m.trainIdx]
                                des1[m.queryIdx] = old_des

                                self.del_kp2.append(m.trainIdx)

                                # if (self.frame_count -frame_alive[m.queryIdx]) > 100:
                                #     frame_alive[m.queryIdx]  = self.frame_count
                                # else:
                                frame_alive[m.queryIdx]  = self.frame_count
                                # else:

                                # del kp2[m.trainIdx]

                                counts2 += 1




                                if len(old_pts)>0:
                                 
                                    if self.comand == "sift" or self.comand == "old_point" or self.comand == "line":
                                        # print("old_pts",old_pts)
                                        cv2.drawMarker(self.result, tuple(np.int32((old_pts))), (0,255,0), markerType=cv2.MARKER_TRIANGLE_UP, markerSize=9, thickness=2, line_type=cv2.LINE_8)
                            
                                    if self.comand == "line":

                                        cv2.drawMarker(self.result, tuple(np.int32((before_point))), (0,255,0), markerType=cv2.MARKER_TRIANGLE_UP, markerSize=9, thickness=2, line_type=cv2.LINE_8)

                                        cv2.line(self.result,tuple(np.int32((old_pts))),tuple(np.int32((before_point)),(255,0,0),2))

                        # frame_alive = [x + 1 for x in frame_alive]
                        print("kp2",len(kp2))
                        print("des2",len(des2))
                        # print("del_kp2",del_kp2)
                        # n_k = len(point_data)- len(self.del_kp2)
                        # o_k =len(self.del_kp2)

                        # x_right = self.result.shape[1] - 15 
                        # line_spacing = 25 
                        # font = cv2.FONT_HERSHEY_SIMPLEX
                        # font_scale = 0.6
                        # thickness = 2
                        # y_start =15
                        # color = (255, 255, 255)
                        # # テキスト内容
                        # text =f"good_match ({counts}/{len(kp2)})"

                        # text2 =f"re_match ({counts2}/{len(kp2)})"
                        # all_p = len(kp2)

                        # # テキストサイズを計算 (右端揃え用)
                        # text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                        # x_text = x_right - text_size[0]  # テキストの右端を画像右端に合わせる

                        # # 画像上に描画
                        # cv2.putText(self.result, text, (x_text, y_start), font, font_scale, color, thickness)
                        # y_start += line_spacing  # 次の行に移動

                        # cv2.putText(self.result, text2, (x_text, y_start), font, font_scale, color, thickness)


                        if self.del_kp2 != []:
                            del_kp2=list(set(self.del_kp2))
                            for index in sorted(del_kp2, reverse=True):
                                # print(index)
                                del kp2[index]
                                del des2[index]
                            
                        del_siftIdx = []

                        print("len(frame_alive)",len(frame_alive))


                        
                        
                        for f_idx , alive_count in enumerate(frame_alive):
                            frame_le = frame_le - alive_count
                            if frame_le > 200:


                                # kp1 = np.delete(kp1, f_idx, axis=0)
                                del kp1[f_idx]
                                del des1[f_idx]
                                del_siftIdx.append(f_idx)


                  
                        
                        for inx in sorted( del_siftIdx, reverse=True):
                            del frame_alive[inx]

                      
                        
                        #SIFT特徴点の追加

                        if self.siftcount[kt][0][int(self.sift_match[kt])] < 80 and len(kp1)< 8000 :

                            if self.comand == "sift" or self.comand == "add_point":
                                for point in kp2 :
                                    # colar =get_id_color(self.best_match_template_id)
                                    cv2.drawMarker(self.result, tuple(np.int32((point.pt ))), (255,0,0), markerType=cv2.MARKER_TRIANGLE_UP, markerSize=9, thickness=2, line_type=cv2.LINE_8)


                            kp1.extend(kp2)
                                

                            des1.extend(des2)

                            # num_rows = kp2.shape[0]
                            num_rows = len(kp2)

                            #新しい特徴分だけ追加
                            frame_alive.extend([self.frame_count] * num_rows)



                            self.clipped_images[self.best_match_template_id] = (kp1, des1,class_id,det_box, frame_alive)
                        
                        
                    else:
                            print("none_count3",none_count3)
                            if none_count3 !=0 and not none_count3 < 0 :

                                print("self.track_id_dict", self.track_id_dict)

                                self.best_match_template_id = self.new_id-none_count3+1

                                


                                # if not  int(kt) in overlap_dict :

                                # num_rows = kp2.shape[0]

                                num_rows =  len(kp2)
                                
                                frame_count = [1] * num_rows

                                print("box",self.detec_box)


                                self.clipped_images[self.new_id-none_count3+1] = (kp2, des2,class_id,self.detec_box, frame_count)
                                self.match_sift[kt] = self.new_id-none_count3+1

                                
                                colar =get_id_color(self.best_match_template_id)

                                none_count3 -=1


                                kp, new_des, _,_= self.assigned_template_ids[kt]
                                for  new_kp in kp:

                                    # print(new_kp.pt)

                                # pt2 = tuple(np.int32(dst_pts[0]))

                                    if self.comand == "sift":
                                        
                                        cv2.drawMarker(self.result, (int(new_kp.pt[0]),int(new_kp.pt[1])), colar, markerType=cv2.MARKER_TRIANGLE_UP, markerSize=9, thickness=2, line_type=cv2.LINE_8)




                d_ind = sorted(d_ind, reverse=True)

                keys_list = list(self.clipped_images.keys())

                print(self.clipped_images.keys())

                print("fff",keys_list)

                
                for l_id in d_trac:

                    if l_id in self.clipped_images:

                        del self.clipped_images[l_id]

                    # print("clipped_images",self.clipped_images)


                feature_list = []

                self.id_list =[]

                lost_points = []

                # print(len(self.feature_box_dict[self.frame_count][0]["feature"]))
                # print(len(self.good_new.reshape(-1, 1, 2)))

                del_point= []

                a_mask = self.seg_mask.copy()





                # オプティカルフロー勾配法の新しい特徴点の登録　修正部分
                for point_num , values in enumerate(self.feature_box_dict[self.frame_count][0]["feature"]) :

                    new_point, old_point, trackid = values
                    #print(values)
                    #print(trackid)
                    lost_feature_flag = True

                    a,b = new_point


                    count = 0

                    for num, mask in self.mask_box.items(): #mottrackstrack_result

                        object_id = ""
             
                        if object_id == ""  and num in self.sift_match:



                            if self.sift_match[num]!= 'None':
                                # object_id = self.re_match[count]
                                object_id = self.sift_match[num]
                            else:
                                object_id == 'None'
                            
                        
                        a,b = new_point
                        c,d = old_point

                        if mask[int(b),int(a)] ==255:
     

                            if self.comand == "flow":
                                
                                txt_bk_color = get_id_color(int(trackid[0]))

                                oldp_color = (0, 0, 255) 

    
                                
                                self.result = cv2.circle(self.result,(int(a),int(b)),5,oldp_color,-1)
                                self.result = cv2.circle(self.result,(int(c),int(d)),5,txt_bk_color,-1)
                                cv2.line(track, (int(a),int(b)),(int(c),int(d)), txt_bk_color, 2)

                            lost_feature_flag = False



                            if len(trackid) > 0:
                                if object_id == 'None' or object_id == '':
                                    
                                    lost_points.append(point_num)
                                    # del self.feature_box_dict[self.frame_count][0]["feature"][point_num]

                                elif int(trackid[0]) != int(object_id):    #int(self.re_match[count]) :



                                    # print(trackid[0])
                                    # print('でなない')
                                    lost_points.append(point_num)
                                    # lost_feature_flag = True

                                if "or" in str(object_id) :
                                    continue



                            else:
               

                                if "or" in str(object_id) :
                                    txt_bk_color = colar
                                elif object_id !='' and object_id !='None':
                                    txt_bk_color = get_id_color(int(object_id))
               
                                    trackid.append(int(object_id))

                     

                            count +=1
     



                    if lost_f_id !=[] or  d_trac !=[]:
                        if trackid[0] in lost_f_id or trackid[0] in d_trac:
                            del_point.append(point_num)



                    feature_list.append([new_point,old_point,trackid])




                feature = {"feature": feature_list}



                # print(len(self.feature_box_dict[self.frame_count][0]["feature"]))
                lost_points = list(set(lost_points))
                # print("lost")
                lost_points =sorted(lost_points, reverse=True)
                # print(lost_points)
                print(len(self.feature_box_dict[self.frame_count][0]["feature"]))

                for point_lost in lost_points:
                    if 0 <= point_lost < len(self.feature_box_dict[self.frame_count][0]["feature"]):

                        del self.feature_box_dict[self.frame_count][0]["feature"][point_lost]
                        self.good_new =np.delete(self.good_new, point_lost,axis=0)
                        self.good_old =np.delete(self.good_old, point_lost,axis=0)






                best_overlap_count = 0

            

            self.track_list.append(track)
            if( len(self.track_list)> 3 ):
                self.track_list.pop(0)
            

            # if self.comand == "flow":
            for t in self.track_list :
                self.result = np.where(t!=0,t,self.result)


            self.matchkline2 = self.matchkline

            self.prvs = next



            feature_list2 =[]
            


            #　勾配法の特徴点追加
            if   self.point_relia:
                counta = 0
                mask = img.copy()

                # gpu_image = cv2.cuda_GpuMat()

                # gpu_image.upload(mask)
                add_count = len(self.new_key)


                new_addpoint = []


                sift_point= []

                add_flow = {}

                none_count3 = sum(1 for v in self.sift_match.values() if v == 'None')


                self.mask_box



                for id_n, bbox in enumerate(yolo_segments):
                    probability = bbox.probability
                    x = bbox.xmin
                    y = bbox.ymin
                    xmax = bbox.xmax
                    ymax = bbox.ymax
                    height = ymax - y
                    width = xmax - x
                    class_id = bbox.class_id
                    x_masks = bbox.x_masks
                    y_masks = bbox.y_masks

                    mask_pairs = list(zip(y_masks, x_masks))

                    m_points = np.array(mask_pairs, dtype=np.int32)





                    box_info = [x,y,xmax,ymax]


                    object_id = ""


                    if is_unknown_object(class_id, probability,box_info,img_size=self.img_size) and height < 700 and id_n in self.new_key:

                        mask = cv2.drawContours(mask, [m_points], -1, 255, thickness=cv2.FILLED)


                        distance = 10  # 削る距離（ピクセル単位）
                        kernel = np.ones((distance, distance), np.uint8)  # 距離に基づくカーネル


                        mask= cv2.erode(mask, kernel, iterations=1)

                        if id_n in self.assigned_template_ids:
                            kp, new_des,_ ,_= self.assigned_template_ids[id_n]

             

                            # sift_point.append(kp)
                            track_id = []

                            print("kep",len(kp))


                            for  get_point in kp:

                                # a,b = add_point.ravel()
                                a, b = get_point.pt
                                new_point = a,b
                                old_point = a,b
                                # print("new",new_point)




                                #sift_add

                                if id_n  in self.sift_match and self.sift_match[id_n] !='None':
                                    object_id = int(self.sift_match[id_n])
                                else:

                                    object_id = self.new_id-none_count3+1
                                    none_count3 -=1


                                track_id.append(object_id)






                                if object_id in add_flow:
                                    if object_id in self.fe_idelist:
                                        # print("len(self.fe_idelist[trackid[0]])",len(self.fe_idelist[track_id[0]]))
                                        if len(self.fe_idelist[object_id]) < 50:
                                            self.fe_idelist[object_id].append([new_point])

                                            add_flow[object_id].append(new_point)

                                            new_addpoint.append([new_point])



                                            feature_list2.append([new_point,old_point,track_id])
                                        else:
                                            break
                                else:

                                    if object_id in self.fe_idelist:
                                        # print("len(self.fe_idelist[trackid[0]])",len(self.fe_idelist[track_id[0]]))

                                        if len(self.fe_idelist[object_id]) < 50:
                                            add_flow.setdefault(object_id, [new_point])
                                            self.fe_idelist[object_id].append([new_point])
                                            new_addpoint.append([new_point])



                                            feature_list2.append([new_point,old_point,track_id])
                                        else:
                                            break
                                    else:
                                        add_flow.setdefault(object_id, [new_point])
                                        self.fe_idelist.setdefault(object_id, [new_point])

                                        new_addpoint.append([new_point])



                                        feature_list2.append([new_point,old_point,track_id])



                mask_img = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)



                p_new = cv2.goodFeaturesToTrack(next, mask = mask_img, **feature_params2)

                print("特徴点追加")
                print("追加ID ",self.new_key)
                # mask = img.copy()
                count2= 0
                print("rematch" , self.re_match)



                for add_point in p_new :

                    # print("add_point",add_point)



                    a,b = add_point.ravel()
                    # a, b = get_point.pt
                    new_point = a,b
                    old_point = a,b
                    # print("new",new_point)
                    track_id = []


                    for track_result in mottracks:
                        tracker_id = self.track_id_dict[track_result.id]
                        bbox = track_result.box
                        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        reactangle2 = [x1, y1,x2-x1,y2-y1]
                        object_id = ""


                        if is_point_inside_bounding_box(new_point, reactangle2):
                            object_id = tracker_id

                            if object_id != "":
                            # cal = cv2.circle(result,(int(a),int(b)),5,(255,255,255),-1)

                                track_id.append(int(object_id))

                                # new_addpoint.append([new_point])


                                if track_id[0] in add_flow:
                                    if track_id[0] in self.fe_idelist:
                                        # print("len(self.fe_idelist[trackid[0]])",len(self.fe_idelist[track_id[0]]))
                                        if len(self.fe_idelist[track_id[0]]) < 50:
                                            self.fe_idelist[track_id[0]].append([new_point])

                                            add_flow[track_id[0]].append(new_point)

                                            new_addpoint.append([new_point])



                                            feature_list2.append([new_point,old_point,track_id])
                                        else:
                                            break
                                else:

                                    if track_id[0] in self.fe_idelist:
                                        # print("len(self.fe_idelist[trackid[0]])",len(self.fe_idelist[track_id[0]]))

                                        if len(self.fe_idelist[track_id[0]]) < 50:
                                            add_flow.setdefault(track_id[0], [new_point])
                                            self.fe_idelist[track_id[0]].append([new_point])
                                            new_addpoint.append([new_point])



                                            feature_list2.append([new_point,old_point,track_id])
                                        else:
                                            break
                                    else:
                                        add_flow.setdefault(track_id[0], [new_point])
                                        self.fe_idelist.setdefault(track_id[0], [new_point])

                                        new_addpoint.append([new_point])



                                        feature_list2.append([new_point,old_point,track_id])

                #特徴追加


                if new_addpoint != []:
        
                    new_addpoint = np.array(new_addpoint, dtype=np.float32)

                    self.good_new = np.concatenate((self.good_new.reshape(-1, 1, 2), new_addpoint), axis=0)
                    self.feature_box_dict[self.frame_count][0]["feature"].extend(feature_list2)






                print(len(self.good_new.reshape(-1, 1, 2)))
                print(len(self.feature_box_dict[self.frame_count][0]["feature"]))




                if len(self.curent_object_dict) !=0:
                    bbox = {"bbox": self.curent_object_dict}
                    self.feature_box_dict[self.frame_count].append(bbox)



                #５０フレーム記録したら、古いものから消す
                if len(self.feature_box_dict) > 50 :
                    # 一番前の要素を取得
                    first_key = list(self.feature_box_dict.keys())[0]
                    self.feature_box_dict.pop(first_key,None)



        #result = cv2.resize(result,(width,height))
        self.print_fps(self.result)


        #     # # メモリ使用量を取得
        #     # snapshot = tracemalloc.take_snapshot()
        #     # top_stats = snapshot.statistics('lineno')

        #     # print("[ Top 10 memory usage ]")
        #     # for stat in top_stats[:10]:
        #     #     print(stat)



        #デバック用画面の作成
        
        # self.result = np.hstack((self.result, self.testmask))
        # self.detect = np.hstack((self.flow, self.detect))

        # self.result =cv2.vconcat([self.result, self.detect])

        if self.matched_images !=[]:
            rows = []

    


            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_name = f"matched_images_grid_{current_time}.jpg"
            output_name2 = f"matched_images_grid2_{current_time}.jpg"
           
            final_save_path = os.path.join(self.output_folder, output_name)

            final_save_path2 = os.path.join(self.output_folder, output_name2)


            print("gazou",len( self.matched_images ))

            for i in range(0, len(self.matched_images), 2):
                row_images = self.matched_images[i:i+2]
                # for mask in masks_shaped:
                #     the_mask = mask.copy()
                #     the_mask = np.stack([the_mask] * 3,axis=-1)
                #     row_images[the_mask[:, :, 0] > 0.5] = row_images[the_mask[:, :, 0] > 0.5] * 0.5 + np.array(color) * 0.5



                # 列の不足分を黒画像で埋める
                if len(row_images) < 2:
                    h, w, c = row_images[0].shape
                    black_image = np.zeros((h, w, c), dtype=np.uint8)
                    row_images.append(black_image)
                
                # 横に並べる
                rows.append(cv2.hconcat(row_images))
            
            # 縦に並べる
            final_result = cv2.vconcat(rows)

            self.print_fps(final_result)



            cv2.imwrite(final_save_path, final_result)

            cv2.imwrite(final_save_path2,self.detect)


            # cv2.namedWindow('Match', cv2.WINDOW_NORMAL)
            # cv2.imshow('Match', final_result)


        # if self.matched_images2 !=[]:
        #     rows = []

    


        #     current_time2 = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        #     output_name = f"matched_images_grid_{current_time2}.jpg"
           
        #     final_save_path = os.path.join(self.output_folder2, output_name)


        #     print("gazou",len( self.matched_images2 ))

        #     for i in range(0, len(self.matched_images2), 2):
        #         row_images = self.matched_images2[i:i+2]

        #         # 列の不足分を黒画像で埋める
        #         if len(row_images) < 2:
        #             h, w, c = row_images[0].shape
        #             black_image = np.zeros((h, w, c), dtype=np.uint8)
        #             row_images.append(black_image)
                
        #         # 横に並べる
        #         rows.append(cv2.hconcat(row_images))
            
        #     # 縦に並べる
        #     final_result = cv2.vconcat(rows)

        #     self.print_fps(final_result)



        #     cv2.imwrite(final_save_path, final_result)
            



            # final_result = np.vstack(self.matched_images)

            # cv2.namedWindow('Match', cv2.WINDOW_NORMAL)
            # cv2.imshow('Match', final_result)

            







            







        cv2.namedWindow('OpenCV Capture', cv2.WINDOW_NORMAL)
        cv2.imshow("OpenCV Capture", self.result)






        # cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
        # cv2.imshow("mask",self.mask)





        #print(len(self.frame_object_list))

        #Publish情報

        # if self.frame_object_list:
        #     detected_object_list = self.create_msg(self.frame_object_list, detected_object_list, frame)

        # self.detection_publisher.publish(detected_object_list)




        print("終わり")


        # cap.release()
        #cv2.destroyAllWindows()
        cv2.waitKey(1)
        return self.result



    def  create_msg(self, frame_object_list: List[FrameObject], detected_object_list: DetectedObjectList, frame: ColorImageFrame) -> DetectedObjectList:
        for frame_object in frame_object_list:
            action, bounding_box_src, size, mask_img, time, class_id, object_id= frame_object.item.items
            x, y, width, height = bounding_box_src.items

            detected_object = DetectedObject()
            detected_object.action = action.value
            detected_object.mask = self.bridge.cv2_to_compressed_imgmsg(mask_img, 'png')
            #print(detected_object.action)
            bounding_box = BoundingBox()
            bounding_box.x = float(x)
            bounding_box.y = float(y)
            bounding_box.width = float(width)
            bounding_box.height = float(height)
            self.ta_box = [bounding_box.x,bounding_box.y,bounding_box.width,bounding_box.height]
            self.ta_object_id =object_id
            detected_object.bounding_box = bounding_box
            print(object_id)

            detected_object.object_id = str(object_id)

            detected_object_list.object_list.append(detected_object)

            self.frame_object_list.remove(frame_object)

            if self.is_debug_mode:
                try:
                    item_color_img = frame.new_image if action == DetectedObjectActionEnum.BRING_IN else frame.old_image
                    print('イベントが検出されました(',
                        f'action: {action.value}, x: {x}, y: {y}, width: {width}, height: {height}, size: {size},class_id:{class_id})')
                    icon = np.zeros((height + 10, width, 3), dtype=np.uint8)
                    icon[0:height, 0:width, :] = item_color_img[y:y + height, x:x + width, :]

                    img_height, img_width = item_color_img.shape[:2]
                    icon = cv2.resize(icon.copy(), (img_width // 2, img_height // 2))
                    cv2.putText(icon, f'Action : {action.value}', (0, img_height // 2 - 5), cv2.FONT_HERSHEY_PLAIN, 1.5,(255, 255, 255), thickness=2)

                    self.object_list[self.object_index] = icon
                    self.object_index = (self.object_index + 1) % 4
                except Exception as e:
                    print(e)


                #for bbox in frame_object_list:
                    #color = random.choice(self._colors)
                    #result_img = cv2.rectangle(frame.new_image, (x, y), (x + width, y + height), color, thickness=3)
                #brack_img = np.zeros_like(frame.new_image)
                #img = self.print_fps(brack_img)
                #tile_img = cv2.hconcat([result_img, img])
                #cv2.namedWindow('yolox_object_detection', cv2.WINDOW_NORMAL)
                #cv2.imshow("yolox_object_detection", tile_img)
                #cv2.waitKey(1)


            return detected_object_list
#
# main
#
def main(args=None):
    rclpy.init(args=args)

    capture_node = CaptureNode()

    capture_node.process_video()


    try:
        rclpy.spin(capture_node)
    except KeyboardInterrupt:
        pass

    finally:

        # 終了処理
        capture_node.destroy_node()
        rclpy.shutdown()



#sift segment
def get_segmented_sift(mask_img,kp_point,kp,des): #マスク画像、キーポイントの座標郡、キーポイント、記述子
    segmented_kp = []
    segmented_kp_point = []
    segmented_des = []
    #print("shape",mask_img.shape)

    for k in range(len(kp_point)):
        x_i = int(kp_point[k][0])
        y_i = int(kp_point[k][1])
        #print("mask_img",mask_img.shape)
        #print("x_i",x_i)
        #print("y_i",y_i)
        if mask_img[y_i][x_i] == 1 and mask_img[y_i-1][x_i] == 1 and mask_img[y_i+1][x_i] == 1 and mask_img[y_i-1][x_i+1] == 1 and mask_img[y_i-1][x_i-1] == 1 and mask_img[y_i+1][x_i-1] == 1 and mask_img[y_i+1][x_i+1] == 1 and mask_img[y_i][x_i-1] == 1 and mask_img[y_i][x_i+1] == 1:
            segmented_kp.append(kp[k])
            segmented_kp_point.append([kp_point[k][0],kp_point[k][1]])
            segmented_des.append(des[k])

    return segmented_kp , segmented_kp_point , segmented_des


def is_point_inside_bounding_box(point, bbox):
    x, y = point


    # 矩形の頂点を取得
    rect_x, rect_y, rect_width, rect_height = bbox
    rect_top_left = (rect_x, rect_y)
    rect_top_right = (rect_x + rect_width, rect_y)
    rect_bottom_left = (rect_x, rect_y + rect_height)
    rect_bottom_right = (rect_x + rect_width, rect_y + rect_height)

    # 線分が矩形の内部にあるかチェック
    if point_in_rectangle(point, bbox) :
        return True

    return False


def intersect(rectangle, segment):
    # 矩形の頂点を取得
    rect_x, rect_y, rect_width, rect_height = rectangle
    rect_top_left = (rect_x, rect_y)
    rect_top_right = (rect_x + rect_width, rect_y)
    rect_bottom_left = (rect_x, rect_y + rect_height)
    rect_bottom_right = (rect_x + rect_width, rect_y + rect_height)

    # 線分の端点を取得
    seg_x1, seg_y1, seg_x2, seg_y2 = segment
    seg_start = (seg_x1, seg_y1)
    seg_end = (seg_x2, seg_y2)

    # 線分が矩形の内部にあるかチェック
    if point_in_rectangle(seg_start, rectangle) or point_in_rectangle(seg_end, rectangle):
        return True

    # 線分と矩形の各辺との交差判定
    if line_segment_intersect(seg_start, seg_end, rect_top_left, rect_top_right):
        return True
    if line_segment_intersect(seg_start, seg_end, rect_top_right, rect_bottom_right):
        return True
    if line_segment_intersect(seg_start, seg_end, rect_bottom_right, rect_bottom_left):
        return True
    if line_segment_intersect(seg_start, seg_end, rect_bottom_left, rect_top_left):
        return True

    return False


def point_in_rectangle(point, rectangle):
    x, y = point
    y = y #+311
    rect_x, rect_y, rect_width, rect_height = rectangle
    return rect_x <= x <= rect_x + rect_width and rect_y <= y <= rect_y + rect_height


def line_segment_intersect(seg1_start, seg1_end, seg2_start, seg2_end):
    # 2つの線分の方程式の係数を計算
    a1, b1, c1 = line_equation(seg1_start, seg1_end)
    a2, b2, c2 = line_equation(seg2_start, seg2_end)


    # 交差点の座標を計算
    try:
        x = (b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1)
        y = (a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1)
    except ZeroDivisionError:
        x = 0
        y = 0

    # 交差点が線分の内部にあるかチェック
    if (min(seg1_start[0], seg1_end[0]) <= x <= max(seg1_start[0], seg1_end[0]) and
            min(seg1_start[1], seg1_end[1]) <= y <= max(seg1_start[1], seg1_end[1]) and
            min(seg2_start[0], seg2_end[0]) <= x <= max(seg2_start[0], seg2_end[0]) and
            min(seg2_start[1], seg2_end[1]) <= y <= max(seg2_start[1], seg2_end[1])):
        return True

    return False

def line_equation(start_point, end_point):
    x1, y1 = start_point
    x2, y2 = end_point
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    return a, b, c

def point_in_rectangle(point, rectangle):
    x, y = point
    rect_x, rect_y, rect_width, rect_height = rectangle
    return rect_x <= x <= rect_x + rect_width and rect_y <= y <= rect_y + rect_height


def is_match(frame, other):
    frame_item_x, frame_item_y, frame_item_width, frame_item_height = frame

    other_item_x, other_item_y, other_item_width, other_item_height = other


    bbox_x = abs(frame_item_x - other_item_x)
    bbox_y = abs(frame_item_y - other_item_y)
    bbox_width = abs(frame_item_width - other_item_width)
    bbox_height = abs(frame_item_height - other_item_height)
    if (bbox_x < 10) and (bbox_y < 10)and(bbox_width < 10) and (bbox_height <10) : #& bbox_width < 30 & bbox_height < 30:
        return True
    else:

        return False

def iou(bbox, bbox2):
    # bbox, bbox2は矩形を表すリストで、a=[xmin, ymin, xmax, ymax]
    A_rect_x, A_rect_y, A_rect_xmax, A_rect_ymax = bbox
    B_rect_x, B_rect_y, B_rect_xmax, B_rect_ymax = bbox2


    a_area = (A_rect_xmax - A_rect_x + 1) * (A_rect_ymax - A_rect_y + 1)
    b_area = (B_rect_xmax - B_rect_x + 1) * (B_rect_ymax - B_rect_y + 1)

    abx_mn = max(A_rect_x, B_rect_x)
    aby_mn = max(A_rect_y, B_rect_y)
    abx_mx = min(A_rect_xmax, B_rect_xmax)
    aby_mx = min(A_rect_ymax, B_rect_ymax)
    w = max(0, abx_mx - abx_mn + 1)
    h = max(0, aby_mx - aby_mn + 1)
    intersect = w*h

    union_area = a_area + b_area - intersect
    if union_area == 0:
        return False

    iou = intersect / (a_area + b_area - intersect)
    #print(iou)


    # if  A_rect_xmax < B_rect_x or B_rect_xmax < A_rect_x or  A_rect_ymax < B_rect_ymax or B_rect_ymax < A_rect_y:
    #     return False
    # return True
    return iou > 0.6

def calculate_overlap(point, bbox ):
    x,y = point

    rect_x, rect_y, rect_width, rect_height = bbox
    overlap_count = 0
    if rect_x <= x <= rect_x + rect_width and rect_y <= y <= rect_y + rect_height:
        overlap_count += 1
    return overlap_count

def check_overlap(box1, box2):
    """
    2つのBounding boxが重なっているかどうかを判定する関数

    Parameters:
        box1 (tuple): 最初のBounding boxの座標 (x_min, y_min, x_max, y_max)
        box2 (tuple): 2つ目のBounding boxの座標 (x_min, y_min, x_max, y_max)

    Returns:
        bool: 2つのBounding boxが重なっていればTrue、重なっていなければFalseを返します。
    """
    x_min1, y_min1, width, height = box1
    x_max1 = x_min1 + width
    y_max1 = y_min1 + height

    x_min2, y_min2, width2, height2 = box2

    x_max2 = x_min2 + width2
    y_max2 = y_min2 + height2

    # 重なっていない場合の条件
    if (x_max1 < x_min2) or (x_min1 > x_max2) or (y_max1 < y_min2) or (y_min1 > y_max2):
        return False
    else:
        return True


def is_unknown_object(class_id: str, probability: float, box=[], img_size=(720,1280),object_threshold=0.80) -> bool:
    """物体と思われるものの規定の物体でないものかどうか調べる関数
    Args:
        class_id (str): 物体のクラス名
        probability (float): 物体かどうかの確からしさ（max 1）
        object_threshold (float): 物体と判定するしきい値（max 1）
    Returns:
        bool: 物体と思われるものの規定の物体でないものかどうか
    """
    #DEFAULT_OBJECTS = ["banana",],"person"'chair''book''keyboard','laptop','microwave'
    #img_size=(720,1280)
    DEFAULT_OBJECTS = ['chair',"bowl",'keyboard',"teddy bear",'microwave','book',"toilet","cell phone","remote",'tie','bottle','cup','laptop','sink','refrigerator','dining table','tv','potted plant','mouse']#["person",'staffed toy','chair',"bed","handbag","backpack","banana","remote","spoon",'dog','cat','laptop','tv','microwave','refrigerator','potted plant','cup','couch','mouse','sink','dining table','skateboard','bottle','cell phone','knife','bowl']
    is_object: bool = probability > object_threshold
    is_default_object = class_id in DEFAULT_OBJECTS
    img_height, img_width = img_size
    edge_box = True

    if class_id == 'person':
        # print("img_size",img_size)
        # print("box[0]",box[0])
        # print("box[1]",box[1])
        # print("box[2]",box[2])
        # print("box[3]",box[3])
        if box[0] <= 0  or box[2] >= img_width-20:  #  or box[1] <= 0 box[3] >= img_height
            edge_box = False  # 見切れている
            # print("img_size",img_size)
            # print("box[0]",box[0])
            # print("box[1]",box[1])
            # print("box[2]",box[2])
            # print("box[3]",box[3])

            print("cheak", edge_box)
        else:
            edge_box = True  # 完全に画面内





    #print(is_object and not(is_default_object))
    return is_object and not(is_default_object) and edge_box


def filter_boxes(boxes) -> bool:
    """
    Filter out boxes with IoU greater than a given threshold.

    Parameters
    ----------
    boxes : list of lists or numpy array
        List of bounding boxes, where each box is represented as [x1, y1, x2, y2].
    iou_threshold : float
        The IoU threshold above which boxes will be filtered out.

    Returns
    -------
    filtered_boxes : list of lists
        The filtered list of bounding boxes.
    """
    filtered_boxes = []
    for i, box in enumerate(boxes):
        keep = True
        x = box.xmin
        y = box.ymin
        xmax = box.xmax
        ymax = box.ymax
        height = ymax - y
        # print(height)
        width = xmax - x
        class_id = box.class_id
        object_id = ""
        # print("class",class_id)

        reactangle = [x,y, width, height]


        for j, other_box in enumerate(boxes):
            x2 = box.xmin
            y2 = box.ymin
            xmax2 = box.xmax
            ymax2 = box.ymax
            height2 = ymax - y
            # print(height)
            width2 = xmax - x
            class_id2 = box.class_id
            object_id = ""

            reactangle2 = [x2,y2, width2, height2]
            if i != j and iou(reactangle, reactangle2)  and class_id == class_id2:
                keep = False
                break
        if keep:
            filtered_boxes.append(box)
    return filtered_boxes

def resolve_conflicts(d):

    new_dict = {}

    # 'or'を含む値を持つキーのリスト
    or_keys = [key for key, value in d.items() if ' or ' in value]

    # 辞書の各要素を走査
    for key, value in d.items():
        if ' or ' in value:
            values = value.split(' or ')

            # 同じ値を持つキーが複数存在するかチェック
            same_or_values = [k for k in or_keys if d[k] == value]
            if len(same_or_values) > 1:
                new_dict[key] = value
            else:
                # すべての値が他の要素に含まれているかチェック
                if all(any(v == other_val or v in other_val.split(' or ') for other_key, other_val in d.items() if other_key != key) for v in values):
                    continue  # すべての値が他の要素に含まれている場合はキーを削除
                else:
                    new_values = [v for v in values if not any(v == other_val or v in other_val.split(' or ') for other_key, other_val in d.items() if other_key != key)]
                    if new_values:
                        new_dict[key] = ' or '.join(new_values)
        else:
            new_dict[key] = value

    return new_dict
def get_id_color(index):
        temp_index = (index + 1) * 5
        color = (
            (37 * temp_index) % 255,
            (17 * temp_index) % 255,
            (29 * temp_index) % 255,
        )
        return color


class MOT:
    def __init__(self):
        # motpy init
        self.model_spec = {'order_pos': 1, 'dim_pos': 2,
                            'order_size': 0, 'dim_size': 2,
                            'q_var_pos': 5000., 'r_var_pos': 0.1}
        self.dt =1 / 30.0  # assume 15 fps

        self.track_id_dict = {}

        self.tracker = MultiObjectTracker(dt=self.dt, model_spec=self.model_spec)



    def track(self, outputs, ratio):
        if outputs[0] is not None:
            outputs = outputs[0].npu().numpy()
            outputs = [Detection(box=box[:4] / ratio, score=box[4] * box[5], class_id=box[6]) for box in outputs]
        else:
            outputs = []

        self.tracker.step(detections=outputs)
        tracks = self.tracker.active_tracks()
        return tracks
    def bboxes2out_detections(self, bboxes:BoundingBoxes):
        out_detections = []#"person"
        DEFAULT_OBJECTS = ["chair"]
 #       logger.info("box")
        for bbox in bboxes.values():
            #print(bbox[5])
            # a,b = bbox[5][0]

            flow = bbox[5]

            class_id = bbox[6]

            print("class0")
            # print(bbox)
            print(class_id )

            #print(bbox._bounding_box)
            #bounding_box_src = bbox._bounding_box
            #x, y, width, height = bounding_box_src.items

            # out_detections.append(Detection(box=[bbox[0], bbox[1],bbox[2],bbox[3],a,b],score=bbox[4]))

            # out_detections.append(Detection(box=[bbox[0], bbox[1], bbox[0]+bbox[2],bbox[1]+bbox[3],a,b],score=bbox[4]))
            if bbox[0] == None :
                out_detections.append(Detection(box=[None,None, None,None,flow],score=None))
            else:
                out_detections.append(Detection(box=[bbox[0], bbox[1], bbox[0]+bbox[2],bbox[1]+bbox[3],flow],score=bbox[4],class_id=class_id))





#         for bbox in bboxes.bounding_boxes:

#        #    if name in DEFAULT_OBJECTS :
#       #      logger.info(bbox)
#       #      bbox_np = bbox.npu().numpy()
#       #      logger.info("bo")

#       #      conf_np = bbox.probability.npu().numpy()
#       #      logger.info("co")

#        #     cls_np = bbox.class_id.npu().numpy()
#       #      logger.info("cl")
#             width=bbox.xmax-bbox.xmin
#             height = bbox.ymax-bbox.ymin
#             #print(bbox.xmax-bbox.xmin)
#             #print(bbox.ymax-bbox.ymin)

#       #      detection_result = np.column_stack((boxes_np, confs_np, cls_np))
#             out_detections.append(Detection(box=[bbox.xmin, bbox.ymin, bbox.xmax,bbox.ymax],score=bbox.probability))
#             #out_detections.append([bbox.xmin.npu().numpy(), bbox.ymin.npu().numpy(), bbox.xmax.npu().numpy(), bbox.ymax.npu().numpy(),bbox.probability.npu().numpy(),bbox.class_id.npu().numpy()])
# #        logger.info("re")

        return out_detections #detection_result #out_detections
        #boxes = bboxes.bounding_boxes




    def create_d_msgs_box(self, track) -> BoundingBox:
        one_box = BoundingBox()

        one_box.id = int(track.id[:3], 16)
        #one_box.class_id = class_tag
        #one_box.probability = float(track.score)
        #one_box.xmin = int(track.box[0])
        #one_box.ymin = int(track.box[1])
        #one_box.xmax = int(track.box[2])
        #one_box.ymax = int(track.box[3])

        return one_box
    def publish_d_msgs(self, tracks, boxes_msg:BoundingBoxes) -> None:

        boxes = BoundingBoxes()
        boxes.header = boxes_msg.header
        boxes.probability = boxes_msg.probability
        boxes.class_id = boxes_msg.class_id
        i = 0
        if(len(tracks)==0):
            self.pub.publish(boxes)
            return

        for track in tracks:
            boxes.bounding_boxes.append(self.create_d_msgs_box(track))

        self.pub.publish(boxes)

    def get_id_color(self,index):
        temp_index = (index + 1) * 5
        color = (
            (37 * temp_index) % 255,
            (17 * temp_index) % 255,
            (29 * temp_index) % 255,
        )
        return color


    def draw_debug(self,image,track_results,track_id_dict):
        debug_image = copy.deepcopy(image)
        #logger.info("copy")
        for track_result in track_results:
            tracker_id = track_id_dict[track_result.id]
            #logger.info("track_id")
            bbox = track_result.box
#            logger.info(bbox)
            class_id = track_result.class_id
            #logger.info("class_id")
            score = track_result.score
            #logger.info("score")

            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            f1,f2 = int(bbox[4]),int(bbox[5])

            #print(x2)
            #print(y2)

            # トラッキングIDに応じた色を取得
            color = self.get_id_color(tracker_id)
            # color = self._COLORS[int(tracker_id)]
            if tracker_id != 5:

                # バウンディングボックス描画
                debug_image = cv2.rectangle(
                    debug_image,
                    (x1, y1),
                    (x2, y2),
                    color,
                    thickness=2,
                )

                # id、ラベル名描画
                track_id = '%.d' % tracker_id
                text = 'ID:%s' % (track_id)
                if class_id == "person":
                    text = 'ID:%s' % (track_id)
                debug_image = cv2.putText(
                    debug_image,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    thickness=2,
                )

        return debug_image
