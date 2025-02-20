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
import numpy as np
import random
import time
import numpy as np
import rclpy
from typing import List

from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
import message_filters

from collections import Counter
from collections import defaultdict

import sys
#print(sys.path)
sys.path.append("/home/azuma/ros2_ws/src/shigure_core/shigure_core/shigure_core/nodes")

from motpy import Detection, MultiObjectTracker

from motpy.testing_viz import draw_track

from shigure_core_msgs.msg import DetectedObjectList, DetectedObject, TrackedObjectList, TrackedObject, PoseKeyPointsList, Cube
from bboxes_ex_msgs.msg import BoundingBoxes ,Segments
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

from memory_profiler import profile
import tracemalloc


from ultralytics import YOLO

class CaptureNode(ImagePreviewNode):
    def __init__(self):
        super().__init__("yolox_object_traking_node")
        # QoS Settings
        shigure_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

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
        self.revs: np.ndarray = np.ndarray
        self.lk_reset = True
        # self.time_synchronizer = message_filters.TimeSynchronizer(
        #     [yolox_bbox_subscriber,people_subscriber, color_subscriber, depth_camera_info_subscriber], 1000)
        self.time_synchronizer = message_filters.TimeSynchronizer(
            [yolox_bbox_subscriber,color_subscriber,depth_camera_info_subscriber,segment_subscriber], 1000)
        self.time_synchronizer.registerCallback(self.callback)
        self.yolox_object_detection_logic = YoloxObjectDetectionLogic()

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

        self.clipped_images: List[np.ndarray] = []  # クリップ画像を保存するリスト

        
        self.sift_c =0
        





     #   self.bbox_item_list = []



        self.previous_object_dict = {}

        self._tracking_info = TrackingInfo()

        self.prvs_box = []
        self.ta_box: tuple[float,float,float,float]  = []

        self.bounding_box: tuple[float, float, float, float] = []

        self.frame_count = 0
        self.add_feature_every_n_frames = 5
        


        #SIFT setting
        self.FLANN_INDEX_KDTREE = 0
        self.index_params = dict(algorithm = self.FLANN_INDEX_KDTREE, trees = 5)
        self.search_params = dict(checks = 50)
        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)#マッチ検出器の定義
        self.sift = cv2.SIFT_create(nfeatures=100) #SIFT特徴点検出器の定義
        self.sift = cv2.ORB_create() #SIFT特徴点検出器の定義





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



        ## for facedetection by OpenCV Haar-like feature based face detector
        #
        # haarcascade_frontalface_default.xml from https://github.com/opencv/opencv/tree/master/data/haarcascades
        # face_cascade = cv2.CascadeClassifier('/home/azuma/ros2_ws/src/dir/data/haarcascades/haarcascade_frontalface_default.xml')
        # # haarcascade_eye.xml from https://github.com/opencv/opencv/tree/master/data/haarcascades
        # eye_cascade = cv2.CascadeClassifier('/home/azuma/ros2_ws/src/dir/data/haarcascades/haarcascade_eye.xml')
        # eye_tree = cv2.CascadeClassifier('/home/azuma/ros2_ws/src/dir/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
        # # haarcascade_upperbody.xml from https://github.com/opencv/opencv/tree/master/data/haarcascades
        # body_cascade = cv2.CascadeClassifier('/home/azuma/ros2_ws/src/dir/data/haarcascades/haarcascade_upperbody.xml')


        # profileface = cv2.CascadeClassifier('/home/azuma/ros2_ws/src/dir/data/haarcascades/haarcascade_profileface.xml')




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


    #def callback(self, yolox_bbox_src: BoundingBoxes,people: PoseKeyPointsList, color_img_src: CompressedImage, camera_info: CameraInfo):
    def callback(self, yolox_bbox_src: BoundingBoxes,color_img_src: CompressedImage, camera_info: CameraInfo, segment_src:Segments):
        self.get_logger().info('Buffering start', once=True)
        self.frame_count_up()
        color_img: np.ndarray = self.bridge.compressed_imgmsg_to_cv2(color_img_src)
        height, width = color_img.shape[:2]


        if not hasattr(self, 'object_list'):
            self.object_list = []
            black_img = np.zeros_like(color_img)
            for i in range(4):
                self.object_list.append(cv2.resize(black_img.copy(), (width // 2, height // 2)))

        self._color_img_buffer.append(color_img)
        if len(self._color_img_buffer)> 1:
            self._color_img_buffer.pop(-1)


        timestamp = Timestamp(color_img_src.header.stamp.sec, color_img_src.header.stamp.nanosec)
        frame = ColorImageFrame(timestamp, self._color_img_buffer[0], color_img) #bufferの先頭の画像と新しい画像
        # self._color_img_frames.add(frame) #ColorImageFrameslistの更新して、listに追加

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


        sec, nano_sec = frame.timestamp.timestamp
        detected_object_list = DetectedObjectList()
        detected_object_list.header.stamp.sec = sec
        detected_object_list.header.stamp.nanosec = nano_sec
        detected_object_list.header.frame_id = camera_info.header.frame_id


        # if self.frame_object_list:
        #     detected_object_list = self.create_msg(self.frame_object_list, detected_object_list, frame)

        # self.detection_publisher.publish(detected_object_list)
        #result = color_img.copy()

        # self.results8 = self.model(color_img,classes = [0,66],conf = 0.1),show=True #show=True
        # masks = self.results8[0].masks

        # for i in results8:

        #     print(results8)




        self.get_logger().info('Buffering end', once=True)

        # オプティカルフローのコード


        # lkの特徴点の更新間隔

        if( len(sys.argv) == 1 ):
            cam = 0
        else :
            cam = int(sys.argv[1])

        # if self.lk_count > 2:
        #     self.lk_reset = True
        #     self.lk_count =0

        # elif self.lk_count == 0 :
        #     self.lk_reset = True

        # else:
        #     self.lk_reset = False


        #self.lk_count = self.lk_count + 1




        #print("camera device num: %d"%cam)

        # cap = cv2.VideoCapture(color_img)
        # cap.set(cv2.CAP_PROP_FPS,30)

        # print("FPS:%f"%cap.get(cv2.CAP_PROP_FPS))ttttttttttttttttttt
        # print("Image Height:%d"%int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # print("Image Width:%d"%int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        # print("FOURCC:%d"%int(cap.get(cv2.CAP_PROP_FOURCC)))
        # proc = "test"
        proc = "lktracking"
        # proc = "color"
        # proc = "sift"
        # proc = "lu"
        ndi_send = None

        # ret,img = cap.read()
        ret = True

        img = color_img
        self.flag = True
        self.tflag = True
        self.t2flag = True

        self.newflag = False
        self.id_flag = True

        point_relia = False
        uncertain_flag = False



        # height = img.shape[0]
        # width = img.shape[1]
        # img = cv2.resize(img,((int)(width/2),(int)(height/2)))

        #self.prvs = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        #
        ## for HSV color space transform
        #
        # if self._count == 0:
        #     self.hsv = np.zeros_like(img)
        #     self.hsv[...,1] = 255
        #     colors_hsv = [[0,0,0]]
        #     for i in range(1, 256):
        #         colors_hsv.append(np.array([random.randint(0,180), random.randint(120,255), 255]))
        #     colors_hsv = np.array(colors_hsv).astype(np.uint8)


        profile_face = True

        #
        ## for LK tracking
        #
        #ShiTomasiコーナー検出器のためのパラメータ
        lk_fnum = 1000
        lk_fnum2 = 500
        feature_params = dict( maxCorners = lk_fnum,
                            qualityLevel = 0.001,
                            minDistance = 3,
                            blockSize = 7 )
        feature_params2 = dict( maxCorners = lk_fnum2,
                    qualityLevel = 0.001,
                    minDistance = 7,
                    blockSize = 7 )
        # Lucas-Kanade法によるオプティカル・フローのためのパラメータ
        lk_params = dict( winSize  = (21,21),
                        maxLevel = 4,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        # random color map
        lk_color = np.random.randint(0,255,(lk_fnum,3))

        #
        ## for Bluring preprocess of input image
        #
        blur_proc = "gaussian"
        gaussian_scale = 1
        median_scale = 5
        bilateral_scale = 20

        #
        ## for histogram equalization
        #
        clahe = None
        self.new_id = 0
        self.brack_img = self.seg_mask.copy()



        #
        ## main loop
        #

        # t_prev = time.perf_counter()


        # t_now = time.perf_counter()
        # duration = t_now-t_prev
        # t_prev = t_now

        # image capture
        # ret,img = cap.read()


        # resize for processing rate
        #img = cv2.resize(img,((int)(width/2),(int)(height/2)))


        #the newest key entry accepted, others flushed
        # key = -1
        # while True:
        #     next_key = cv2.waitKey(1)
        #     if next_key != -1 :
        #         key = next_key
        #     else:
        #         break

        # key = key&0xFF

        # if(ret==True):
        #     if key == ord('q') :
        #         print("a")
        #         # break
        #     elif key == ord('g') : # Monocolor
        #         proc = "gray"
        #     elif key == ord('c') : # Color
        #         proc = "color"

        #     elif key == ord('t') : # Lukas-Kanade feature tracking
        #         proc = "lktracking"
        #         # lk_reset = True

        #     # blurring preprocessing
        #     if blur_proc == "gaussian" :
        #         img = cv2.GaussianBlur(img.astype(np.float32),(gaussian_scale,gaussian_scale),0).astype(np.uint8)
        #         #img = cv2.GaussianBlur(img.astype(np.float32),(0,0),gaussian_scale).astype(np.uint8)
        #     elif blur_proc == "median" :
        #         img = cv2.medianBlur(img,median_scale)
        #     elif blur_proc == "bilateral" :
        #         img = cv2.bilateralFilter(img.astype(np.float32),bilateral_scale,75,75).astype(np.uint8)


        # image processing
        
        if proc == "color" :
            result = img
            next = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mot = MOT()

            if( self.lk_reset == True ):
                if self._count == 0:
                    self.prvs = next

                    self._count = self._count + 1

                # if len( self.track_list)> 10:
                #     self.track_list = self.track_list[5:]




                p0 = cv2.goodFeaturesToTrack(next, mask = None, **feature_params)
                print(p0)



                self.lk_reset = False

            else :

                p0 = self.good_new.reshape(-1,1,2)
                #print(p0)


            # オプティカル・フローを計算

            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prvs, next, p0, None, **lk_params)

            # 良い特徴点を選択
            self.good_new = p1 #[st==1]
            self.good_old = p0 #[st==1]

            # print(len(self.good_new))
            # print(len())


            feature_list = []
            track = np.zeros_like(img)
            track2 = np.zeros_like(img)


            yolox_bboxes = yolox_bbox_src.bounding_boxes #yolox-rosから受け取った物体集合から物体一つずつ取り出す
            yolox_bboxes = filter_boxes(yolox_bboxes)
            bbox_count = 0
            bbox_item_list =[]
            #print(len(yolox_bboxes))
            box_count_t = 0

            for id, bbox in enumerate(yolox_bboxes):
                probability = bbox.probability
                x = bbox.xmin
                y = bbox.ymin
                xmax = bbox.xmax
                ymax = bbox.ymax
                height = ymax - y
                width = xmax - x
                class_id = bbox.class_id
                object_id = ""

                if is_unknown_object(class_id, probability) and height < 500:
                    box_count_t += 1
                    #print("1frame")
                    bbox_count += 1
                    brack_img = np.zeros(color_img.shape[:2])
                    # brack_img[y:y + height, x:x + width] = 255
                    mask_img:np.ndarray = brack_img[y:y + height, x:x + width]

                    # BBOX(左上端座標, 幅, 高さ)
                    bounding_box = BoundingBox(x, y, width, height)
                    area = width*height # BBOXの面積

                    object_id = bbox_count

                    test_item = [x,y, width, height]

                    box_flow = []



                    for i,(new,old) in enumerate(zip(self.good_new,self.good_old)):
                        total_overlap_count = 0
                        track_id = []


                        a,b = new.ravel()
                        new_point = a,b #+211
                        c,d = old.ravel()

                        old_point = c,d #+211
                        colar =(0,255,0)
                        # cv2.line(track, (int(a),int(b)),(int(c),int(d)), colar, 2)
                        # result = cv2.circle(result,(int(a),int(b)),5,colar,-1)


                        reactangle = [x,y, width, height]


                        if is_point_inside_bounding_box(new_point, reactangle):
                            box_flow.append(new_point)
                            # print(new_point)
                            self.b_old = old_point

                            cv2.line(track, (int(a),int(b)),(int(c),int(d)), colar, 2)
                            result = cv2.circle(result,(int(a),int(b)),5,colar,-1)




                            # track_id.append(key)
                            # bbox_item = [x,y, width, height, probability, box_flow]


                            # cv2.line(track, (int(a),int(b)),(int(c),int(d)), colar, 2)
                            # result = cv2.circle(result,(int(a),int(b)),5,colar,-1)
                            if len(box_flow) > 3:
                                break
                            # break



                    if len(box_flow) < 4:
                        print("no")
                        p_new = cv2.goodFeaturesToTrack(next, mask = None, **feature_params2)
                        for id, bbox in enumerate(yolox_bboxes):
                            probability = bbox.probability
                            x = bbox.xmin
                            y = bbox.ymin
                            xmax = bbox.xmax
                            ymax = bbox.ymax
                            height = ymax - y
                            width = xmax - x
                            class_id = bbox.class_id
                            object_id = ""

                            if is_unknown_object(class_id, probability) and height < 500:

                                box_count_t += 1
                                #print("1frame")
                                bbox_count += 1
                                brack_img = np.zeros(color_img.shape[:2])
                                brack_img[y:y + height, x:x + width] = 255
                                mask_img:np.ndarray = brack_img[y:y + height, x:x + width]

                                # BBOX(左上端座標, 幅, 高さ)
                                bounding_box = BoundingBox(x, y, width, height)
                                area = width*height # BBOXの面積

                                object_id = bbox_count

                                test_item = [x,y, width, height]

                                box_flow = []



                                for new_p in p_new:
                                    total_overlap_count = 0
                                    track_id = []


                                    a,b = new_p.ravel()
                                    new_point = a,b #+211

                                    colar =(0,255,0)
                                    # cv2.line(track, (int(a),int(b)),(int(c),int(d)), colar, 2)
                                    # result = cv2.circle(result,(int(a),int(b)),5,colar,-1)


                                    reactangle = [x,y, width, height]


                                    if is_point_inside_bounding_box(new_point, reactangle):

                                        box_flow.append(new_point)
                                        self.good_new= np.append(self.good_new, new_point)
                                        if len(box_flow) > 3:
                                            break


                    # bbox_item = [x,y, width, height, probability, self.good_new]

                    bbox_item = [x,y, width, height, probability, box_flow]
                    bbox_item_list.append(bbox_item)
                    self.flow_p = box_flow

                    #Track用box辞書
                    #self.curent_object_dict[object_id] = bbox_item

            # print(bbox_item_list)


            if box_count_t == 0 :
                flow2 = []
                print("nonebox")

                for i,(new,old) in enumerate(zip(self.good_new,self.good_old)):
                    a,b = new.ravel()
                    new_point = a,b #+211
                    c,d = old.ravel()

                    old_point = c,d #+211
                    if old_point in self.flow_p:
                        flow2.append(new_point)

                if len(flow2)==0:
                    bbox_item =[]
                else:

                    bbox_item = [None,None, None, None, None,flow2]
                    bbox_item_list.append(bbox_item)

            print(bbox_item_list)



            motdetections = mot.bboxes2out_detections(bbox_item_list)
            self.mottracker.step(motdetections)
            mottracks = self.mottracker.active_tracks(min_steps_alive=1)
            #print(len(mottracks))

            for track_result in mottracks:
                if track_result.id not in self.track_id_dict:
                    new_id = len(self.track_id_dict)
                    self.track_id_dict[track_result.id]= new_id
            result = mot.draw_debug(result,mottracks,self.track_id_dict)

            # for r_tc in  mottracks:
            #     print("dffjjjjj")
            #     print(r_tc.box )
            #     bbox = r_tc.box
            #     c ,d = self.b_old
            #     colar =(0,0,255)
            #     #print(int(bbox[5]))

            #     cv2.line(track, (int(bbox[4]),int(bbox[5])),(int(c),int(d)), colar, 2)
            #     result = cv2.circle(result,(int(bbox[4]),int(bbox[5])),5,colar,-1)



            self.track_list.append(track)
            if( len(self.track_list)> 10 ):
                self.track_list.pop(0)

            for t in self.track_list :
                result = np.where(t!=0,t,result)

            self.prvs = next




        elif proc == "test" :
            result = img
            next = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = np.zeros_like(img)
            contours = [np.array([[50, 50], [150, 50], [150, 150], [50, 150]])]  # 多角形例
            cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
            print(mask.shape)  # 例: (480, 640)
            print(img.shape)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            # cv2.rectangle(mask, (50, 50), (200, 200), 255, -1) 
            print(mask.dtype)

            corners = cv2.goodFeaturesToTrack(
                next,  # 入力画像
                maxCorners=100,  # 最大検出数
                qualityLevel=0.01,  # コーナー品質の閾値
                minDistance=10,  # コーナー間の最小距離
                mask=mask  # マスクを指定
            )

            # 4. 検出結果を描画
            
            if corners is not None:
                for corner in corners:
                    x, y = corner.ravel()
                    cv2.circle(result, (int(x), int(y)), 5, (0, 0, 255), -1)  # コーナーを赤丸で描画
            cv2.imshow("Mask", mask)


        # elif proc == "gray" :
        #     result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # elif proc == "denseflow" :

        #     self.next = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #     self.flow = cv2.calcOpticalFlowFarneback(self.prvs,self.next, None, 0.5, 3, 15, 3, 5, 1.2, 0)



        #     mag, ang = cv2.cartToPolar(self.flow[...,0], self.flow[...,1])
        #     # colored denseflow
        #     self.hsv[...,0] = ang*180/np.pi/2
        #     self.hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        #     rgb = cv2.cvtColor(self.hsv,cv2.COLOR_HSV2BGR)
        #     result = cv2.addWeighted(img,0.5,rgb,0.5,0)
        #     flowx = mag*np.cos(ang)
        #     flowy = mag*np.sin(ang)
        #     #result = (np.minimum(np.abs(flowx)*100,255)).astype(np.uint8)
        #     for i in range(0,result.shape[0],15):
        #         for j in range(0,result.shape[1],15):
        #             # try:
        #                 cv2.line(result,(j,i),(int(j+flowx[i,j]*3),int(i+flowy[i,j]*3)),
        #                         (255,255,255),1)
        #                 print(int(j+flowx[i,j]*3))
        #             # except Exception as e:
        #             #     pass
        #                 # print(e)
        #     self.prvs = self.next

        elif proc == "sift" :
            scale_factor = 0.5
            # img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

            # yolox_bboxes = yolo_segments.bounding_boxes #yolox-rosから受け取った物体集合から物体一つずつ取り出す
            # yolox_bboxes = filter_boxes(yolox_bboxes)
            yolo_segments = segment_src.segments
            yolo_segments = filter_boxes(yolo_segments)
            # 各テンプレート画像にIDを割り当て、色を設定
            template_colors = [
                (255, 0, 0),   # 青
                (0, 255, 0),   # 緑
                (0, 0, 255),   # 赤
                (255, 255, 0), # シアン
                (255, 0, 255), # マゼンタ
                (0, 255, 255), # 黄
                # 必要に応じて色を追加
            ]


            if self.first_time == 0 :
                self.first_time += 1
                
                boxes = []
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

                    if is_unknown_object(class_id, probability) and height < 500:
                        boxes.append([x, y, xmax, ymax])
                        clipped_image = img[y:ymax, x:xmax]
                        self.clipped_images.append(clipped_image)  # リストに追加
                
                # self.img1 = self.clipped_images[1] #テンプレート画像
                
                # self.kp1, self.des1 = self.sift.detectAndCompute(self.img1,None) #テンプレート画像の特徴点算出
                result = img
                self.first_frame = img.copy()


            
            else:
                # self.f_frame = self.first_frame.copy()
                # テンプレート画像を左上から横に並べるための初期位置
                x_offset = 10  # 最初のテンプレート画像のX方向のオフセット
                y_offset = 10  # 最初のテンプレート画像のY方向のオフセット
                spacing = 10   # テンプレート画像同士の間隔
                self.h =0
                self.w = 0

            
                result = img.copy()
                result_images = []
                next = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                temp_img = cv2.cvtColor(self.clipped_images[1], cv2.COLOR_BGR2GRAY)
                img2 =  img.copy() #入力画像
                # img1 = self.clipped_images[1] #テンプレート画像
                good = [] #良い特徴点マッチのリスト
                self.assigned_template_ids = []

                for template_id, self.img1 in enumerate(self.clipped_images):

                    h, w = self.img1.shape[:2]
                                
                    x_offset = self.w +spacing 
                    self.h = y_offset + h
                    self.w = x_offset + w
                    overlay_region = result[y_offset:self.h, x_offset:self.w ]

                        # テンプレート画像をコピーして配置
                    if self.img1.shape[0] >= overlay_region.shape[0] and  self.img1.shape[1] >= overlay_region.shape[1]:
                        template_img_resized = cv2.resize(self.img1, (w, h))
                        np.copyto(overlay_region, template_img_resized)
                    
                boxes = []


                for id, bbox in enumerate(yolo_segments ):
                    probability = bbox.probability
                    x = bbox.xmin
                    y = bbox.ymin
                    xmax = bbox.xmax
                    ymax = bbox.ymax
                    height = ymax - y
                    width = xmax - x
                    class_id = bbox.class_id
                    # print(x)
                    # print(y)
                    # print(xmax)
                    # print(ymax)
                    if is_unknown_object(class_id, probability) and height < 500:
                        roi = img2[y:ymax, x:xmax]

                        #BoundingBox
                        self.best_match_count = 0
                        self.best_match_template_id = None
                        self.best_match_ratio= 0.0 
                        
                        
                        # print(img2.shape)
                        gray_img2 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                        kp2, des2 = self.sift.detectAndCompute(gray_img2,None)  #入力画像の特徴点算出

                        for point in kp2:
                            point.pt = (point.pt[0] + x, point.pt[1] + y) 

                        x_offset = 10  # 最初のテンプレート画像のX方向のオフセット
                        y_offset = 10  # 最初のテンプレート画像のY方向のオフセット
                        spacing = 10   # テンプレート画像同士の間隔
                        self.h =0
                        self.w = 0



                        for template_id, self.img1 in enumerate(self.clipped_images):
                            colar = template_colors[template_id % len(template_colors)]

                            h, w = self.img1.shape[:2]
                                        
                            x_offset = self.w +spacing 
                            # self.h = y_offset + h
                            self.w = x_offset + w
                            # overlay_region = result[y_offset:self.h, x_offset:self.w ]

                            #     # テンプレート画像をコピーして配置
                            # template_img_resized = cv2.resize(self.img1, (w, h))
                            # np.copyto(overlay_region, template_img_resized)

                            self.kp1, self.des1 = self.sift.detectAndCompute(self.img1,None) #テンプレート画像の特徴点算出
                            self.des1 = np.asarray(self.des1, dtype=np.float32)
                            des2 = np.asarray(des2, dtype=np.float32)

                            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)



                            # matches = bf.match(self.des1,des2)
                            
                            matches = self.flann.knnMatch(self.des1,des2,k=2) #特徴点マッチを行う

                            good = []

                            for m,n in matches:
                                if m.distance < 0.7*n.distance:
                                    good.append(m)
                            
                            match_ratio = len(good) / len(self.kp1) if len(self.kp1) > 0 else 0
                            

                            MIN_MATCH_COUNT = 5 #最低限マッチしてほしい数
                            if match_ratio > self.best_match_count:
                                self.best_match_count = match_ratio
                                self.best_match_template_id = template_id

                            if len(good)>MIN_MATCH_COUNT:
                                src_pts = np.float32([ self.kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                                matchesMask = mask.ravel().tolist()



                                



                                # マッチングの対応点を全体画像に描画
                                


                                
                                for i, m in enumerate(good):
                                    if matchesMask[i]:  # マッチした点だけ描画
                                        pt1 = tuple(np.int32(src_pts[i][0]))  # テンプレート画像の点
                                        pt2 = tuple(np.int32(dst_pts[i][0]))  # 1フレーム目の全体画像の対応する点
                                      
                                        
                                        pt1_adjusted = (pt1[0] + x_offset, pt1[1] + y_offset)
                                        colar =get_id_color(template_id)
                                        # pt1_adjusted2 = (pt11[0] + x_offset, pt11[1] + y_offset)

                                        # cv2.circle(result, pt1_adjusted, 5, colar, 2)

                                        # cv2.circle(result, pt1_adjusted, 5, colar,-1)  # 対応点を青で描画

                                        # 対応する特徴点を線で結び、点も描画
                                        # cv2.line(result, pt1_adjusted, pt2, colar, 1, cv2.LINE_AA)
                                        cv2.circle(result, pt2, 5, colar, -1)  # 対応点を青で描画
                                    else:
                                        colar =get_id_color(template_id)
                                        pt11 = tuple(np.int32(src_pts[i][0]))
                                        pt11_adjusted2 = (pt11[0] + x_offset, pt11[1] + y_offset)
                                        # cv2.circle(result, pt11_adjusted2, 5, colar, 2)
                        
              
                        # id、ラベル名描画
                        if self.best_match_template_id != None:
                            colar = get_id_color(self.best_match_template_id )
                            # cv2.rectangle(result, (x, y), (xmax, ymax), colar, thickness=3)
                            print(self.assigned_template_ids)
                            # 既にそのテンプレートIDが他の領域に割り当てられていないかを確認
                            if self.best_match_template_id not in self.assigned_template_ids and class_id !="person":
                                cv2.rectangle(result, (x, y), (xmax, ymax), colar, thickness=3)
                                self.assigned_template_ids.append(self.best_match_template_id)


                                track_id = '%.d' % self.best_match_template_id
                                text = 'ID:%s' % (track_id)
                                result = cv2.putText(
                                    result,
                                    text,
                                    (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    colar,
                                    thickness=2,
                                )
                            elif self.best_match_template_id not in self.assigned_template_ids and class_id =="person": 
                                # if self.sift_c == 0:

                                #     boxes.append([x, y, xmax, ymax])
                                #     clipped_image = img[y:ymax, x:xmax]
                                #     self.clipped_images.append(clipped_image)
                                #     self.sift_c += 1
                                #     self.assigned_template_ids.append(4)
                                # colar = get_id_color(4)
                                # cv2.rectangle(result, (x, y), (xmax, ymax), colar, thickness=3)

                                # track_id = '%.d' % len(self.clipped_images)
                                # text = 'ID:%s' % (track_id)
                                # result = cv2.putText(
                                #     result,
                                #     text,
                                #     (x, y - 10),
                                #     cv2.FONT_HERSHEY_SIMPLEX,
                                #     0.7,
                                #     colar,
                                #     thickness=2,
                                # )
                                print("person")


                        else: 
                            
                            # boxes.append([x, y, xmax, ymax])
                            # clipped_image = img[y:ymax, x:xmax]
                            # self.clipped_images.append(clipped_image)
                          

                            # track_id = '%.d' % len(self.clipped_images)
                            # colar = get_id_color(len(self.clipped_images) )
                            # cv2.rectangle(result, (x, y), (xmax, ymax), colar, thickness=3)
                            # text = 'ID:%s' % (track_id)
                            # result = cv2.putText(
                            #     result,
                            #     text,
                            #     (x, y - 10),
                            #     cv2.FONT_HERSHEY_SIMPLEX,
                            #     0.7,
                            #     colar,
                            #     thickness=2,
                            # )


                            print("nans")
                    
                    






                    #             if( np.sum(matchesMask) >= 4 ) :
                    #                 # テンプレート画像の角を元画像に変換
                    #                 h, w = self.img1.shape[:2]
                    #                 pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                    #                 dst = cv2.perspectiveTransform(pts, M)

                    #                 img2_with_polylines = cv2.polylines(img2.copy(), [np.int32(dst)], True, (255, 255, 0), 3, cv2.LINE_AA)

                                    
                    #         else:
                    #                 # print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
                    #                 matchesMask = None  
                            
                    #         draw_params = dict(matchColor=(0, 255, 0), matchesMask=matchesMask, flags=2)
                    #         result = cv2.drawMatches(self.img1, self.kp1, img2_with_polylines, kp2, good, None, **draw_params)
                    
                    



                   


                           








                



            # # kp1, des1 = self.sift.detectAndCompute(img1,None) #テンプレート画像の特徴点算出
            # #このあとにget_segmented_sift関数で絞り込みしてもらう
            # img1 = cv2.drawKeypoints(self.img1,self.kp1,self.img1,color=(0,0,255),flags=0)#キーポイントの記述

            # kp2, des2 = self.sift.detectAndCompute(img2,None)  #入力画像の特徴点算出

            # matches = self.flann.knnMatch(self.des1,des2,k=2) #特徴点マッチを行う
            # good = [] #良い特徴点マッチのリスト


            # for m,n in matches:
            #     if m.distance < 0.7*n.distance:
            #         good.append(m)
            

            # MIN_MATCH_COUNT = 5 #最低限マッチしてほしい数

            # if len(good)>MIN_MATCH_COUNT:
            #     src_pts = np.float32([ self.kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            #     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            #     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            #     matchesMask = mask.ravel().tolist()
            #     if( np.sum(matchesMask) >= 4 ) :
            #         # print(img1.shape)
            #         h,w,c = img1.shape
            #         pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            #         dst = cv2.perspectiveTransform(pts,M)
            
            #         img2 = cv2.polylines(img2,[np.int32(dst)],True,(255,255,0),3, cv2.LINE_AA)

            # else:
            #         # print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
            #         matchesMask = None  

            # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
            #                     singlePointColor = None,
            #                     matchesMask = matchesMask, # draw only inliers
            #                     flags = 2)

            # result = cv2.drawMatches(img1,self.kp1,img2,kp2,good,None,**draw_params)




            






        elif proc == "lu" :
            print("az")
            result = img

        elif proc == "lktracking" :
            self.fe_idelist = {} #特徴点の識別子
            # tracemalloc.start()
            mot = MOT()
            roi_list = []
            next = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            result = img.copy()
            #self.frame_count += 1
            #print(self.frame_count)

            bbox_item_list = []
            mot_item_list = []

            self.id  = []
            self.match = {}
            #cv2.rectangle(result, (0, 311), (900, 711), (255,204,102), thickness=3)


            # if len(wait_item_list) >0 :
            #     self.get_logger().info('Start', once=True)

            #     w_item =  wait_item_list[-1]
            #     bounding_box_src = w_item._bounding_box
            #     x, y, width1, height1 = bounding_box_src.items
            #     rectangle: tuple[float, float, float, float] = [x,y ,width1,height1]
            #     if len(wait_item_list) >1 :
            #         w_item2 =  wait_item_list[-2]
            #         bounding_box_src2 = w_item2._bounding_box
            #         x2, y2, width2, height2 = bounding_box_src2.items
            #         rectangle2: tuple[float, float, float, float] = [x2,y2 ,width2,height2]


            #     #self.roi = (x, y,width1+x, height1+y)

            #     # obj= next[self.roi[1]+300:self.roi[3]+300, self.roi[0]+300:self.roi[2]+300]
            #     #オプティカルフローの特徴点抽出領域を限定
            #     #obj= next[211:711, 0:900]





            if( self.lk_reset == True ):
                if self._count == 0:
                    self.prvs = next

                    self._count = self._count + 1

                # if len( self.track_list)> 10:
                #     self.track_list = self.track_list[5:]

                yolo_segments = segment_src.segments
                yolo_segments = filter_boxes(yolo_segments)
                self.bbox_count = 0

            

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









                    object_id = ""

                    if is_unknown_object(class_id, probability) and height < 500:
                        #print("1frame")

                        brack_img = img.copy()
                        
                        # brack_img[y:y + height, x:x + width] = 255
                        # mask_img:np.ndarray = brack_img[y:y + height, x:x + width]

                        self.mask = cv2.drawContours(brack_img, [m_points], -1, 255, thickness=cv2.FILLED)
                        
                        brack_img = self.seg_mask.copy()

                        mask_img = cv2.fillPoly(brack_img, [m_points], 255)  # 255は白の値

                        # BBOX(左上端座標, 幅, 高さ)
                        bounding_box = BoundingBox(x, y, width, height)
                        area = width*height # BBOXの面積

                        object_id = self.bbox_count

                        test_item = [x,y, width, height]



                        bbox_item = BboxObject(bounding_box, area, mask_img, timestamp,class_id,object_id)
                        bbox_item_list.append(bbox_item)


                        bbox_item2 = [x,y, width, height, probability,class_id]
                        mot_item_list.append(bbox_item2)

                        self.match[object_id] = str(object_id)



                        #Track用box辞書
                        self.curent_object_dict[object_id] = bbox_item
                        self.bbox_count += 1

                        #print(object_id)


                        #test_item = [x,y,xmax,ymax]


                self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
                cv2.imshow("Mask", self.mask)

                p0 = cv2.goodFeaturesToTrack(next, mask = self.mask, **feature_params)
                # print(len(p0))

                


                self.lk_reset = False

            else :

                p0 = self.good_new.reshape(-1,1,2)
                #print(p0)


            # オプティカル・フローを計算

            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prvs, next, p0, None, **lk_params)

            #print(p1)


            if self.frame_count == 1 :
                # 良い特徴点を選択
                self.good_new = p1[st==1]
                self.good_old = p0[st==1]

                yolox_bboxes = yolox_bbox_src.bounding_boxes #yolox-rosから受け取った物体集合から物体一つずつ取り出す
                yolox_bboxes = filter_boxes(yolox_bboxes)
                # self.bbox_count = 0

                # yolo_segments = segment_src.segments
                # yolo_segments = filter_boxes(yolo_segments)

            

                # for id, bbox in enumerate(yolo_segments):
                #     probability = bbox.probability
                #     x = bbox.xmin
                #     y = bbox.ymin
                #     xmax = bbox.xmax
                #     ymax = bbox.ymax
                #     height = ymax - y
                #     width = xmax - x
                #     class_id = bbox.class_id
                #     x_masks = bbox.x_masks
                #     y_masks = bbox.y_masks

                #     mask_pairs = list(zip(y_masks, x_masks))

                #     m_points = np.array(mask_pairs, dtype=np.int32)









                #     object_id = ""

                #     if is_unknown_object(class_id, probability) and height < 500:
                #         #print("1frame")

                #         brack_img = self.seg_mask.copy()
                #         # brack_img[y:y + height, x:x + width] = 255
                #         # mask_img:np.ndarray = brack_img[y:y + height, x:x + width]

                #         mask_img = cv2.fillPoly(brack_img, [m_points], 255)  # 255は白の値

                #         # BBOX(左上端座標, 幅, 高さ)
                #         bounding_box = BoundingBox(x, y, width, height)
                #         area = width*height # BBOXの面積

                #         object_id = self.bbox_count

                #         test_item = [x,y, width, height]



                #         bbox_item = BboxObject(bounding_box, area, mask_img, timestamp,class_id,object_id)
                #         bbox_item_list.append(bbox_item)


                #         bbox_item2 = [x,y, width, height, probability,class_id]
                #         mot_item_list.append(bbox_item2)

                #         self.match[object_id] = str(object_id)



                #         #Track用box辞書
                #         self.curent_object_dict[object_id] = bbox_item
                #         self.bbox_count += 1

                #         #print(object_id)


                #         #test_item = [x,y,xmax,ymax]



                feature_list = []
                track = np.zeros_like(img)
                box_flow = []
                box_flow2 = []
                points_in_mask = []
                fe_ide = []

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
                        distance = 20  # 削る距離（ピクセル単位）
                        kernel = np.ones((distance, distance), np.uint8)  # 距離に基づくカーネル


                        brack_img= cv2.erode(brack_img, kernel, iterations=1)




                        # if is_point_inside_bounding_box(new_point, reactangle):

                        if brack_img[int(b), int(a)] == 255:
                            

                            txt_bk_color = get_id_color(int(key))
                            # m_points = np.array(mask_pairs, dtype=np.int32)
                                # cv2.line(track, (int(a),int(b)),(int(c),int(d)), txt_bk_color, 2)
                            # result = cv2.circle(result,(int(a),int(b)),5,txt_bk_color,-1)
                            # contour = m_points.reshape((-1, 1, 2)).astype(np.int32)


                            # distance = abs(cv2.pointPolygonTest(contour, (a,b), True))

                            # if distance > 7:
                            result = cv2.circle(result,(int(a),int(b)),5,txt_bk_color,-1)

                            track_id.append(key)
                                # if len(box_flow) < 20:
                                # box_flow.append(new_point)

                            # box_flow.append(new_point)
                            # if key == 1 :
                            #     if len(box_flow) < 20:
                            #         box_flow.append(new_point)
                            #     # box_flow.append(new_point)
                            # elif key == 2 :
                            #     if len(box_flow) < 20:
                            #         box_flow.append(new_point)
                                # box_flow2.append(new_point)
                            









                    if len(track_id) > 0 :
                        # cv2.line(track, (int(a),int(b)),(int(c),int(d)), colar, 2)
                        # result = cv2.circle(result,(int(a),int(b)),5,colar,-1)
                        feature_list.append([new_point,old_point,track_id])
                    else:
                        fe_ide.append(i)


                # print("feature_list")
                # print(feature_list)

                # print(self.curent_object_dict)
                # for mot_item in mot_item_list:
                #     mot_item.insert(5, box_flow)
                #     # mot_item.append(box_flow)



                feature = {"feature": feature_list}

                bbox = {"bbox": self.curent_object_dict}

                self.feature_box_dict.setdefault(self.frame_count, [feature])
                

                self.feature_box_dict[self.frame_count].append(bbox)

                self.inactive_counter =[0] * len(self.feature_box_dict[self.frame_count][0]["feature"])  # 各特徴点のカウンターをリストで初期化

                
                for values in self.feature_box_dict[self.frame_count][0]["feature"]:

                    new_point, old_point, trackid = values

                    

                    #print(trackid)
                    a,b = new_point
                    c,d = old_point
                    

                    box_flow = []

                    if len(trackid) > 0:

                        if trackid[0] in self.fe_idelist:
                            self.fe_idelist[trackid[0]].append([new_point])
                        else:
                            self.fe_idelist.setdefault(trackid[0], [[new_point]])

                
                for bo_id, mot_item in enumerate(mot_item_list):
                    box_flow = self.fe_idelist[int(bo_id)]
                    mot_item.insert(5, box_flow)



                # mot
                # print(mot_item_list)


                motdetections = mot.bboxes2out_detections(mot_item_list)
                #print(motdetections)
                self.mottracker.step(motdetections, self.match)
                mottracks = self.mottracker.active_tracks(min_steps_alive=1)
                #print(len(mottracks))

                for track_result in mottracks:
                    if track_result.id not in self.track_id_dict:
                        new_id = len(self.track_id_dict)
                        self.track_id_dict[track_result.id]= new_id
                result = mot.draw_debug(result,mottracks,self.track_id_dict)
                # print(fe_ide)
                # print(self.good_new.shape[0])

                for del_fe in sorted(fe_ide,reverse=True):
                    self.good_new= np.delete(self.good_new, del_fe, axis=0)
                    self.good_old= np.delete(self.good_old, del_fe, axis=0)








                print(self.feature_box_dict)

            else:
                #print(len(p1))
                print("開始")
                # print(img.shape)



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




                if len(self.good_new) != len(self.feature_box_dict[self.frame_count -1][0]["feature"]):
                    print("特徴の数が違う")
                    print(len(self.good_new))
                    #print(len(self.good_old))
                    print(len(self.feature_box_dict[self.frame_count -1][0]["feature"]))

                #print(self.good_new)
               # print(self.feature_box_dict[self.frame_count -1][0]["feature"])

                print("naiii")
                # print(p1[st==0])

                # if len(p1[st==0]) >0 :
                #     l_colar =(0,0,255)
                    # for lost_p in p1[st==0]:
                    #     l_a,l_b = lost_p.ravel()
                        # result = cv2.circle(result,(int(l_a),int(l_b)),5,l_colar,3)






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

                #print(self.feature_box_dict[2])


                yolox_bboxes = yolox_bbox_src.bounding_boxes #yolox-rosから受け取った物体集合から物体一つずつ取り出す
                yolox_bboxes = filter_boxes(yolox_bboxes)

                self.ID_point_count_dict = {}
                track = np.zeros_like(img)
                #print(yolox_bboxes)

                lost_box = {}


                yolo_segments = segment_src.segments
                yolo_segments = filter_boxes(yolo_segments)

            

                # for id, bbox in enumerate(yolo_segments):
                #     probability = bbox.probability
                #     x = bbox.xmin
                #     y = bbox.ymin
                #     xmax = bbox.xmax
                #     ymax = bbox.ymax
                #     height = ymax - y
                #     width = xmax - x
                #     class_id = bbox.class_id
                #     x_masks = bbox.x_masks
                #     y_masks = bbox.y_masks

                #     mask_pairs = list(zip(y_masks,x_masks))
                    
                #     if is_unknown_object(class_id, probability) and height < 500:


                #         mask_image = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                #         # for x, y in mask_pairs:
                #         #     mask_image[x, y] = 255  # 白いピクセルに設定

                #         points = np.array(mask_pairs, dtype=np.int32)

                #         # 輪郭を描いてその範囲を白塗り
                #         cv2.fillPoly(mask_image, [points], 255)  # 255は白の値
                #         # 輪郭線を描画
                #         cv2.polylines(mask_image, [points], isClosed=True, color=255, thickness=1)
                        
                #         # contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                #         # cv2.drawContours(mask_image, contours, -1, 255, thickness=cv2.FILLED)
                #         # cv2.namedWindow('Mask Image', cv2.WINDOW_NORMAL)

                #         window_name = f"Mask Image {id + 1}"

                #         cv2.imshow(window_name, mask_image)










                ##IOUを使ったlost検出　却下
                # #TrackingされているBox情報の補完
                # for key, p_item in self.curent_object_dict.items() :
                #     #print( p_item)

                #     detected = False
                #     bounding_box_src = p_item._bounding_box


                #     x, y, width, height = bounding_box_src.items
                #     xmax = width + x
                #     ymax = height + y
                #     p_point =[x,y, xmax, ymax]


                #     p_reactangle = [x,y,width, height]

                #     for bbox in (yolox_bboxes):
                #         probability = bbox.probability
                #         x = bbox.xmin
                #         y = bbox.ymin
                #         xmax = bbox.xmax
                #         ymax = bbox.ymax
                #         height = ymax - y
                #     # print(height)
                #         class_id = bbox.class_id
                #         object_id = ""
                #         width = xmax - x
                #         d_point = [x,y, xmax, ymax]

                #         if is_unknown_object(class_id, probability) and height < 700 :



                #             if iou(p_point,d_point ):
                #                 detected = True

                #                 break

                #     if not detected :
                #         lost_box[key] = p_reactangle


                print(color_img.shape[:2])
                mask_img = np.zeros(img.shape[:2])


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

                    find_box = []
                    






                    if is_unknown_object(class_id, probability) and height < 500 :
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

                        distance = 20  # 削る距離（ピクセル単位）
                        kernel = np.ones((distance, distance), np.uint8)  # 距離に基づくカーネル


                        brack_img= cv2.erode(brack_img, kernel, iterations=1)
                        
                        cv2.fillPoly(self.brack_img, [m_points], 255)



                        # cv2.imshow("demo",self.brack_img)


                        for values in self.feature_box_dict[self.frame_count][0]["feature"] :
                            #print(values)

                            new_point, old_point, trackid = values

                            #print(trackid)
                            a,b = new_point
                            c,d = old_point
                           

                            contour = m_points.reshape((-1, 1, 2)).astype(np.int32)



                            if 0 <= a < color_img.shape[1] and 0 <= b < color_img.shape[0]:  # 範囲チェック

                                if brack_img[int(b), int(a)] == 255:
                                    distance = abs(cv2.pointPolygonTest(contour, (a,b), True))

                                    # if distance > 7:

                                
                                        # box_flow.append(new_point)
                                        # if len(box_flow) < 20:
                                            # box_flow.append(new_point)
                                        

                                        # #print(trackid[0])
                                        # if len(trackid) == 1 and trackid[0] != None and trackid[0] != "None":
                                        # print(trackid[0])
                                    txt_bk_color = get_id_color(int(trackid[0]))
                                            # txt_bk_color = (self._COLORS[int(trackid[0])] * 255 * 0.7).astype(np.uint8).tolist()
                                            # cv2.line(track, (int(a),int(b)),(int(c),int(d)), txt_bk_color, 2)
                                    result = cv2.circle(result,(int(a),int(b)),5,txt_bk_color,-1)
                                        # elif len(trackid) > 1:

                                        #     txt_bk_color = get_id_color(int(trackid[0]))

                                        #     # txt_bk_color = (self._COLORS[int(trackid[0])] * 255 * 0.7).astype(np.uint8).tolist()

                                        #     cv2.line(track, (int(a),int(b)),(int(c),int(d)), txt_bk_color, 2)
                                        #     result = cv2.circle(result,(int(a),int(b)),5,txt_bk_color,-1)
                                        # else:
                                        #     # txt_bk_color = (self._COLORS[trackid[0]] * 255 * 0.7).astype(np.uint8).tolist()

                                        #     cv2.line(track, (int(a),int(b)),(int(c),int(d)), colar, 2)
                                        #     result = cv2.circle(result,(int(a),int(b)),5,colar,-1)


                                        #print(trackid)
                                        # print(len(points_in_mask))

                                    for id in trackid:
                                        if id in  total_counts:
                                            total_counts[id] += 1
                                            #print(id)

                                        else:

                                            total_counts.setdefault(id, 0)
                                            total_counts[id] += 1
                                

                                # elif len(trackid) > 0:
                                #     # box_flow.append("None")

                                #     # print(trackid)
                                #     txt_bk_color = get_id_color(int(trackid[0]))
                                #     # cv2.line(track, (int(a),int(b)),(int(c),int(d)), txt_bk_color, 2)
                                #     result = cv2.circle(result,(int(a),int(b)),5,txt_bk_color,2)















                            # elif len(trackid) > 0:
                                # print("不安定")
                                # print(new_point)
                                # # print(reactangle)
                                # if len(box_flow) < 5:

                                #     box_flow.append(None)

                        # for _ in range(20-len(box_flow)) :
                        #     box_flow.append(None)





                            # for key, p_reactangle in self.curent_object_dict.items() :
                            #     if not iou(p_reactangle, reactangle):
                            #         if is_point_inside_bounding_box(new_point, reactangle):


                        # print("各detectboxごとのTack_featureの数")
                        print(total_counts)
                        if  total_counts == {}:
                            point_relia = True
                        find_box.append(total_counts)
                        self.ID_point_count_dict.setdefault(num, []).append(total_counts)



                        #ID_point_count_dict.update(num, point_counts)
                        #print(ID_point_count_dict)
                        bounding_box = BoundingBox(x, y, width, height) # BBOX(左上端座標, 幅, 高さ)


                        bbox_item = BboxObject(bounding_box, area, brack_img, timestamp,class_id,object_id)

                        bbox_item2 = [x,y, width, height, probability,class_id]

                        mot_item_list.append(bbox_item2)

                        bbox_item_list.append(bbox_item)

                # brack_img = np.zeros(img.shape[:2])
                print("aho")
                print(len(self.feature_box_dict[self.frame_count][0]["feature"]))
                print(len(self.good_new))

                # 削除対象のインデックスを記録するリスト
                indices_to_remove = []



                fe_ide = []
                
             

                    
                print("各detectboxごとのTack_featureの数")
                print(self.ID_point_count_dict)
                print(mot_item_list)

                # print(len(self.fe_idelist[0]))
                # print(len(self.fe_idelist[1]))
                # cv2.imshow("demo2",self.brack_img)
                # kernel = np.ones((10, 10), np.uint8)  # 5x5の構造要素（調整可能）
                # dilated_mask = cv2.erode(self.brack_img, kernel, iterations=1)  # 1回膨張
                # cv2.imshow("demo",self.brack_img)


                # 空の辞書を持つキーを取り出す
                empty_keys = [key for key, value in self.ID_point_count_dict.items() if value == [{}]]



                self.match = {}
                # # matting
                # for outer_key, inner_list in self.ID_point_count_dict.items():
                #     max_key = None
                #     max_value = float('-inf')

                #     for inner_dict in inner_list:
                #         for inner_key, inner_value in inner_dict.items():
                #             if inner_value > max_value:
                #                 max_value = inner_value
                #                 max_key = inner_key

                #     match[outer_key] = max_key

                print("結果")
                self.new_key = empty_keys






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
                            if (totals[inner_key] < 20 and  max(inner_dict.values()) == inner_value) :
                                self.new_key.append(outer_key)
                                point_relia = True

                            if totals.get(inner_key, 0) < 1:
                                continue


                            ratio = inner_value / totals[inner_key]
        # print(totals[inner_key])
                                                # print(inner_value )


                            potential_assignments[outer_key].append((ratio, inner_key, inner_value))

                # # Step 3: Select the best assignments based on ratios

                # #　or表示の条件に合計の50％以下かどうか
                # outer_totals = {}
                # for outer_key2, inner_list2 in self.ID_point_count_dict.items():
                #     outer_totals[outer_key2] = sum(inner_value for inner_dict in inner_list2 for inner_value in inner_dict.values())

                # # Step 2: Collect all potential assignments with their ratios


                # for outer_key2, inner_list2 in self.ID_point_count_dict.items():

                #     for inner_dict2 in inner_list2:
                #         loop =0
                #         for inner_key2, inner_value2 in inner_dict2.items():
                #             ratio2 = inner_value2 / outer_totals[outer_key2]
                #             if len(potential_assignments[outer_key2]) > loop:
                #                 potential_assignments[outer_key2][loop]= potential_assignments[outer_key2][loop]+(ratio2,)
                #                 loop += 1
                #             else :
                #                 continue


                # Step 3: Select the best assignments based on ratios
                assignments = {}
                print(potential_assignments.items())

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
                # def find_best_assignment(values):
                #     # Check if there are multiple candidates with ratio > 0.5
                #     #print(values)
                #     candidates = [inner_key for ratio, inner_key, inner_value,ratio2 in values if ratio >= 0.8 and ratio2 > 0.5]
                #     if len(candidates) > 1:
                #         return " or ".join(map(str, candidates))

                #     # Return the best available candidate
                #     for ratio, inner_key, inner_value , ratio2 in values:

                #         return str(inner_key)
                #     return None
                # Function to find the best assignment
                def find_best_assignment(values):
                    print(values)
                    ratio, inner_key, inner_value, ratio2 = values
                    candidates = [inner_key]

                    # candidates = [inner_key for ratio, inner_key, inner_value, ratio2 in values ]#if ratio >= 0.1 and ratio2 >= 0.1
                    print(candidates)

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

                # for outer_key in potential_assignments:
                #     values = potential_assignments[outer_key]
                #     # Sort by ratio descending
                #     values.sort(key=lambda x: x[0], reverse=True)

                #     # Assign the best available candidate
                #     assigned_inner_key = find_best_assignment(values)

                #     if assigned_inner_key is not None:
                #         self.match[outer_key] = assigned_inner_key
                print("できそう")
                print(assignments.items())
                # assignments = {
                #     key: max(values, key=lambda x: x[2])  # 一番目の値で最大を取得
                #     for key, values in assignments.items()
                # }

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

                # final_assignments = {
                #     key: max(values, key=lambda x: x[2])
                #     for key, values in final_assignments.items()
                # }




                print(final_assignments.items())

                for outer_key, values in final_assignments.items():
                    assigned_inner_key = find_best_assignment(values)
                    print( assigned_inner_key)
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


                print(self.match)

                # self.match = dict(sorted(self.match.items(), key=lambda item: item[1]))


                print(self.match)

                #mot_item_list をId順に並べる　

                print("数")

                print(len(self.feature_box_dict[self.frame_count][0]["feature"]))
                mean = []

                distance = 20  # 削る距離（ピクセル単位）
                kernel = np.ones((distance, distance), np.uint8)  # 距離に基づくカーネル


                self.brack_img= cv2.erode(self.brack_img, kernel, iterations=1)

                


                #self.fe_idelist
                for idx,values in enumerate(self.feature_box_dict[self.frame_count][0]["feature"]):
                #print(values)

                    new_point, old_point, trackid = values

                    

                    #print(trackid)
                    a,b = new_point
                    c,d = old_point
                    

                    # if is_point_inside_bounding_box(new_point, reactangle):
                        # print("x")
                        # print(x)

                    # m_points = np.array(mask_pairs, dtype=np.int32)


                    # # 輪郭を描いてその範囲を白塗り
                    # cv2.fillPoly(brack_img, [m_points], 255)  # 255は白の値

                    # mask_img = brack_img

                    box_flow2 = []

                    if 0 <= int(a) < self.brack_img.shape[1] and 0 <= int(b) < self.brack_img.shape[0]:  # 範囲チェック
                    

                        if self.brack_img[int(b), int(a)] == 255:
                            
                            # distances = abs(cv2.pointPolygonTest(contour, (a,b), True))
                            

                            if len(trackid) > 0:

                                if trackid[0] in self.fe_idelist:
                                    self.fe_idelist[trackid[0]].append([new_point])
                                else:
                                    self.fe_idelist.setdefault(trackid[0], [[new_point]])




                            #print(trackid[0])
                            if len(trackid) == 1 and trackid[0] != None and trackid[0] != "None":
                            # print(trackid[0])
                                txt_bk_color = get_id_color(int(trackid[0]))
                                
                                # cv2.line(track, (int(a),int(b)),(int(c),int(d)), txt_bk_color, 2)
                                # result = cv2.circle(result,(int(a),int(b)),5,txt_bk_color,-1)
                                
                            elif len(trackid) > 1:

                                txt_bk_color = get_id_color(int(trackid[0]))




                                # txt_bk_color = (self._COLORS[int(trackid[0])] * 255 * 0.7).astype(np.uint8).tolist()

                                # cv2.line(track, (int(a),int(b)),(int(c),int(d)), txt_bk_color, 2)
                                # result = cv2.circle(result,(int(a),int(b)),5,txt_bk_color,-1)


                            if len(self.inactive_counter) > idx:
                                self.inactive_counter.append(0)
                            else:
                                self.inactive_counter[idx] = 0 

                            # print(self.feature_box_dict[self.frame_count][0]["feature"][idx])
                    


                        elif len(trackid) > 0:
      
                            self.inactive_counter[idx] += 1  

      
                            # # 20フレーム連続してマスク外にいる場合は辞書から削除
                            if self.inactive_counter[idx] >= 20  :
                                self.inactive_counter[idx] =0
                   
                                if trackid[0] in self.fe_idelist:
                                    self.fe_idelist[trackid[0]].append("None")
                                else:
                                    self.fe_idelist.setdefault(trackid[0], ["None"])
                              

                                # if not find_c: 
                                mean.append(trackid[0])
                                
                                indices_to_remove.append(idx)
                            else:
                                if trackid[0] in self.fe_idelist:
                                    self.fe_idelist[trackid[0]].append([new_point])
                                else:
                                    self.fe_idelist.setdefault(trackid[0], [[new_point]])


                            txt_bk_color = get_id_color(int(trackid[0]))
                            # result = cv2.circle(result,(int(a),int(b)),5,txt_bk_color,2)

                        
                        else: 
                            print("iru")

                            

                            result = cv2.circle(result,(int(a),int(b)),5,colar,2)
                            
                            self.inactive_counter[idx] += 1  

                            # 20フレーム連続してマスク外にいる場合は辞書から削除
                            if self.inactive_counter[idx] >= 5:
                                self.inactive_counter[idx] = 0 
                                # 辞書から特徴点を削除
                                # print(self.good_new)
                                # self.good_new= np.delete(self.good_new, idx, axis=0)
                                # del self.feature_box_dict[self.frame_count][0]["feature"][idx]  # 現在のインデックスの特徴点を削除
                                # self.inactive_counter.pop(idx)  # カウンターも削除
                                # # 削除後、インデックスを調整する必要があるため、ループを続ける
                                # indices_to_remove.append(idx)
                    
                    elif len(trackid) > 0:
                        # print("nnnnnnnnnnnnnr")
                        # print(new_point)
                        find_c = False

                        # for find_id in self.match.values():
                        #     if find_id != 'None':
                        #         if int(trackid[0]) == int(find_id):
                        #             find_c = True

                        self.inactive_counter[idx] += 1 
                        
                        # if not find_c:

                        #     mean.append(trackid[0])


                        if self.inactive_counter[idx] >= 20:
                            self.inactive_counter[idx] = 0

                            if trackid[0] in self.fe_idelist:
                                self.fe_idelist[trackid[0]].append("None")
                            else:
                                self.fe_idelist.setdefault(trackid[0], ["None"])

                            # if not find_c:

                            mean.append(trackid[0])

                            print(trackid[0])



                            indices_to_remove.append(idx)
                        else:
                            if trackid[0] in self.fe_idelist:
                                self.fe_idelist[trackid[0]].append([new_point])
                            else:
                                self.fe_idelist.setdefault(trackid[0], [[new_point]])

                            

                        # else:
                        #     if trackid[0] in self.fe_idelist:
                        #         self.fe_idelist[trackid[0]].append([new_point])
                        #     else:
                        #         self.fe_idelist.setdefault(trackid[0], [[new_point]])



                
                # mean = list(set(mean))
                # test =[]
                # for g in mean :
                #     test.append(self.fe_idelist[g])
                
                # mean = [x for x in mean if str(x) not in self.match.values()]

                # print(mean)
                # print(test)


                # if mean !=[]:
                #     self.mottracker.delete_index(mean,test)


                

                


                # print(indices_to_remove)
                indices_to_remove = list(set(indices_to_remove))
                indices_to_remove = sorted(indices_to_remove, reverse=True)
                print(indices_to_remove)
                for idx in indices_to_remove:
                    del self.feature_box_dict[self.frame_count][0]["feature"][idx]  # 特徴点を削除
                    del self.inactive_counter[idx]  # カウンターも削除
                    self.good_new= np.delete(self.good_new, idx, axis=0)
                    self.good_old= np.delete(self.good_old, idx, axis=0)

                print("aho2")
                print(len(self.feature_box_dict[self.frame_count][0]["feature"]))
                print(len(self.good_new))

                for mot_item, bo_id  in zip(mot_item_list,self.match.values()):
                    if bo_id != 'None':
                        box_flow = self.fe_idelist[int(bo_id)]
                        mot_item.insert(5, box_flow)
                    else:
                        box_flow = []
                        mot_item.insert(5, box_flow)







                # #mot
                motdetections = mot.bboxes2out_detections(mot_item_list)
                # print("box_view")
                # print(motdetections)
                d_trac = self.mottracker.step(motdetections, self.match)
                mottracks = self.mottracker.active_tracks(min_steps_alive=3)
                print("count_box")
                # print((d_trac))

                for track_result in mottracks:
                    self.new_id = len(self.track_id_dict)
                    if track_result.id not in self.track_id_dict:

                        self.track_id_dict[track_result.id]= self.new_id

                    tracker_id = self.track_id_dict[track_result.id]
                    # print(tracker_id)
                
                result = mot.draw_debug(result,mottracks,self.track_id_dict)




                #self.oneflag = True
                #print(self.feature_box_dict[self.frame_count][0]["feature"] )



                for id_num in self.id :
                    if "or" in str(id_num) :
                        print("or_flag")
                        uncertain_flag = True
                    # else:

                    #     print("nobo")


                feature_list = []

                self.id_list =[]

                lost_points = []

                # print(len(self.feature_box_dict[self.frame_count][0]["feature"]))
                # print(len(self.good_new.reshape(-1, 1, 2)))



                # 新しい特徴店の登録　修正部分
                for point_num , values in enumerate(self.feature_box_dict[self.frame_count][0]["feature"]) :

                    new_point, old_point, trackid = values
                    #print(values)
                    #print(trackid)
                    lost_feature_flag = True

            
        # print(self.good_new.reshape(-1, 1, 2)[point_num])
                    # print(values)
                    


                    for num, w_item in enumerate(yolo_segments): #mottrackstrack_result

                    #     tracker_id = self.track_id_dict[track_result.id]
                    #     # print("asssssssssssssssssss")
                    #     # print(tracker_id)
                    #     if not tracker_id in self.id_list:
                    #         self.id_list.append(tracker_id)
                    #     bbox = track_result.box
                    #     x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    #     reactangle = [x1, y1,x2-x1,y2-y1]


                    #     if is_point_inside_bounding_box(new_point, reactangle):
                    #         a,b = new_point
                    #         c,d = old_point
                    #         if len(trackid) > 0:

                    #             if "or" in str(tracker_id) :
                    #                     continue
                    #             elif not int(tracker_id) in trackid :
                    #                 # object_id = self.match[object_id]
                    #                 trackid = []

                    #                 trackid.append(int(tracker_id))



                    #         else:
                    #             # if "lost" in str(self.match[object_id]) :
                    #             #     object_id = object_id[-1]
                    #             #     track_id.append(self.match[object_id])
                    #             #     txt_bk_color = (self._COLORS[self.match[object_id]] * 255 * 0.7).astype(np.uint8).tolist()
                    #             #     cv2.line(track, (int(a),int(b)),(int(c),int(d)), txt_bk_color, 2)
                    #             #     result = cv2.circle(result,(int(a),int(b)),5,txt_bk_color,-1)

                    #             if "or" in str(tracker_id) :
                    #                 txt_bk_color = colar
                    #             else:
                    #                 txt_bk_color = get_id_color(int(tracker_id))
                    #                 # txt_bk_color = (self._COLORS[int(object_id)] * 255 * 0.7).astype(np.uint8).tolist()
                    #                 cv2.line(track, (int(a),int(b)),(int(c),int(d)),txt_bk_color, 2)
                    #                 result = cv2.circle(result,(int(a),int(b)),5,txt_bk_color,-1)
                    #                 #print(object_id)

                    #                 trackid.append(int(tracker_id))



                        # print(self.id_list)



                        # 特徴点登録　yolo
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

                        reactangle = [x,y, width, height]

                        count = 0

                        if is_unknown_object(class_id, probability) and height < 500 :
                        # print("ss" )

                            #result = cv2.rectangle(result, (x, y), (x + width, y + height), (255,204,102), thickness=3)

                            if object_id == "" :
                         
                                #print(object_id )

                                # if not len(bbox_item_list) > len(self.id) :
                                #     #print("aa")

                                if num in self.match.keys() and self.match[num]!= 'None':       
                                    object_id = self.match[num]
                                    # print(object_id)

                            # print("id")
                            #print(object_id)
                            if object_id != None and object_id != "" :
                                if num in self.match:
                                    object_id = self.match[num]
                                    #print(object_id)

                            if is_point_inside_bounding_box(new_point, reactangle):
                                a,b = new_point
                                c,d = old_point
                                # mask_pairs = list(zip(y_masks,x_masks))


                                # brack_img =  self.seg_mask.copy()
                        
                            
                                # m_points = np.array(mask_pairs, dtype=np.int32)
                                # # print(m_points )


                                # # 輪郭を描いてその範囲を白塗り
                                # cv2.fillPoly(brack_img, [m_points], 255)  # 255は白の値
                                # #print(num)
                                # if brack_img[int(b), int(a)] == 255:
                                lost_feature_flag = False

                                
                            

                                if len(trackid) > 0:
                                    if self.match[num] == 'None':
                                        lost_points.append(point_num)
                                        # del self.feature_box_dict[self.frame_count][0]["feature"][point_num]

                                    elif int(trackid[0]) != int(self.match[num]) :
                                        

                                        
                                        # print(trackid[0])
                                        print('でなない')
                                        lost_points.append(point_num)
                                        lost_feature_flag = True

                                        
                                        break
                
                                        # del self.feature_box_dict[self.frame_count][0]["feature"][point_num]
                                        # self.good_new =np.delete(self.good_new,point_num, axis=0)
                                        # self.good_old =np.delete(self.good_old, point_num, axis=0)


                                    # print(self.match)
                                    # if "lost" in str(object_id) :
                                    #     object_id = object_id[-1]


                                    if "or" in str(object_id) :
                                        continue
                                    # elif not int(object_id) in trackid :
                                    #     # object_id = self.match[object_id]
                                    #     trackid = []

                                    #     trackid.append(int(object_id))


                                else:
                                    # if "lost" in str(self.match[object_id]) :
                                    #     object_id = object_id[-1]
                                    #     track_id.append(self.match[object_id])
                                    #     txt_bk_color = (self._COLORS[self.match[object_id]] * 255 * 0.7).astype(np.uint8).tolist()
                                    #     cv2.line(track, (int(a),int(b)),(int(c),int(d)), txt_bk_color, 2)
                                    #     result = cv2.circle(result,(int(a),int(b)),5,txt_bk_color,-1)

                                    if "or" in str(object_id) :
                                        txt_bk_color = colar
                                    elif object_id !='' and object_id !='None':
                                        txt_bk_color = get_id_color(int(object_id))
                                        # txt_bk_color = (self._COLORS[int(object_id)] * 255 * 0.7).astype(np.uint8).tolist()
                                        # # cv2.line(track, (int(a),int(b)),(int(c),int(d)),txt_bk_color, 2)
                                        # # result = cv2.circle(result,(int(a),int(b)),5,txt_bk_color,-1)
                                        #print(object_id)

                                        trackid.append(int(object_id))

                                        # uncertain_flag = False



                                    #txt_bk_color = (self._COLORS[trackid[0]] * 255 * 0.7).astype(np.uint8).tolist()
                                # else:
                                #     print("ghghh")
                                #     print(object_id)
                                #     print(self.match)
                                #     if "lost" in str(object_id) :
                                #         object_id = object_id[-1]
                                #     counta += 1
                                #     object_id =  len(self.curent_object_dict)+ counta

                                #     if "or" in str(object_id) :
                                #         txt_bk_color = colar
                                #     else:
                                #         txt_bk_color = (self._COLORS[int(object_id)] * 255 * 0.7).astype(np.uint8).tolist()
                                #         track_id.append(int(object_id))
                                #         print(track_id)

                                #         cv2.line(track, (int(a),int(b)),(int(c),int(d)), txt_bk_color, 2)
                                #         result = cv2.circle(result,(int(a),int(b)),5,txt_bk_color,-1)
                           
                                




                            
                    # print(len(self.feature_box_dict[self.frame_count][0]["feature"]))

                    # print(len(self.good_new))
                    # if  lost_feature_flag:
                    #     print(self.good_new.reshape(-1, 1, 2)[point_num])
                    #     m = self.good_new[point_num][0, 0]
                    #     n = self.good_new[point_num][0, 1]
                    #     print(m)
                    #     if int(self.feature_box_dict[self.frame_count][0]["feature"][point_num][0][0]) ==  int(m) and int(self.feature_box_dict[self.frame_count][0]["feature"][point_num][0][1]) ==  int(n):
                    #         del self.feature_box_dict[self.frame_count][0]["feature"][point_num]
                    #         self.good_new =  np.delete(self.good_new, point_num, axis=0)
                    #         self.good_old =np.delete(self.good_old ,point_num , axis=0)
                    

                    # print(len(self.feature_box_dict[self.frame_count][0]["feature"]))

                    # print(len(self.good_new))
                    



                    feature_list.append([new_point,old_point,trackid])



                   # self.oneflag = False


                    # print("feature_list")
                  #  print(feature_list)

                    # print(self.curent_object_dict)


                feature = {"feature": feature_list}
                #print(feature)

                # bbox = {"bbox": self.curent_object_dict}



                # print(self.feature_box_dict[self.frame_count][0]["feature"])
                ##これで更新
                # self.feature_box_dict[self.frame_count][0]["feature"] = feature_list
                # print(len(self.feature_box_dict[self.frame_count][0]["feature"]))

                # print(len(self.good_new))


                # print(len(self.feature_box_dict[self.frame_count][0]["feature"]))
                lost_points = list(set(lost_points))
                print("lost")
                lost_points =sorted(lost_points, reverse=True)
                print(lost_points)
                print(len(self.feature_box_dict[self.frame_count][0]["feature"]))

                # for point_lost in lost_points:
                #     if 0 <= point_lost < len(self.feature_box_dict[self.frame_count][0]["feature"]):
                            
                #         del self.feature_box_dict[self.frame_count][0]["feature"][point_lost]
                #         self.good_new =np.delete(self.good_new, point_lost,axis=0)
                #         self.good_old =np.delete(self.good_old, point_lost,axis=0)


                #print(self.feature_box_dict[self.frame_count][0]["feature"])
                # print(self.feature_box_dict)

                # for i in range(len(detections)):
                #     for j in range(i + 1, len(detections)):
                #         boxA = detections[i]['bbox']
                #         boxB = detections[j]['bbox']
                #         iou = calculate_iou(boxA, boxB)



                # self.good_new = p1[st==1]
                # self.good_old = p0[st==1]


                #lostしたboxの表示
                # for l_num, lost_item in lost_box.items():


                #     x, y, width, height = lost_item

                #     object_id = "lost"+str(l_num)

                #     l_reactangle = [x,y, width, height]

                #     if object_id in self.match and not l_num in self.match  :

                #         result = cv2.rectangle(result, (x, y), (x + width, y + height), colar, thickness=3)
                #         cv2.putText(result, f'ID : {object_id}', (x,y),cv2.FONT_HERSHEY_PLAIN, 1.5, colar, thickness=2)





                #print(ID_point_count_dict)
                    # for obj_id, (new, old) in self.trackable_objects.items():
                    #     new_point = self.good_new[obj_id]
                    #     new = np.vstack((new, new_point))
                    #     self.trackable_objects[obj_id] = (new, old)


                # for obj_id, (new, old) in self.trackable_objects.items():
                #     color = (0, 255, 0)  # オブジェクトごとに色を割り当てる
                #     for i in range(1, len(new)):
                #         a, b = new[i].ravel()
                #         c, d = old[i].ravel()
                #         result = cv2.line(result, (a, b), (c, d), color, 2)
                #         result = cv2.circle(result, (a, b), 5, color, -1)
                #         cv2.putText(result, f"ID: {obj_id}", (a, b), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)




                best_overlap_count = 0

                # # 物体追跡を描画
                # track = np.zeros_like(img)
                # for i,(new,old) in enumerate(zip(self.good_new,self.good_old)):
                #     total_overlap_count = 0

                #     a,b = new.ravel()
                #     point = a,b #+211
                #     c,d = old.ravel()
                #     point2 = c,d #+211
                #     colar =(0,255,0)

                #     cv2.line(track, (int(a),int(b)),(int(c),int(d)), colar, 2)
                #     result = cv2.circle(result,(int(a),int(b)),5,colar,-1)
                

                # # #mot
                # motdetections = mot.bboxes2out_detections(mot_item_list)
                # print("box_view")
                # print(motdetections)
                # d_trac = self.mottracker.step(motdetections, self.match)
                # mottracks = self.mottracker.active_tracks(min_steps_alive=3)
                # print("count_box")
                # # print((d_trac))

                # for track_result in mottracks:
                #     self.new_id = len(self.track_id_dict)
                #     if track_result.id not in self.track_id_dict:

                #         self.track_id_dict[track_result.id]= self.new_id

                #     tracker_id = self.track_id_dict[track_result.id]
                #     # print(tracker_id)
                
                # result = mot.draw_debug(result,mottracks,self.track_id_dict)






            self.track_list.append(track)
            if( len(self.track_list)> 10 ):
                self.track_list.pop(0)

            for t in self.track_list :
                result = np.where(t!=0,t,result)

            self.prvs = next


            # else:


            #     self.current_object_dict = {}
            #     if( len(self.track_list)> 10 ):
            #         self.track_list = []

        #result = cv2.resize(result,(width,height))
        self.print_fps(result)


            # cv2.putText(result,
            #             text="FPS:%f"%(1./duration),org=(10,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #             fontScale=1.0,color=(0,255,0),thickness=2,lineType=cv2.LINE_4)


        # wait bounding_boxの表示& 新しい特徴店の追加
        #print(len(wait_item_list))


        if proc == "lktracking":

            #不確かなboxがなく、数が少ないなら新しく特徴点の追加を行う
        # print(len(self.feature_box_dict[self.frame_count][0]["feature"]))
            feature_list2 =[]
            new_addpoint = []
            none_dict = {key: value for key, value in self.match.items() if value == 'None'}
            flg_m = False
            # self.brack_img = self.brack_img.astype(np.uint8)
            # mask = self.brack_img

            
        

            if not uncertain_flag and point_relia:
                counta = 0
                p_new = cv2.goodFeaturesToTrack(next, mask =None, **feature_params)
                print("特徴点追加")
                # print(self.new_key)

                for add_point in p_new :
                    track_id = []

                    a,b = add_point.ravel()
                    new_point = a,b
                    old_point = a,b

                    # for num, track_result  in enumerate(mottracks):
                    #     class_id =  self.track_id_dict[track_result.id]
                    #     bbox = track_result.box
                    #     x, y, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    #     height = ymax - y
                    #     width = xmax - x

                    #     reactangle = [x,y, width, height]


                    #     if is_point_inside_bounding_box(new_point, reactangle)  :

                    #         track_id.append(int(class_id))



                    #         txt_bk_color = get_id_color(int(class_id))
                    #         # txt_bk_color = (self._COLORS[int(object_id)] * 255 * 0.7).astype(np.uint8).tolist()

                    #         # result = cv2.circle(result,(int(a),int(b)),5,txt_bk_color,-1)
                    #         new_addpoint.append([new_point])

                            

                    #         feature_list2.append([new_point,old_point,track_id])

                                





                    # # # # yolo
                    loop = 0
                    for num, w_item  in enumerate(yolo_segments):
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

                        total_counts = {}
                        x_masks = w_item.x_masks
                        y_masks = w_item.y_masks

                        reactangle = [x,y, width, height]
                        reactangle3 = [x,y, xmax, ymax]

                        if is_unknown_object(class_id, probability) and height < 700 and num in self.new_key :
                            # print("何だ")
                            if object_id == "" :

                                if num in self.match.keys() and self.match[num]!= 'None':
                                    object_id = self.match[num]
                                else:
                                    for track_result in mottracks:
                                        tracker_id = self.track_id_dict[track_result.id]
                                        bbox = track_result.box
                                        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                                        reactangle2 = [x1, y1,x2,y2]

                                        if iou(reactangle3, reactangle2):
                                            object_id = tracker_id
                                            flg_m = True
                                            break

                                    if not flg_m :

                                        index = list(none_dict.keys()).index(num) if num in none_dict else 0
                                        self.new_id +=1
                                        object_id = self.new_id
                                
                        






                        #print(object_id )
                                loop += 1


                                # if num in self.match:
                                    # object_id = self.match[num]
                                    #print(object_id)
                                # print("aaa")
                                # if brack_img[int(b), int(a)] == 255:

                                if is_point_inside_bounding_box(new_point, reactangle):
                                    mask_pairs = list(zip(y_masks,x_masks))


                                    brack_img =  self.seg_mask.copy()
                            
                                
                                    m_points = np.array(mask_pairs, dtype=np.int32)
                                    # print(m_points )


                                    # 輪郭を描いてその範囲を白塗り
                                    cv2.fillPoly(brack_img, [m_points], 255)  # 255は白の値

                                    if brack_img[int(b), int(a)] == 255:
                                        
                                    

                                        # print("sdsds")
                                        if object_id == '':
                                            self.new_id +=1
                                            object_id = self.new_id
                                            

                                        track_id.append(int(object_id))

                                        txt_bk_color = get_id_color(int(object_id))
                                        # txt_bk_color = (self._COLORS[int(object_id)] * 255 * 0.7).astype(np.uint8).tolist()

                                        # result = cv2.circle(result,(int(a),int(b)),5,txt_bk_color,-1)
                                        new_addpoint.append([new_point])



                                        feature_list2.append([new_point,old_point,track_id])

                if new_addpoint is not None:
                    # print(new_addpoint)
                    # print("ddddddddddddddd")
                    # print(len(self.good_new.reshape(-1, 1, 2)))
                    if new_addpoint != []:
                        # self.good_new = np.array(new_addpoint)
                        print(new_addpoint)
                        # if self.good_new.size % 2 != 0:
                        #     self.good_new = self.good_new[:self.good_new.size - 1]

                        self.good_new = np.concatenate((self.good_new.reshape(-1, 1, 2), new_addpoint), axis=0)
                        self.feature_box_dict[self.frame_count][0]["feature"].extend(feature_list2)
                        # self.feature_box_dict[self.frame_count][0]["feature"] = feature_list2

                
            # if self.good_new.size % 2 != 0:
            #     self.good_new = self.good_new[:self.good_new.size - 1]
            print(len(self.good_new.reshape(-1, 1, 2)))
            print(len(self.feature_box_dict[self.frame_count][0]["feature"]))








            # counta = 0
            # for num, w_item  in enumerate(yolox_bboxes):
            #     probability = w_item.probability
            #     x = w_item.xmin
            #     y = w_item.ymin
            #     xmax = w_item.xmax
            #     ymax = w_item.ymax
            #     height = ymax - y
            #     # print(height)
            #     width = xmax - x
            #     class_id = w_item.class_id
            #     object_id = ""


            #     point_counts = {}
            #     total_counts = {}

            #     reactangle = [x,y, width, height]


            #     brack_img = np.zeros(color_img.shape[:2])
            #     brack_img[y:y + height, x:x + width] = 255
            #     mask_img:np.ndarray = brack_img[y:y + height, x:x + width]

            #     # BBOX(左上端座標, 幅, 高さ)
            #     bounding_box = BoundingBox(x, y, width, height)
            #     area = width*height # BBOXの面積


            #     count = 0

            #     if is_unknown_object(class_id, probability) and height < 500 :
            #         #print("ss" )
            #         bbox_item = BboxObject(bounding_box, area, mask_img, timestamp,class_id,object_id)

            #         #result = cv2.rectangle(result, (x, y), (x + width, y + height), (255,204,102), thickness=3)

            #         if object_id == "" :
            #             #print(object_id )

            #             # if not len(bbox_item_list) > len(self.id) :
            #             #     #print("aa")

            #             if num in self.match:
            #                 object_id = self.match[num]
            #                     #print(object_id)

            #         # print("id")
            #         #print(object_id)
            #         if object_id != None and object_id != "" :
            #             if num in self.match:
            #                 object_id = self.match[num]

            #             if "or" in object_id :
            #                 #  uncertain_flag = True
            #                  txt_bk_color = colar
            #             else:
            #                 #print(object_id)
            #                 txt_bk_color = (self._COLORS[int(object_id)] * 255 * 0.7).astype(np.uint8).tolist()
            #                 self.curent_object_dict[int(object_id)] = bbox_item

            #             #detectの表示　

            #             result = cv2.rectangle(result, (x, y), (x + width, y + height), txt_bk_color, thickness=3)
            #             cv2.putText(result, f'ID : {object_id}', (x,y),cv2.FONT_HERSHEY_PLAIN, 1.5, txt_bk_color, thickness=2)

            #             print(int(object_id))
            #             self.curent_object_dict[int(object_id)] = bbox_item

                #     else:
                #         #print(str(num))
                #         #print(self.match)
                #         if num in self.match:
                #             #print("a")
                #             object_id = self.match[num]
                #         # else:
                #         #     print("gbgg")
                #         #     print(len(self.curent_object_dict))
                #         #     counta += 1
                #         #     object_id =  len(self.curent_object_dict)+ counta


                #         txt_bk_color = (self._COLORS[int(object_id)] * 255 * 0.7).astype(np.uint8).tolist()
                #         result = cv2.rectangle(result, (x, y), (x + width, y + height), txt_bk_color, thickness=3)


                # # txt_bk_color = (self._COLORS[object_id] * 255 * 0.7).astype(np.uint8).tolist()
                #     cv2.putText(result, f'ID : {object_id}', (x,y),cv2.FONT_HERSHEY_PLAIN, 1.5, txt_bk_color, thickness=2)


                #     # #Track用box辞書
                    # print(int(object_id))
                    # self.curent_object_dict[int(object_id)] = bbox_item
            #修正中
            # for l_num, lost_item in self.curent_object_dict.items():
            #         # print(lost_item._bounding_box)


            #         bounding_box_src = lost_item._bounding_box
            #         x, y, width, height = bounding_box_src.items

            #         object_id = l_num

            #         l_reactangle = [x,y, width, height]
            #         lost_counts = {}

            #         if object_id != None and object_id != "" :

            #             if  l_num in self.tracked_objects:
            #                 #print(self.match)
            #                 if str(l_num) in self.match.values():

            #                     self.tracked_objects[l_num]['last_seen'] = self.frame_count
            #             else:
            #                 self.tracked_objects[l_num] = {'last_seen': self.frame_count}

            #         # ids_to_remove = []
            #         # for obj_id, info in self.tracked_objects.items():
            #         #     if self.frame_count - info['last_seen'] > self.max_missing_frames:
            #         #         #print(self.frame_count - info['last_seen'])
            #         #         ids_to_remove.append(obj_id)

            #         #     for obj_id in ids_to_remove:
            #         #         del self.tracked_objects[obj_id]

            #         #         del self.curent_object_dict[obj_id]



            #         #print(object_id)
            #         txt_bk_color = (self._COLORS[int(object_id)] * 255 * 0.7).astype(np.uint8).tolist()
            #         # print(x)
            #         # print(y)

            #         result = cv2.rectangle(result, (x, y), (x + width, y + height), txt_bk_color, thickness=3)
            #         cv2.putText(result, f'ID : {object_id}', (x,y),cv2.FONT_HERSHEY_PLAIN, 1.5, txt_bk_color, thickness=2)






            if len(self.curent_object_dict) !=0:
                bbox = {"bbox": self.curent_object_dict}
                self.feature_box_dict[self.frame_count].append(bbox)


            # print(self.feature_box_dict[1])

            # # 一番前の要素を取得
            # first_key = list(self.feature_box_dict.keys())[0]


            #５０フレーム記録したら、古いものから消す
            if len(self.feature_box_dict) > 50 :
                # 一番前の要素を取得
                first_key = list(self.feature_box_dict.keys())[0]
                self.feature_box_dict.pop(first_key,None)

            # print("辞書の数")
            # print(len(self.feature_box_dict))
            # print(len(self.curent_object_dict))


            # # メモリ使用量を取得
            # snapshot = tracemalloc.take_snapshot()
            # top_stats = snapshot.statistics('lineno')

            # print("[ Top 10 memory usage ]")
            # for stat in top_stats[:10]:
            #     print(stat)









        # for w_item in wait_item_list:
        #     bounding_box_src = w_item._bounding_box
        #     x, y, width, height = bounding_box_src.items
        #     result = cv2.rectangle(result, (x, y), (x + width, y + height), (255,204,102), thickness=3)

        #     # object_id_list = w_item._object_id.split('_')
        #     # cv2.putText(result, f'ID : {object_id_list[-1]}', (x,y),cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 128, 255), thickness=2)


        # for b_item in bring_in_list:
        #     bounding_box_src = b_item._bounding_box
        #     x, y, width, height = bounding_box_src.items
        #     b_item_id = b_item._object_id
        #     result = cv2.rectangle(result, (x, y), (x + width, y + height), (255,0,102), thickness=3)
        #     b_item_id_list = b_item_id.split('_')
        #     cv2.putText(result, f'ID : {b_item_id_list[-1]}', (x,y),cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 128, 255), thickness=2)

        # for frame_obj in self.frame_object_list:
        #     action_str = ''
        #     action = frame_obj._item._action
        #     if action == DetectedObjectActionEnum.TAKE_OUT:
        #         action_str = 'TAKE_OUT'
        #     else :
        #         action_str = ''

        #     x = frame_obj._item._bounding_box._x
        #     y = frame_obj._item._bounding_box._y

        #     cv2.putText(result, f'{action_str}', (x+10, y),cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 128, 255), thickness=2)


        cv2.namedWindow('OpenCV Capture', cv2.WINDOW_NORMAL)
        cv2.imshow("OpenCV Capture", result)

        # send image to NDI camera
        if ndi_send != None:
            ndi_img = cv2.cvtColor(result,cv2.COLOR_BGR2BGRA)
            ndi_frame.data = ndi_img
            ndi_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX
            ndi.send_send_video_v2(ndi_send, ndi_frame)

        if ndi_send != None:
            NDI_finish(ndi_send)
        #print(len(self.frame_object_list))

        if self.frame_object_list:
            detected_object_list = self.create_msg(self.frame_object_list, detected_object_list, frame)

        self.detection_publisher.publish(detected_object_list)


        # cap.release()
        #cv2.destroyAllWindows()
        cv2.waitKey(1)

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

    try:
        rclpy.spin(capture_node)
    except KeyboardInterrupt:
        pass

    finally:

        # 終了処理
        capture_node.destroy_node()
        rclpy.shutdown()

def increase_param():
    global gaussian_scale, median_scale, bilateral_scale
    gaussian_scale += 6
    if gaussian_scale > 149:
        gaussian_scale = 149
    median_scale += 6
    if median_scale > 199:
        median_scale = 199
    bilateral_scale += 5
    if bilateral_scale > 30:
        bilateral_scale = 30

def decrease_param():
    global gaussian_scale, median_scale, bilateral_scale
    gaussian_scale -= 6
    if gaussian_scale < 1:
        gaussian_scale = 1
    median_scale -= 6
    if median_scale < 1:
        median_scale = 1
    bilateral_scale -= 6
    if bilateral_scale < 1:
        bilateral_scale = 1

def reset_param():
    global gaussian_scale, median_scale, bilateral_scale
    gaussian_scale = 1


def NDI_setup():
    try:
        global ndi
        import NDIlib as ndi
    except ModuleNotFoundError as e:
        print(e)
        print("NDI module not found, so this function unavailable.")
        return False, None, None

    if not ndi.initialize():
        print("NDI module found but cannot be initialized.")
        return False, None, None
    else:
        print("NDI successfully initialized.")

    send_settings = ndi.SendCreate()
    send_settings.ndi_name = "ndi-python"
    ndi_send = ndi.send_create(send_settings)
    ndi_frame = ndi.VideoFrameV2()

    return True, ndi_send, ndi_frame


def NDI_finish(ndi_send):
    if ndi_send != None:
        ndi.send_destroy(ndi_send)
        ndi.destroy()
        return True
    else:
        return False

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
    return iou > 0.5

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


def is_unknown_object(class_id: str, probability: float, object_threshold=0.3) -> bool:
    """物体と思われるものの規定の物体でないものかどうか調べる関数
    Args:
        class_id (str): 物体のクラス名
        probability (float): 物体かどうかの確からしさ（max 1）
        object_threshold (float): 物体と判定するしきい値（max 1）
    Returns:
        bool: 物体と思われるものの規定の物体でないものかどうか
    """
    #DEFAULT_OBJECTS = ["banana",],"person"'chair''book''keyboard',

    DEFAULT_OBJECTS = ['refrigerator','dining table','chair','tv','microwave','laptop','potted plant','mouse']#["person",'staffed toy','chair',"bed","handbag","backpack","banana","remote","spoon",'dog','cat','laptop','tv','microwave','refrigerator','potted plant','cup','couch','mouse','sink','dining table','skateboard','bottle','cell phone','knife','bowl']
    is_object: bool = probability > object_threshold
    is_default_object = class_id in DEFAULT_OBJECTS




    #print(is_object and not(is_default_object))
    return is_object and not(is_default_object)


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
            if i != j and iou(reactangle, reactangle2)  :#and class_id == class_id2
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
            outputs = outputs[0].cpu().numpy()
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
        for bbox in bboxes:
            #print(bbox[5])
            # a,b = bbox[5][0]

            flow = bbox[5]

            class_id = bbox[6]

            print("class0")
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
#       #      bbox_np = bbox.cpu().numpy()
#       #      logger.info("bo")

#       #      conf_np = bbox.probability.cpu().numpy()
#       #      logger.info("co")

#        #     cls_np = bbox.class_id.cpu().numpy()
#       #      logger.info("cl")
#             width=bbox.xmax-bbox.xmin
#             height = bbox.ymax-bbox.ymin
#             #print(bbox.xmax-bbox.xmin)
#             #print(bbox.ymax-bbox.ymin)

#       #      detection_result = np.column_stack((boxes_np, confs_np, cls_np))
#             out_detections.append(Detection(box=[bbox.xmin, bbox.ymin, bbox.xmax,bbox.ymax],score=bbox.probability))
#             #out_detections.append([bbox.xmin.cpu().numpy(), bbox.ymin.cpu().numpy(), bbox.xmax.cpu().numpy(), bbox.ymax.cpu().numpy(),bbox.probability.cpu().numpy(),bbox.class_id.cpu().numpy()])
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
            #class_id = int(track_result.class_id)
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
