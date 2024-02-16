# coding: utf-8
from itertools import chain
import random
from itertools import chain
import random

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





from shigure_core_msgs.msg import DetectedObjectList, DetectedObject, TrackedObjectList, TrackedObject, BoundingBox,PoseKeyPointsList, Cube
from bboxes_ex_msgs.msg import BoundingBoxes

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
            '/rs/aligned_depth_to_color/cameraInfo', 
            qos_profile=shigure_qos
        )




        self.track_list = []
        self.lk_count :int = 0
        self.revs: np.ndarray = np.ndarray
        self.lk_reset = True
        self.time_synchronizer = message_filters.TimeSynchronizer(
            [yolox_bbox_subscriber,people_subscriber, color_subscriber, depth_camera_info_subscriber], 1000)	
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
        
        self._colors = []
        for i in range(255):
            self._colors.append(tuple([random.randint(128, 192) for _ in range(3)]))
        
        #tracking box
        self.current_object_dict = {}
        self.previous_object_dict = {}
        
        self._tracking_info = TrackingInfo()

        self.prvs_box = []
        self.ta_box: tuple[float,float,float,float]  = []

        self.bounding_box: tuple[float, float, float, float] = []



        self.object_index = 0


    def callback(self, yolox_bbox_src: BoundingBoxes,people: PoseKeyPointsList, color_img_src: CompressedImage, camera_info: CameraInfo):	
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
        
        timestamp = Timestamp(color_img_src.header.stamp.sec, color_img_src.header.stamp.nanosec)
        frame = ColorImageFrame(timestamp, self._color_img_buffer[0], color_img) #bufferの先頭の画像と新しい画像
        self._color_img_frames.add(frame) #ColorImageFrameslistの更新して、listに追加

        frame_object_dict,bring_in_list,wait_item_list,people_item_list,take_out_people_id,take_out_obj_class_id = self.yolox_object_detection_logic.execute(yolox_bbox_src, timestamp,people,color_img,self.frame_object_list,self._judge_params,self.take_out_people_id ,self.take_out_obj_class_id ,self.bring_in_list,self.wait_item_list)
        
        #if self._count == 0:
            #self.start_item_list = start_item_list
        self.bring_in_list = bring_in_list
        self.wait_item_list = wait_item_list
        self.people_item_list = people_item_list
        self.take_out_people_id = take_out_people_id
        self.take_out_obj_class_id = take_out_obj_class_id		
        #count = 1
        #self._count = count
        self.frame_object_list = list(chain.from_iterable(frame_object_dict.values())) #frame_object_dictをすべて取り出し
        
        
        sec, nano_sec = frame.timestamp.timestamp
        detected_object_list = DetectedObjectList()
        detected_object_list.header.stamp.sec = sec
        detected_object_list.header.stamp.nanosec = nano_sec
        detected_object_list.header.frame_id = camera_info.header.frame_id

        
        # if self.frame_object_list:		
        #     detected_object_list = self.create_msg(self.frame_object_list, detected_object_list, frame)

        # self.detection_publisher.publish(detected_object_list)
        #result = color_img.copy()




        self.get_logger().info('Buffering end', once=True)

        # オプティカルフローのコード


        # lkの特徴点の更新間隔

        if( len(sys.argv) == 1 ):
            cam = 0
        else :
            cam = int(sys.argv[1])
        
        if self.lk_count > 2:
            self.lk_reset = True
            self.lk_count =0

        elif self.lk_count == 0 :
            self.lk_reset = True

        else: 
            self.lk_reset = False
        

        #self.lk_count = self.lk_count + 1


        
            
        #print("camera device num: %d"%cam)

        # cap = cv2.VideoCapture(color_img)
        # cap.set(cv2.CAP_PROP_FPS,30)

        # print("FPS:%f"%cap.get(cv2.CAP_PROP_FPS))ttttttttttttttttttt
        # print("Image Height:%d"%int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # print("Image Width:%d"%int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        # print("FOURCC:%d"%int(cap.get(cv2.CAP_PROP_FOURCC)))

        proc = "lktracking"
        ndi_send = None

        # ret,img = cap.read()
        ret = True

        img = color_img
        self.flag = True
        self.tflag = True
        self.t2flag = True

        self.newflag = False
        self.id_flag = True

        # height = img.shape[0]
        # width = img.shape[1]
        # img = cv2.resize(img,((int)(width/2),(int)(height/2)))

        #self.prvs = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        #
        ## for HSV color space transform
        #
        if self._count == 0:
            self.hsv = np.zeros_like(img)
            self.hsv[...,1] = 255
            colors_hsv = [[0,0,0]]
            for i in range(1, 256):
                colors_hsv.append(np.array([random.randint(0,180), random.randint(120,255), 255]))
            colors_hsv = np.array(colors_hsv).astype(np.uint8)

        #
        ## for facedetection by OpenCV Haar-like feature based face detector
        #
        # haarcascade_frontalface_default.xml from https://github.com/opencv/opencv/tree/master/data/haarcascades
        face_cascade = cv2.CascadeClassifier('/home/azuma/ros2_ws/src/dir/data/haarcascades/haarcascade_frontalface_default.xml')
        # haarcascade_eye.xml from https://github.com/opencv/opencv/tree/master/data/haarcascades
        eye_cascade = cv2.CascadeClassifier('/home/azuma/ros2_ws/src/dir/data/haarcascades/haarcascade_eye.xml')
        eye_tree = cv2.CascadeClassifier('/home/azuma/ros2_ws/src/dir/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
        # haarcascade_upperbody.xml from https://github.com/opencv/opencv/tree/master/data/haarcascades
        body_cascade = cv2.CascadeClassifier('/home/azuma/ros2_ws/src/dir/data/haarcascades/haarcascade_upperbody.xml')


        profileface = cv2.CascadeClassifier('/home/azuma/ros2_ws/src/dir/data/haarcascades/haarcascade_profileface.xml')
        profile_face = True

        #
        ## for LK tracking
        #
        #ShiTomasiコーナー検出器のためのパラメータ
        lk_fnum = 1000
        feature_params = dict( maxCorners = lk_fnum,
                            qualityLevel = 0.001,
                            minDistance = 7,
                            blockSize = 7 )
        # Lucas-Kanade法によるオプティカル・フローのためのパラメータ
        lk_params = dict( winSize  = (15,15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
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


        elif proc == "lktracking" :
            roi_list = []
            next = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            result = img.copy()
            #cv2.rectangle(result, (0, 311), (900, 711), (255,204,102), thickness=3)
            
            
            if len(wait_item_list) >0 :
                self.get_logger().info('Start', once=True)
                
                w_item =  wait_item_list[-1]
                bounding_box_src = w_item._bounding_box
                x, y, width1, height1 = bounding_box_src.items
                rectangle: tuple[float, float, float, float] = [x,y ,width1,height1]
                if len(wait_item_list) >1 :
                    w_item2 =  wait_item_list[-2]
                    bounding_box_src2 = w_item2._bounding_box
                    x2, y2, width2, height2 = bounding_box_src2.items
                    rectangle2: tuple[float, float, float, float] = [x2,y2 ,width2,height2]


                self.roi = (x, y,width1+x, height1+y)

                # obj= next[self.roi[1]+300:self.roi[3]+300, self.roi[0]+300:self.roi[2]+300]
                #オプティカルフローの特徴点抽出領域を限定
                obj= next[211:711, 0:900]
                



            
                if( self.lk_reset == True ):
                    if self._count == 0:
                        self.prvs = obj
                        
                        self._count = self._count + 1
                    
                    # if len( self.track_list)> 10:
                    #     self.track_list = self.track_list[5:]




                    p0 = cv2.goodFeaturesToTrack(obj, mask = None, **feature_params)


                    self.lk_reset = False

                else :

                    p0 = self.good_new.reshape(-1,1,2)
                

                # オプティカル・フローを計算

                p1, st, err = cv2.calcOpticalFlowPyrLK(self.prvs, obj, p0, None, **lk_params)

                # 良い特徴点を選択
                self.good_new = p1[st==1]
                self.good_old = p0[st==1]



                best_overlap_count = 0

                # 物体追跡を描画
                track = np.zeros_like(img)
                for i,(new,old) in enumerate(zip(self.good_new,self.good_old)):
                    total_overlap_count = 0

                    a,b = new.ravel()
                    point = a,b+211 
                    c,d = old.ravel()
                    point2 = c,d+211
                    colar =(0,255,0)
                    # cv2.line(track, (int(a),int(b)),(int(c),int(d)), colar, 2)
                    # result = cv2.circle(result,(int(a),int(b)),5,colar,-1)

                    if len(wait_item_list) >1 :

      
                        if is_point_inside_bounding_box(point, rectangle):
                            result = cv2.circle(result,(int(a),int(b+211)),5,colar,-1)


                            if w_item._object_id == "":
                                #print('a1')

                                for t_item in wait_item_list:

                                    t_bounding_box_src = t_item._bounding_box
                                    t_x, t_y, t_width, t_height = t_bounding_box_src.items
                                    t_rectangle: tuple[float, float, float, float] = [t_x,t_y ,t_width,t_height]
                               
                                #     total_overlap_count += calculate_overlap(point, t_rectangle)
                                
                                # if  total_overlap_count > best_overlap_count:

                                #     best_overlap_count = total_overlap_count
                                #     best_bbox = t_rectangle
                                
                                    if self.tflag :


                                        if is_point_inside_bounding_box(point2, t_rectangle):

                                            if t_item._object_id != "":
                                                print('a1')
                                                self.object_id = t_item._object_id

                                                self.current_object_dict[self.object_id] = bounding_box_src.items
                                                object_id_list = self.object_id.split('_')
                                                w_item._object_id = self.object_id
                                                cv2.putText(result, f'ID : {object_id_list[-1]}', (x,y),cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 128, 255), thickness=2)
                                                self.tflag = False
                                                self.newflag = False
                                                self.t_rectangle = t_rectangle
                                            # else:
                                            #     #print('a2')
                                            #     object_id_num:int = self._tracking_info.new_object_id()
                                            #     self.current_object_dict[object_id_num] = bounding_box_src.items
                                            #     #idの最後の数字だけ抽出
                                            #     object_id_list = object_id_num.split('_')
                                            #     w_item._object_id = object_id_num
                                            #     #print(object_id_list[-1])
                                            #     cv2.putText(result, f'ID : {object_id_list[-1]}', (x, y),cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 128, 255), thickness=2)
                                            #     self.tflag = False
                                            #     self.newflag = False  
                                            #     self.t_rectangle = t_rectangle
                                                
                                        # IOUの重なり
                                        # elif check_overlap(rectangle,t_rectangle):
                                        #     if t_item._object_id != "":
                                        #         #print('a3')
                                        #         self.object_id = t_item._object_id

                                        #         self.current_object_dict[self.object_id] = bounding_box_src.items
                                        #         object_id_list = self.object_id.split('_')
                                        #         w_item._object_id = self.object_id
                                        #         cv2.putText(result, f'ID : {object_id_list[-1]}', (x,y),cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 128, 255), thickness=2)
                                        #         self.tflag = False
                                        #         self.newflag = False
                                        #         self.t_rectangle = t_rectangle

                                            # else:
                                            #     #print('a4')
                                            #     object_id_num:int = self._tracking_info.new_object_id()
                                            #     self.current_object_dict[object_id_num] = bounding_box_src.items
                                            #     #idの最後の数字だけ抽出
                                            #     object_id_list = object_id_num.split('_')
                                            #     w_item._object_id = object_id_num
                                            #     #print(object_id_list[-1])
                                            #     cv2.putText(result, f'ID : {object_id_list[-1]}', (x, y),cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 128, 255), thickness=2)
                                            #     self.tflag = False
                                            #     self.newflag = False          
                                            #     self.t_rectangle = t_rectangle  
                            
                            if is_point_inside_bounding_box(point2, rectangle2):
                                cv2.line(track, (int(a),int(b+211)),(int(c),int(d+211)), colar, 2)
                            


                        
                        if is_point_inside_bounding_box(point, rectangle2):
                            result = cv2.circle(result,(int(a),int(b+211)),5,colar,-1)
                            if w_item2._object_id == "":
                            #print('a2')
                                for t2_item in wait_item_list:
                                    t2_bounding_box_src = t2_item._bounding_box
                                    t2_x, t2_y, t2_width, t2_height = t2_bounding_box_src.items
                                    t2_rectangle: tuple[float, float, float, float] = [t2_x,t2_y ,t2_width,t2_height]

                                    if self.t2flag :


                                        if is_point_inside_bounding_box(point2, t2_rectangle):

                                            if t2_item._object_id != "":
                                                print('b1')
                                                self.object_id = t2_item._object_id

                                                self.current_object_dict[self.object_id] = bounding_box_src.items
                                                object_id_list = self.object_id.split('_')
                                                w_item2._object_id = self.object_id
                                                cv2.putText(result, f'ID : {object_id_list[-1]}', (x2,y2),cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 128, 255), thickness=2)
                                                self.t2flag = False
                                                self.newflag = False
                                                self.t_rectangle = t2_rectangle
                                            else:
                                                #print('b2')
                                                print('id add')  
                                                object_id_num:int = self._tracking_info.new_object_id()
                                                self.current_object_dict[object_id_num] = bounding_box_src.items
                                                #idの最後の数字だけ抽出
                                                object_id_list = object_id_num.split('_')
                                                w_item2._object_id = object_id_num
                                                #print(object_id_list[-1])
                                                cv2.putText(result, f'ID : {object_id_list[-1]}', (x, y),cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 128, 255), thickness=2)
                                                self.t2flag = False
                                                self.newflag = False  
                                                self.t_rectangle = t2_rectangle
                                            
                                        
                                        #IOUの重なり

                                        # elif check_overlap(rectangle2,t2_rectangle):
                                        #     if t2_item._object_id != "":
                                        #         #print('b3')
                                        #         self.object_id = t2_item._object_id

                                        #         self.current_object_dict[self.object_id] = bounding_box_src.items
                                        #         w_item2._object_id = self.object_id
                                        #         object_id_list = self.object_id.split('_')
                                        #         cv2.putText(result, f'ID : {object_id_list[-1]}', (x2,y2),cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 128, 255), thickness=2)
                                        #         self.t2flag = False
                                        #         self.newflag = False
                                        #         self.t_rectangle = t2_rectangle
                                        #     else:
                                        #         #print('b4')
                                        #         object_id_num:int = self._tracking_info.new_object_id()
                                        #         self.current_object_dict[object_id_num] = bounding_box_src.items
                                        #         #idの最後の数字だけ抽出
                                        #         object_id_list = object_id_num.split('_')
                                        #         w_item2._object_id = object_id_num
                                        #         #print(object_id_list[-1])
                                        #         cv2.putText(result, f'ID : {object_id_list[-1]}', (x, y),cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 128, 255), thickness=2)
                                        #         self.t2flag = False
                                        #         self.newflag = False          
                                        #         self.t_rectangle = t2_rectangle  




                                                    

                                    if is_point_inside_bounding_box(point2, t2_rectangle):
                                        cv2.line(track, (int(a),int(b+211)),(int(c),int(d+211)), colar, 2)





                    if is_point_inside_bounding_box(point, rectangle):

                        result = cv2.circle(result,(int(a),int(b+211)),5,colar,-1)
                        #print(len(wait_item_list))

                        



                        if  len(list(self.current_object_dict)) == 0 :
                            takeout_flag = True

                            if self.ta_box:

                                if is_point_inside_bounding_box(point2, self.ta_box ) or check_overlap(self.ta_box,rectangle):
                                    print('takeout_true')

                                    self.object_id = self.ta_object_id
                                    
                                    w_item._object_id = self.ta_object_id
                                
                                    self.current_object_dict[self.object_id] = bounding_box_src.items
                                    #idの最後の数字だけ抽出
                                    
                                    object_id_list = self.object_id.split('_')

                                    cv2.putText(result, f'ID : {object_id_list[-1]}', (x,y),cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 128, 255), thickness=2)
                                    self.flag = False
                                    self.newflag = False    
                                    takeout_flag = False
                                    
                            for b_item in bring_in_list:
                                bounding_box_src = b_item._bounding_box
                                b_x, b_y, b_width, b_height = bounding_box_src.items
                                rectangle4: tuple[float, float, float, float] = [b_x, b_y, b_width, b_height]
                                if is_point_inside_bounding_box(point2, rectangle4) :
                                    print('take_out2')
                                    b_item_id = b_item._object_id  
                                    w_item._object_id = b_item_id
                                    self.current_object_dict[b_item_id] = bounding_box_src.items

                                    #idの最後の数字だけ抽出
                                    
                                    object_id_list = b_item_id.split('_')

                                    cv2.putText(result, f'ID : {object_id_list[-1]}', (x,y),cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 128, 255), thickness=2)
                                    self.flag = False
                                    self.newflag = False    
                                    takeout_flag = False


                                               

                                                      
                              
                            if takeout_flag:  
                                print('not_take')      
                                # 新しい持ち込み候補に新規登録 
                                object_id_num:int = self._tracking_info.new_object_id() 
                                #x, y, width1, height1 = bounding_box_src.items

                                self.current_object_dict[object_id_num] = bounding_box_src.items
                                #idの最後の数字だけ抽出
                                object_id_list = object_id_num.split('_')
                                w_item._object_id = object_id_num
                                #print(object_id_list[-1])
                                cv2.putText(result, f'ID : {object_id_list[-1]}', (x, y),cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 128, 255), thickness=2)
                                self.flag = False
                                self.newflag = False 


                            
                self.previous_object_dict = self.current_object_dict
                        


                    

                    # cv2.line(track, (int(a+self.roi[0]),int(b+self.roi[1])),(int(c+self.roi[0]),int(d+self.roi[1])), colar, 2)
                    # result = cv2.circle(result,(int(a+self.roi[0]),int(b+self.roi[1])),5,colar,-1)

                    # cv2.line(track, (int(a),int(b+311)),(int(c),int(d+311)), colar, 2)
                    # result = cv2.circle(result,(int(a),int(b+311)),5,colar,-1)


                    
                                        

                self.track_list.append(track)
                if( len(self.track_list)> 5 ):
                    self.track_list.pop(0)

                for t in self.track_list :
                    result = np.where(t!=0,t,result)
                
                self.prvs = obj
            
            else:
                
                self.current_object_dict = {}
                if( len(self.track_list)> 10 ):
                    self.track_list = []


    
        if  self.newflag :
            takeout_flag = True

            if self.bounding_box:
                 if check_overlap(self.bounding_box,rectangle):
                    object_id, prev_item = list(self.previous_object_dict.items())[-1]
                    self.object_id = object_id
                                        
                    w_item._object_id = self.object_id
                            
                    self.current_object_dict[self.object_id] = bounding_box_src.items
                    #idの最後の数字だけ抽出
                                
                    object_id_list = self.object_id.split('_')

                    cv2.putText(result, f'ID : {object_id_list[-1]}', (x,y),cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 128, 255), thickness=2)
                                
                    self.flag = False
                    self.newflag = False 
                    takeout_flag = False 

            if takeout_flag:
                print('id_add')
            
                # 新しい持ち込み候補に新規登録 
                object_id_num:int = self._tracking_info.new_object_id() 
                self.current_object_dict[object_id_num] = bounding_box_src.items
                x3, y3, width3, height3 =  bounding_box_src.items
                self.bounding_box: tuple[float, float, float, float] = [x3,y3 ,width3,height3]
                #idの最後の数字だけ抽出
                object_id_list = list(object_id_num)
                w_item._object_id = object_id_num
                #cv2.putText(result, f'ID : {object_id_list[-1]}', (x, y),cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 128, 255), thickness=2)
                self.flag  = False     
                self.newflag = False



        
        result = cv2.resize(result,(width,height))
        self.print_fps(result)


            # cv2.putText(result,
            #             text="FPS:%f"%(1./duration),org=(10,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #             fontScale=1.0,color=(0,255,0),thickness=2,lineType=cv2.LINE_4)
            

        # wait bounding_boxの表示
        #print(len(wait_item_list))

        for w_item in wait_item_list:
            bounding_box_src = w_item._bounding_box
            x, y, width, height = bounding_box_src.items
            result = cv2.rectangle(result, (x, y), (x + width, y + height), (255,204,102), thickness=3)

            # object_id_list = w_item._object_id.split('_')
            # cv2.putText(result, f'ID : {object_id_list[-1]}', (x,y),cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 128, 255), thickness=2)
                                        

        for b_item in bring_in_list:
            bounding_box_src = b_item._bounding_box
            x, y, width, height = bounding_box_src.items
            b_item_id = b_item._object_id
            result = cv2.rectangle(result, (x, y), (x + width, y + height), (255,0,102), thickness=3)
            b_item_id_list = b_item_id.split('_')
            cv2.putText(result, f'ID : {b_item_id_list[-1]}', (x,y),cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 128, 255), thickness=2)

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
    y = y+311
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
    A_rect_x, A_rect_y, A_rect_width, A_rect_height = bbox
    B_rect_x, B_rect_y, B_rect_width, B_rect_height = bbox2


    a_area = (A_rect_width + 1) * ( A_rect_height + 1)
    b_area = (B_rect_width+ 1) * (B_rect_height+ 1)

    abx_mn = max(A_rect_x + A_rect_width, B_rect_x + B_rect_width)
    aby_mn = max(A_rect_y + A_rect_height,B_rect_y + B_rect_height)
    abx_mx = min(A_rect_x, B_rect_x)
    aby_mx = min(A_rect_y, B_rect_y)
    w = max(0, abx_mx - abx_mn + 1)
    h = max(0, aby_mx - aby_mn + 1)
    intersect = w*h

    iou = intersect / (a_area + b_area - intersect)
    return iou > 0.7

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




