# coding: utf-8
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

from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
import message_filters





from shigure_core_msgs.msg import DetectedObjectList, DetectedObject, BoundingBox,PoseKeyPointsList
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

class CaptureNode(ImagePreviewNode):
	def __init__(self):
		super().__init__("caputure_node")
		# QoS Settings
		shigure_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
		color_subscriber = message_filters.Subscriber(
			self, 
			CompressedImage,
			'/rs/color/compressed', 
			qos_profile = shigure_qos
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
		#self._count = 0
		
		self._colors = []
		for i in range(255):
			self._colors.append(tuple([random.randint(128, 192) for _ in range(3)]))
		
		
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


		# オプティカルフローのコード


		# lkの特徴点の更新間隔

		if( len(sys.argv) == 1 ):
			cam = 0
		else :
			cam = int(sys.argv[1])
		
		if self.lk_count > 10:
			self.lk_reset = True
			self.lk_count =0

		elif self.lk_count == 0 :
			self.lk_reset = True

		else: 
			self.lk_reset = False
		

		self.lk_count = self.lk_count + 1


		
			
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



		if ret == False:
			print("cam not captured")
			sys.exit(1)

		# height = img.shape[0]
		# width = img.shape[1]
		img = cv2.resize(img,((int)(width/2),(int)(height/2)))

		# self.prvs = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		#
		## for HSV color space transform
		#
		hsv = np.zeros_like(img)
		hsv[...,1] = 255
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

		t_prev = time.perf_counter()
		#while(True):
		#print("FPS",cap.get(cv2.CAP_PROP_FPS))
		#print("Image Height",cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		#print("Image Width ",cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		#print("FOURCC",int(cap.get(cv2.CAP_PROP_FOURCC)))

		t_now = time.perf_counter()
		duration = t_now-t_prev
		t_prev = t_now

		# image capture
		# ret,img = cap.read()
		if ret == False:
			print("image not captured. stop.")

		# resize for processing rate
		img = cv2.resize(img,((int)(width/2),(int)(height/2)))


		# the newest key entry accepted, others flushed
		key = -1
		while True:
			next_key = cv2.waitKey(1)
			if next_key != -1 : 
				key = next_key
			else:
				break

		key = key&0xFF 

		if(ret==True):
			if key == ord('q') :
				print("a")
				# break
			elif key == 0 :        # Up arrow
				increase_param()
			elif key == 1 :        # Down arrow
				decrease_param()
			elif key == 27 :       # ESC
				reset_param()
			elif key == ord('G') : # Gaussian blur
				blur_proc = "gaussian"
				reset_param()
			elif key == ord('m') : # Median blur
				blur_proc = "median"
				reset_param()
			elif key == ord('B') : # Bilateral blur
				blur_proc = "bilateral"
				reset_param()
			elif key == ord('g') : # Monocolor
				proc = "gray"
			elif key == ord('c') : # Color
				proc = "color"
			elif key == ord('x') : # Sobel x-gradient
				proc = "sobelx"
			elif key == ord('y') : # Sobel y-gradient
				proc = "sobely"
			elif key == ord('a') : # Sobel angle
				proc = "angle"
			elif key == ord('s') : # Sobel grad
				proc = "sobel"
			elif key == ord('h') : # Hue image
				proc = "hue"
			elif key == ord('b') : # binarize (otsu)
				proc = "binary"
			elif key == ord(':') : # SIFT features
				proc = "sift"
				sift = cv2.SIFT_create()
			elif key == ord('f') : # Fanneback Denseflow
				proc = "denseflow"
			elif key == ord('l') : # Labeling
				proc = "labeling"
			elif key == ord(';') : # Labeling
				proc = "labeling2"
			elif key == ord('t') : # Lukas-Kanade feature tracking
				proc = "lktracking"
				# lk_reset = True
			elif key == ord('F') : # Face detection
				proc = "facedetect"
			elif key == ord('U') : # Upper body detection
				proc = "bodydetect"
			elif key == ord('p') : # Profile Face detection
				if proc == "facedetect":
					profile_face = not profile_face
					if profile_face:
						print("profile face mode")
					else:
						print("frontal face mode")

			elif key == ord('n') : # NDI camera sending
				if ndi_send == None:
					ret,ndi_send,ndi_frame = NDI_setup()
				else:
					NDI_finish(ndi_send)
					ndi_send = None
			elif key == ord('d') : # DoG (Difference of Gaussian) filter
				proc = "dog"
				blur_proc == "gaussian"
			elif key == ord('e') : # Histogram equalization
				proc = "hist_eq"
				blur_proc == "gaussian"
			elif key == ord('E') : # Constrast Limited Histogram equalization
				proc = "hist_eq_CLAHE"
				blur_proc == "gaussian"

			# blurring preprocessing 
			if blur_proc == "gaussian" :
				img = cv2.GaussianBlur(img.astype(np.float32),(gaussian_scale,gaussian_scale),0).astype(np.uint8)
				#img = cv2.GaussianBlur(img.astype(np.float32),(0,0),gaussian_scale).astype(np.uint8)
			elif blur_proc == "median" :
				img = cv2.medianBlur(img,median_scale)
			elif blur_proc == "bilateral" :
				img = cv2.bilateralFilter(img.astype(np.float32),bilateral_scale,75,75).astype(np.uint8)


			# image processing
			if proc == "color" :
				result = img
			elif proc == "gray" : 
				result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			elif proc == "sobelx" :
				img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				dx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=1)
				result = np.abs(dx*10)
				result = np.where(result>255,255,result).astype(np.uint8)
			elif proc == "sobely" :
				img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				dy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=1)
				result = np.abs(dy*10)
				result = np.where(result>255,255,result).astype(np.uint8)
			elif proc == "sobel" :
				img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				dx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=1)
				dy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=1)
				result = np.sqrt((dx*dx+dy*dy)/2)*10
				result = np.where(result>255,255,result).astype(np.uint8)
			elif proc == "angle" :
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				dx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
				dy = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
				#result = np.arctan2(dy,dx)
				h = gray
				h[:] = np.arctan2(dy[:],dx[:])/np.pi*128
				h = np.where(h<0, h+128, h)
				h = np.where(h>255, 255, h).astype(np.uint8)
				s = np.full(h.shape,255,np.uint8)
				#v = np.full(h.shape,255,np.uint8)
				v = np.sqrt((dx*dx+dy*dy)/2)*10
				v = np.where(v>255,255,v).astype(np.uint8)
				result = img
				result[:,:,0] = h
				result[:,:,1] = s
				result[:,:,2] = v
				result = cv2.cvtColor(result,cv2.COLOR_HSV2BGR)
			elif proc == "hue" :
				hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
				hsv[:,:,1] = 255
				hsv[:,:,2] = 255
				result = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
			elif proc =="binary" :
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				ret, result = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			elif proc == "sift" :
				gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
				kp = sift.detect(gray,None)
				#img=cv2.drawKeypoints(gray,kp,img)
				result = cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
			elif proc == "denseflow" :
				self.next = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
				flow = cv2.calcOpticalFlowFarneback(self.prvs,self.next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
				mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
				# colored denseflow
				hsv[...,0] = eang*180/np.pi/2
				hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
				rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
				result = cv2.addWeighted(img,0.5,rgb,0.5,0)
				flowx = mag*np.cos(ang)
				flowy = mag*np.sin(ang)
				#result = (np.minimum(np.abs(flowx)*100,255)).astype(np.uint8)
				for i in range(0,result.shape[0],15):
					for j in range(0,result.shape[1],15):
						try:
							cv2.line(result,(j,i),(int(j+flowx[i,j]*3),int(i+flowy[i,j]*3)),
									(255,255,255),1)
						except	:
							print('overflow')
				self.prvs = self.next
			elif proc == "labeling" :
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				ret, bin = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
				labelnum, label = cv2.connectedComponents(bin)
				#print(label.shape)
				#print(label.dtype)
				result = np.zeros_like(img)
				result = colors_hsv[label%256]
				#for i in range(label.shape[0]):
				#    for j in range(label.shape[1]):
				#        result[i,j] = colors_hsv[label[i,j]%256]
				result = cv2.cvtColor(result,cv2.COLOR_HSV2BGR)
			elif proc == "labeling2" :
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				ret, bin = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
				num, label, stats, centroids = cv2.connectedComponentsWithStats(bin)
				result = np.zeros_like(img)
				result = colors_hsv[label%256]
				result = cv2.cvtColor(result,cv2.COLOR_HSV2BGR)
				for i in range(1,num):
					if stats[i][4] < 100:
						continue
					x0 = stats[i][0]
					y0 = stats[i][1]
					x1 = stats[i][0]+stats[i][2]
					y1 = stats[i][1]+stats[i][3]
					cv2.rectangle(result,(x0,y0),(x1,y1),(0,0,255))
					cv2.putText(result,"ID: "+str(i+1),(x0,y1+15),cv2.FONT_HERSHEY_PLAIN,1,(0,255,255))
					cv2.putText(result,"S: "+str(stats[i][4]),(x0,y1+30),cv2.FONT_HERSHEY_PLAIN,1,(0,255,255))
					cv2.putText(result,"X: "+str(int(centroids[i][0])),(x1-10,y1+15),cv2.FONT_HERSHEY_PLAIN,1,(0,255,255))
					cv2.putText(result,"Y: "+str(int(centroids[i][1])),(x1-10,y1+30),cv2.FONT_HERSHEY_PLAIN,1,(0,255,255))
			elif proc == "facedetect" :
				result = img.copy()
				gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
				faces = face_cascade.detectMultiScale(gray, 1.3, 5)
				for (x,y,w,h) in faces:
					result = cv2.rectangle(result,(x,y),(x+w,y+h),(255,0,0),2)
					roi_gray = gray[y:y+h, x:x+w]
					roi_color = result[y:y+h, x:x+w]
					#eyes = eye_cascade.detectMultiScale(roi_gray)
					eyes = eye_tree.detectMultiScale(roi_gray)
					for (ex,ey,ew,eh) in eyes:
						cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
				if profile_face:
					flipped_gray = cv2.flip(gray,1) # flip horizontally
					profiles = profileface.detectMultiScale(gray, 1.3, 5)
					for (x,y,w,h) in profiles:
						result = cv2.rectangle(result,(x,y),(x+w,y+h),(0,0,255),2)
					profiles = profileface.detectMultiScale(flipped_gray, 1.3, 5)
					for (x,y,w,h) in profiles:
						H,W = flipped_gray.shape
						result = cv2.rectangle(result,(W-1-x-w,y),(W-1-x,y+h),(0,0,255),2)
			elif proc == "bodydetect" :
				result = img.copy()
				gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
				bodies = body_cascade.detectMultiScale(gray, 1.3, 5)
				for (x,y,w,h) in bodies:
					result = cv2.rectangle(result,(x,y),(x+w,y+h),(255,0,0),2)

			elif proc == "lktracking" :
				next = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				if( self.lk_reset == True ):
					self.prvs = next
					self.track_list = []
					p0 = cv2.goodFeaturesToTrack(next, mask = None, **feature_params)
					self.lk_reset = False
				else :
					p0 = self.good_new.reshape(-1,1,2)

				# オプティカル・フローを計算
				p1, st, err = cv2.calcOpticalFlowPyrLK(self.prvs, next, p0, None, **lk_params)

				# 良い特徴点を選択
				self.good_new = p1[st==1]
				self.good_old = p0[st==1]

				# 物体追跡を描画
				track = np.zeros_like(img)
				result = img.copy()
				for i,(new,old) in enumerate(zip(self.good_new,self.good_old)):
					a,b = new.ravel()
					c,d = old.ravel()
					cv2.line(track, (int(a),int(b)),(int(c),int(d)), lk_color[i].tolist(), 2)
					result = cv2.circle(result,(int(a),int(b)),3,lk_color[i].tolist(),-1)

				self.track_list.append(track)
				if( len(self.track_list)> 10 ):
					self.track_list.pop(0)

				for t in self.track_list :
					result = np.where(t!=0,t,result)

				self.prvs = next

			elif proc == "dog" :
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
				blurred_gray = cv2.GaussittanBlur(gray,(gaussian_scale*2+1,gaussian_scale*2+1),0)
				#blurred_gray = cv2.GaussianBlur(gray,(0,0),gaussian_scale*2+1)
				result = np.clip(((gray - blurred_gray)*3.0+128.0), a_min=0, a_max=255).astype(np.uint8)

			elif proc == "hist_eq" :
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				equ = cv2.equalizeHist(gray)
				result = equ

			elif proc == "hist_eq_CLAHE" :
				if clahe is None:
					clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				result = clahe.apply(gray)
				
			result = cv2.resize(result,(width,height))
			cv2.putText(result,
						text="FPS:%f"%(1./duration),org=(10,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
						fontScale=1.0,color=(0,255,0),thickness=2,lineType=cv2.LINE_4)
			

			# wait bounding_boxの表示
			for w_item in wait_item_list:
				bounding_box_src = w_item._bounding_box
				x, y, width, height = bounding_box_src.items
				result = cv2.rectangle(result, (x, y), (x + width, y + height), (255,204,102), thickness=3)
		






			#cv2.namedWindow('OpenCV Capture', cv2.WINDOW_NORMAL)
			cv2.imshow("OpenCV Capture", result)

			# send image to NDI camera
			if ndi_send != None:
				ndi_img = cv2.cvtColor(result,cv2.COLOR_BGR2BGRA)
				ndi_frame.data = ndi_img
				ndi_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX
				ndi.send_send_video_v2(ndi_send, ndi_frame)

		if ndi_send != None:
			NDI_finish(ndi_send)  
		# cap.release()
		#cv2.destroyAllWindows()
		#cv2.waitKey(1)	

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
		print()
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


	

