import datetime
from itertools import chain
import random

import re
import cv2
import message_filters
from typing import List
import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
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



class YoloxObjectDetectionNode(ImagePreviewNode):
	object_list: list
	def __init__(self):
		super().__init__("yolox_object_detection_node")
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
			#print('brack')
		
		if len(self._color_img_buffer) > 30:
			self._color_img_buffer = self._color_img_buffer[1:]
			self._color_img_frames.get(-30).new_image = color_img
		self._color_img_buffer.append(color_img)
		
		timestamp = Timestamp(color_img_src.header.stamp.sec, color_img_src.header.stamp.nanosec)
		frame = ColorImageFrame(timestamp, self._color_img_buffer[0], color_img)
		self._color_img_frames.add(frame)


		frame_object_dict,bring_in_list,wait_item_list,people_item_list,test_item_list = self.yolox_object_detection_logic.execute(yolox_bbox_src, timestamp,people,color_img,self.frame_object_list,self._judge_params,self.bring_in_list,self.wait_item_list)
		
		#if self._count == 0:
			#self.start_item_list = start_item_list
		self.bring_in_list = bring_in_list
		self.wait_item_list = wait_item_list
		self.people_item_list = people_item_list
		self.test_item_list = test_item_list		
		#count = 1
		#self._count = count
		self.frame_object_list = list(chain.from_iterable(frame_object_dict.values()))
		
		#result_img = cv2.cvtColor(subtraction_analysis_img, cv2.COLOR_GRAY2BGR)
		#print(len(bring_in_list))
		
		if self._color_img_frames.is_full():
			self.get_logger().info('Buffering end', once=True)
			
			frame = self._color_img_frames.top_frame
			
			sec, nano_sec = frame.timestamp.timestamp
			detected_object_list = DetectedObjectList()
			detected_object_list.header.stamp.sec = sec
			detected_object_list.header.stamp.nanosec = nano_sec
			detected_object_list.header.frame_id = camera_info.header.frame_id
			
			timestamp_str = str(frame.timestamp)
			if timestamp_str in frame_object_dict.keys():
				frame_object_list = frame_object_dict.get(timestamp_str)
				#print(time)
				# 検知開始時間が同じのフレームオブジェクトが検知済でなければ警告
				if not all([frame_object.is_finished() for frame_object in frame_object_list]):
					self.get_logger().warning('検知が終了していないオブジェクトを含んでいます')
					
				detected_object_list = self.create_msg(frame_object_list, detected_object_list, frame)
			
			self.detection_publisher.publish(detected_object_list)
			if self.is_debug_mode:
				img_height, img_width = color_img.shape[:2]

				result_img = color_img.copy()
				yolox_img = color_img.copy()
				
				#for s_item in start_item_list:
					#bounding_box_src = s_item._bounding_box
					#x, y, width, height = bounding_box_src.items
					#result_img = cv2.rectangle(result_img, (x, y), (x + width, y + height), (0,153,255), thickness=3)
					#brack_img = np.zeros_like(color_img)
					#img = self.print_fps(result_img)
					
				for w_item in wait_item_list:
					bounding_box_src = w_item._bounding_box
					x, y, width, height = bounding_box_src.items
					result_img = cv2.rectangle(result_img, (x, y), (x + width, y + height), (255,204,102), thickness=3)
					
				for b_item in bring_in_list:
					bounding_box_src = b_item._bounding_box
					x, y, width, height = bounding_box_src.items
					result_img = cv2.rectangle(result_img, (x, y), (x + width, y + height), (255,0,102), thickness=3)
					
				for bbox in yolox_bbox_src.bounding_boxes:

					x = bbox.xmin
					y = bbox.ymin
					xmax = bbox.xmax
					ymax = bbox.ymax
					yolox_img = cv2.rectangle(yolox_img, (x, y), (xmax, ymax), (102,204,51), thickness=3)
				
				#人物が検出されたかどうか <--1以上検出されるとTrue
				if len(self.people_item_list ) > 0:
					for bounding_box in self.people_item_list :
						peoplex_min, peopley_min, people_width, people_height = bounding_box.items
						people_item = [peoplex_min,peopley_min, people_width+peoplex_min, people_height+peopley_min]

						for bbox in bring_in_list:
							bounding_box_src = bbox._bounding_box
							class_id = bbox._class_id
							b_item_x, b_item_y, b_item_width, b_item_height = bounding_box_src.items
							object_item = [b_item_x,b_item_y ,b_item_width+b_item_x,b_item_height+b_item_y]
							
							# #人物と持ち込み物体の重なりを判定
							# if is_overlap(people_item, object_item):
							# 	#print(f'class_ed  : {class_id }')
							# 	cv2.putText(result_img, f'OVERLAP', (b_item_x, b_item_height+b_item_y),cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 75, 0), thickness=3)

				# すべての人物領域を書く
				for person in people.pose_key_points_list:
					bounding_box = person.bounding_box
					left = np.clip(int(bounding_box.x), 0, img_width- 1)
					top = np.clip(int(bounding_box.y), 0, img_height - 1)
					right = np.clip(int(bounding_box.x + bounding_box.width), 0, img_width - 1)
					bottom = np.clip(int(bounding_box.y + bounding_box.height), 0, img_height - 1)
					part_count: int  = len(person.point_data)

					for part in range(part_count):
						# 対象の位置
						pixel_point = person.point_data[part].pixel_point

						x = np.clip(int(pixel_point.x), 0, img_width - 1)
						y = np.clip(int(pixel_point.y), 0, img_height - 1)

						#骨格ポイントの表示
						cv2.circle(result_img, (x, y), 5, (255, 0, 0), thickness=-1)
						#cv2.rectangle(result_img, (left, top), (right, bottom), (255, 0, 0), thickness=3)
						# text_w, text_h = cv2.getTextSize(f'ID : {re.sub(".*_", "", person.people_id)}',
						# 						 cv2.FONT_HERSHEY_PLAIN, 1.5, 2)[0]
						
						#人物のBoundingBox表示
						#cv2.rectangle(result_img, (left, top), (left + text_w, top - text_h), (255, 0, 0), -1)
						# cv2.putText(result_img, f'ID : {re.sub(".*_", "", person.people_id)}', (left, top),
						# 	cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), thickness=2)
						

					POSE_PAIRS:List[List[int]] =  [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[8,12],[9,10],[10,11],[11,22],[11,24],[12,13],[13,14],[14,19],[14,21],[15,0],[15,17],[16,0],[16,18],[19,20],[22,11],[22,23]]
						
					# Draw Skeleton
					for pair in POSE_PAIRS:
						partA:int  = pair[0]
						partB:int  = pair[1]
						
						if person.point_data[partA].pixel_point.x > 0 and person.point_data[partA].pixel_point.y > 0 and  person.point_data[partB].pixel_point.x>0  and person.point_data[partB].pixel_point.x>0  :
							#骨格の表示
							cv2.line(result_img, (int(person.point_data[partA].pixel_point.x),int(person.point_data[partA].pixel_point.y)), (int(person.point_data[partB].pixel_point.x),int(person.point_data[partB].pixel_point.y)), (0, 255, 255), 3)
							segment: tuple[float, float, float, float] =[person.point_data[partA].pixel_point.x,person.point_data[partA].pixel_point.y,person.point_data[partB].pixel_point.x,person.point_data[partB].pixel_point.y]
							for bbox in bring_in_list:
								bounding_box_src = bbox._bounding_box
								class_id = bbox._class_id
								b_item_x, b_item_y, b_item_width, b_item_height = bounding_box_src.items
								rectangle: tuple[float, float, float, float] = [b_item_x,b_item_y ,b_item_width,b_item_height]
								
								if intersect(rectangle,segment):
									cv2.putText(result_img, f'OVERLAP', (b_item_x, b_item_height+b_item_y),cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 75, 0), thickness=3)

				for frame_obj in self.frame_object_list:
					action_str = ''
					action = frame_obj._item._action
					if action == DetectedObjectActionEnum.TAKE_OUT:
						action_str = 'TAKE_OUT'
					else:
						action_str = 'BRING_IN'
					x = frame_obj._item._bounding_box._x
					y = frame_obj._item._bounding_box._y
					
					cv2.putText(result_img, f'{action_str}', (x, y),cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), thickness=2)

					
				tile_img = cv2.hconcat([yolox_img, result_img])
				cv2.namedWindow('yolox_object_detection', cv2.WINDOW_NORMAL)
				cv2.imshow("yolox_object_detection", tile_img)
				cv2.waitKey(1)
			#else:
				#print(f'[{datetime.datetime.now()}] fps : {self.fps}', end='\r')
				
			
			
			
				
	def  create_msg(self, frame_object_list: List[FrameObject], detected_object_list: DetectedObjectList, frame: ColorImageFrame) -> DetectedObjectList:
		for frame_object in frame_object_list:
			action, bounding_box_src, size, mask_img, time, class_id= frame_object.item.items
			x, y, width, height = bounding_box_src.items
			
			detected_object = DetectedObject()
			detected_object.action = action.value
			detected_object.mask = self.bridge.cv2_to_compressed_imgmsg(mask_img, 'png')
			
			bounding_box = BoundingBox()
			bounding_box.x = float(x)
			bounding_box.y = float(y)
			bounding_box.width = float(width)
			bounding_box.height = float(height)
			detected_object.bounding_box = bounding_box
			
			detected_object_list.object_list.append(detected_object)
			
			self.frame_object_list.remove(frame_object)
			
			if self.is_debug_mode:
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
	  
def main(args=None):
	rclpy.init(args=args)
	
	yolox_object_detection_node = YoloxObjectDetectionNode()
	
	try:
		rclpy.spin(yolox_object_detection_node)
		
	except KeyboardInterrupt:
		pass
		
	finally:
		print()
		# 終了処理
		yolox_object_detection_node.destroy_node()
		rclpy.shutdown()

# def iou(a, b):
#     # a, bは矩形を表すリストで、a=[xmin, ymin, xmax, ymax]
#     ax_mn, ay_mn, ax_mx, ay_mx = a[0], a[1], a[2], a[3]
#     bx_mn, by_mn, bx_mx, by_mx = b[0], b[1], b[2], b[3]

#     a_area = (ax_mx - ax_mn + 1) * (ay_mx - ay_mn + 1)
#     b_area = (bx_mx - bx_mn + 1) * (by_mx - by_mn + 1)

#     abx_mn = max(ax_mn, bx_mn)
#     aby_mn = max(ay_mn, by_mn)
#     abx_mx = min(ax_mx, bx_mx)
#     aby_mx = min(ay_mx, by_mx)
#     w = max(0, abx_mx - abx_mn + 1)
#     h = max(0, aby_mx - aby_mn + 1)
#     intersect = w*h

#     iou = intersect / (a_area + b_area - intersect)
#     return iou
def is_overlap(Personbox, Bring_inbox):
	"""物体と思われるものの規定の物体でないものかどうか調べる関数
	Args:
		Personbox: 人物のバウンディングボックスの座標 (左上隅のx, y座標, 右下隅のx, y座標)
		Bring_inbox: Bring_in判定されているバウンディングボックスの座標 (左上隅のx, y座標, 右下隅のx, y座標)
	Returns:
		bool: 重なっている場合はTrue、そうでない場合はFalseを返します
	"""
	px_mn, py_mn, px_mx, py_mx = Personbox[0], Personbox[1], Personbox[2],Personbox[3]
	bx_mn, by_mn, bx_mx, by_mx = Bring_inbox[0], Bring_inbox[1], Bring_inbox[2], Bring_inbox[3]
	# 2つのバウンディングボックスが重なっているかどうかを判定
	if (px_mn <= bx_mx and  px_mx >= bx_mn) and (py_mn <=  by_mx and py_mx >= by_mn):
		return True
	else: 
		return False

def is_unknown_object(class_id: str, probability: float, object_threshold=0.48) -> bool:
	"""物体と思われるものの規定の物体でないものかどうか調べる関数
	Args:
		class_id (str): 物体のクラス名
		probability (float): 物体かどうかの確からしさ（max 1）
		object_threshold (float): 物体と判定するしきい値（max 1）
	Returns:
		bool: 物体と思われるものの規定の物体でないものかどうか
	"""
	DEFAULT_OBJECTS = ['person','chair','laptop','tv','microwave','refrigerator','potted plant','cup','keyboard','couch','mouse']
	is_object: bool = probability > object_threshold
	is_default_object = class_id in DEFAULT_OBJECTS
	return is_object and not(is_default_object)   
     
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

if __name__ == '__main__':
	main()
