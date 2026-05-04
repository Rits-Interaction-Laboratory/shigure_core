import cv2
import message_filters
from typing import List
import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, SetParametersResult
from rcl_interfaces.msg import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, CameraInfo
from shigure_core_msgs.msg import DetectedObjectList, DetectedObject, BoundingBox, PoseKeyPointsList
from bboxes_ex_msgs.msg import BoundingBoxes



from shigure_core.nodes.node_image_preview import ImagePreviewNode
from shigure_core.nodes.yolox_object_detection.frame_object import FrameObject
from shigure_core.nodes.yolox_object_detection.logic import YoloxObjectDetectionLogic
from shigure_core.nodes.yolox_object_detection.visualizer import YoloxVisualizer

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
		# ros params
		self.declare_parameter('fhist_size', 10,
			ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER,
				description='判定に使用するフレーム数。'))
		self.fhist_size = self.get_parameter('fhist_size').get_parameter_value().integer_value

		self.declare_parameter('confirm_threshold', 0.6,
			ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
				description='持ち込み確定しきい値。'))
		self.confirm_threshold = self.get_parameter('confirm_threshold').get_parameter_value().double_value

		self.declare_parameter('dismiss_threshold', 0.5,
			ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
				description='追跡破棄しきい値。'))
		self.dismiss_threshold = self.get_parameter('dismiss_threshold').get_parameter_value().double_value

		self.declare_parameter('take_out_threshold', 0.5,
			ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
				description='持ち出し確定しきい値。'))
		self.take_out_threshold = self.get_parameter('take_out_threshold').get_parameter_value().double_value

		self.declare_parameter('match_dist_threshold', 10,
			ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER,
				description='同一物体と判定する距離しきい値。'))
		self.match_dist_threshold = self.get_parameter('match_dist_threshold').get_parameter_value().integer_value

		self.declare_parameter('probability_threshold', 0.3,
			ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
				description='物体検出確率しきい値。'))
		self.probability_threshold = self.get_parameter('probability_threshold').get_parameter_value().double_value

		self.declare_parameter('stuffed_toy_threshold', 0.2,
			ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
				description='ぬいぐるみ専用物体検出確率しきい値。'))
		self.stuffed_toy_threshold = self.get_parameter('stuffed_toy_threshold').get_parameter_value().double_value

		self.declare_parameter('buffer_size', 25,
			ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER,
				description='カラー画像のバッファサイズ。'))
		self.buffer_size = self.get_parameter('buffer_size').get_parameter_value().integer_value
		self.yolox_object_detection_logic.buffer_size = self.buffer_size

		self.add_on_set_parameters_callback(self._on_set_parameters_yolox)

		
	def _on_set_parameters_yolox(self, params: List[Parameter]) -> SetParametersResult:
		"""
		パラメータ変更時に呼び出されるコールバックです.

		:param params: 変更されたパラメータのリスト
		:return: パラメータ変更の成否
		"""
		for param in params:
			if param.name == 'fhist_size':
				self.fhist_size = param.value
				self.get_logger().info(f'FhistSize : {param.value}')
			elif param.name == 'confirm_threshold':
				self.confirm_threshold = param.value
				self.get_logger().info(f'ConfirmThreshold : {param.value}')
			elif param.name == 'dismiss_threshold':
				self.dismiss_threshold = param.value
				self.get_logger().info(f'DismissThreshold : {param.value}')
			elif param.name == 'take_out_threshold':
				self.take_out_threshold = param.value
				self.get_logger().info(f'TakeOutThreshold : {param.value}')
			elif param.name == 'match_dist_threshold':
				self.match_dist_threshold = param.value
				self.get_logger().info(f'MatchDistThreshold : {param.value}')
			elif param.name == 'probability_threshold':
				self.probability_threshold = param.value
				self.get_logger().info(f'ProbabilityThreshold : {param.value}')
			elif param.name == 'stuffed_toy_threshold':
				self.stuffed_toy_threshold = param.value
				self.get_logger().info(f'StuffedToyThreshold : {param.value}')
			elif param.name == 'buffer_size':
				self.buffer_size = param.value
				self.yolox_object_detection_logic.buffer_size = param.value
				self.get_logger().info(f'BufferSize : {param.value}')
		return SetParametersResult(successful=True)

	def callback(self, yolox_bbox_src: BoundingBoxes, people: PoseKeyPointsList, color_img_src: CompressedImage, camera_info: CameraInfo):
		self.get_logger().info('Buffering start', once=True)
		self.frame_count_up()
		color_img: np.ndarray = self.bridge.compressed_imgmsg_to_cv2(color_img_src)
		sec = color_img_src.header.stamp.sec
		nano_sec = color_img_src.header.stamp.nanosec

		# ロジックの実行 (yoloxの物体検出結果から，物体の持ち込み/移動/持ち出し<イベント>を判定する)
		self.yolox_object_detection_logic.execute(
			yolox_bbox_src, sec, nano_sec, people, color_img,
			self.fhist_size, self.confirm_threshold, self.dismiss_threshold,
			self.take_out_threshold, self.match_dist_threshold,
			self.probability_threshold, self.stuffed_toy_threshold
		)
		self.frame_object_list = self.yolox_object_detection_logic.consume_frame_object_list()
		
		self.get_logger().info('Buffering end', once=True)
		
		detected_object_list = DetectedObjectList()
		detected_object_list.header.stamp.sec = sec
		detected_object_list.header.stamp.nanosec = nano_sec
		detected_object_list.header.frame_id = camera_info.header.frame_id

		if self.frame_object_list:		
			detected_object_list = self.create_msg(self.frame_object_list, detected_object_list)
		
		self.detection_publisher.publish(detected_object_list)

		if self.is_debug_mode:
			YoloxVisualizer.draw(
				color_img, yolox_bbox_src,
				self.yolox_object_detection_logic.wait_item_list,
				self.yolox_object_detection_logic.bring_in_list,
				people, self.frame_object_list
			)
		else:
			cv2.destroyAllWindows()
			
			
			
			
				
	def create_msg(self, frame_object_list: List[FrameObject], detected_object_list: DetectedObjectList) -> DetectedObjectList:
		for frame_object in frame_object_list:
			action, bounding_box_src, _, mask_img, _, _ = frame_object.item.items
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
			detected_object.bounding_box = bounding_box
			
			detected_object_list.object_list.append(detected_object)

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


if __name__ == '__main__':
	main()
