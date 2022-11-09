import datetime

import cv2
import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from sensor_msgs.msg import Image, CompressedImage

from shigure_core.nodes.bg_subtraction.depth_frames import DepthFrames
from shigure_core.nodes.bg_subtraction.logic import BgSubtractionLogic
from shigure_core.nodes.node_image_preview import ImagePreviewNode
from shigure_core.util import compressed_depth_util, frame_util


class BgSubtractionNode(ImagePreviewNode):

    def __init__(self):
        super().__init__('bg_subtraction_node')

        # QoS Settings
        shigure_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # ros params
        input_round_descriptor = ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER,
                                                     description='Max of input depth on debug img.')
        self.declare_parameter('input_round', 1500, input_round_descriptor)
        self.input_round: int = self.get_parameter("input_round").get_parameter_value().integer_value
        avg_round_descriptor = ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER,
                                                   description='Max of avg depth on debug img.')
        self.declare_parameter('avg_round', 1500, avg_round_descriptor)
        self.avg_round: int = self.get_parameter("avg_round").get_parameter_value().integer_value
        sd_round_descriptor = ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER,
                                                  description='Max of sd depth on debug img.')
        self.declare_parameter('sd_round', 500, sd_round_descriptor)
        self.sd_round: int = self.get_parameter("sd_round").get_parameter_value().integer_value

        self.depth_frames = DepthFrames()
        self.bg_subtraction_logic = BgSubtractionLogic()
        self.publisher_ = self.create_publisher(CompressedImage, '/shigure/bg_subtraction', 10)
        self.subscription = self.create_subscription(CompressedImage, '/rs/aligned_depth_to_color/compressedDepth',
                                                     self.get_depth_callback, shigure_qos)

        if self.is_debug_mode:
            self.get_logger().info(f'InputRound : {self.input_round}')
            self.get_logger().info(f'AvgRound : {self.avg_round}')
            self.get_logger().info(f'SdRound : {self.sd_round}')

    def get_depth_callback(self, image_rect_raw: CompressedImage):
        try:
            self.get_logger().info(f'Buffering start', once=True)
            # FPS計測
            self.frame_count_up()

            frame: np.ndarray = compressed_depth_util.convert_compressed_depth_img_to_cv2(image_rect_raw)
            result, data = self.bg_subtraction_logic.execute(self.depth_frames, frame)
            self.depth_frames.add_frame(frame)

            if result:
                self.get_logger().info(f'Buffering end', once=True)

                msg: Image = self.bridge.cv2_to_compressed_imgmsg(data, 'png')
                msg.header.stamp = image_rect_raw.header.stamp
                self.publisher_.publish(msg)

                if self.is_debug_mode:
                    # 平均/分散の算出(有効フレーム考慮)
                    valid_pixel = self.depth_frames.get_valid_pixel()
                    avg = self.depth_frames.get_average() * valid_pixel
                    sd = self.depth_frames.get_standard_deviation() * valid_pixel

                    # cv2描画用にuint8に丸め込む
                    rounded_frame = frame_util.convert_frame_to_uint8(self.depth_frames.frames[-1], self.input_round)
                    rounded_avg = frame_util.convert_frame_to_uint8(avg, self.avg_round)
                    rounded_sd = frame_util.convert_frame_to_uint8(sd, self.sd_round)

                    # 色つけ
                    color_depth = cv2.applyColorMap(rounded_frame, cv2.COLORMAP_OCEAN)
                    color_avg = cv2.applyColorMap(rounded_avg, cv2.COLORMAP_OCEAN)
                    color_sd = cv2.applyColorMap(rounded_sd, cv2.COLORMAP_OCEAN)

                    img = self.print_fps(data)
                    tile_img = cv2.vconcat([cv2.hconcat([color_depth, color_avg]), cv2.hconcat([color_sd, img])])
                    cv2.namedWindow('bg_subtraction', cv2.WINDOW_NORMAL)
                    cv2.imshow("bg_subtraction", tile_img)
                    cv2.waitKey(1)
                else:
                    print(f'[{datetime.datetime.now()}] fps : {self.fps}', end='\r')

        except Exception as err:
            self.get_logger().error(err)


def main(args=None):
    rclpy.init(args=args)

    bg_subtraction_node = BgSubtractionNode()

    try:
        rclpy.spin(bg_subtraction_node)

    except KeyboardInterrupt:
        pass

    finally:
        # 終了処理
        bg_subtraction_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
