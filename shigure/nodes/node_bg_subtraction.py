import cv2
import numpy as np
import rclpy
from sensor_msgs.msg import Image, CompressedImage

from shigure.nodes.bg_subtraction.depth_frames import DepthFrames
from shigure.nodes.bg_subtraction.logic import BgSubtractionLogic
from shigure.nodes.node_image_preview import ImagePreviewNode
from shigure.util import compressed_depth_util, frame_util


class BgSubtractionNode(ImagePreviewNode):

    def __init__(self):
        super().__init__('bg_subtraction_node')
        self.depth_frames = DepthFrames()
        self.bg_subtraction_logic = BgSubtractionLogic()
        self.publisher_ = self.create_publisher(Image, '/shigure/bg_subtraction', 10)
        self.subscription = self.create_subscription(CompressedImage, '/rs/aligned_depth_to_color/compressedDepth',
                                                     self.get_depth_callback, 10)

    def get_depth_callback(self, image_rect_raw: CompressedImage):
        try:
            # FPS計測
            self.frame_count_up()

            frame: np.ndarray = compressed_depth_util.convert_compressed_depth_img_to_cv2(image_rect_raw)
            result, data = self.bg_subtraction_logic.execute(self.depth_frames, frame)
            self.depth_frames.add_frame(frame)

            if result:
                msg: Image = self.bridge.cv2_to_imgmsg(data)
                msg.header.stamp = image_rect_raw.header.stamp
                self.publisher_.publish(msg)

                if self.is_debug_mode:
                    # 平均/分散の算出(有効フレーム考慮)
                    valid_pixel = self.depth_frames.get_valid_pixel()
                    avg = self.depth_frames.get_average() * valid_pixel
                    sd = self.depth_frames.get_standard_deviation() * valid_pixel

                    # cv2描画用にuint8に丸め込む
                    rounded_frame = frame_util.convert_frame_to_uint8(self.depth_frames.frames[-1], 1500)
                    rounded_avg = frame_util.convert_frame_to_uint8(avg, 1500)
                    rounded_sd = frame_util.convert_frame_to_uint8(sd, 1500)

                    # 色つけ
                    color_depth = cv2.applyColorMap(rounded_frame, cv2.COLORMAP_OCEAN)
                    color_avg = cv2.applyColorMap(rounded_avg, cv2.COLORMAP_OCEAN)
                    color_sd = cv2.applyColorMap(rounded_sd, cv2.COLORMAP_OCEAN)

                    img = self.print_fps(data)
                    tile_img = cv2.vconcat([cv2.hconcat([color_depth, color_avg]), cv2.hconcat([color_sd, img])])
                    cv2.imshow("Result", tile_img)
                    cv2.waitKey(1)

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
