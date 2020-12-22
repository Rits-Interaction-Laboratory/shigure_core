import cv2
import rclpy
import numpy as np
from sensor_msgs.msg import Image, CompressedImage

from shigure.nodes.node_image_preview import ImagePreviewNode
from shigure.util import frame_util, compressed_depth_util


class RealsensePreviewNode(ImagePreviewNode):
    """Realsenseのプレビューノード"""

    depth_frame: np.ndarray
    depth_compressed_depth_frame: np.ndarray
    color_frame: np.ndarray
    color_compressed_frame: np.ndarray

    def __init__(self):
        super().__init__("realsense_preview_node")
        self.depth_subscription = self.create_subscription(Image, '/camera/depth/image_rect_raw',
                                                           self.get_depth_callback, 10)
        self.depth_compressed_depth_subscription = \
            self.create_subscription(CompressedImage,
                                     '/camera/depth/image_rect_raw/compressedDepth',
                                     self.get_depth_compressed_depth_callback,
                                     10)
        self.color_subscription = self.create_subscription(Image, '/camera/color/image_raw',
                                                           self.get_color_callback, 10)
        self.color_compressed_subscription = self.create_subscription(CompressedImage,
                                                                      '/camera/depth/image_rect_raw/compressed',
                                                                      self.get_color_compressed_callback, 10)
        self.images_attr = ['depth_frame', 'depth_compressed_depth_frame',
                            'color_frame', 'color_compressed_frame']

    def get_depth_callback(self, image_rect_raw: Image):
        src = self.bridge.imgmsg_to_cv2(image_rect_raw)
        self.depth_frame = convert_depth_to_color(src)
        self.preview()

    def get_depth_compressed_depth_callback(self, image_rect_raw: CompressedImage):
        src = compressed_depth_util.convert_compressed_depth_img_to_cv2(image_rect_raw)
        src = convert_depth_to_color(src)
        src = cv2.pyrMeanShiftFiltering(src, 20, 50)
        self.depth_compressed_depth_frame = src

    def get_color_callback(self, image_rect_raw: Image):
        self.color_frame = self.bridge.imgmsg_to_cv2(image_rect_raw, "bgr8")

    def get_color_compressed_callback(self, image_rect_raw: CompressedImage):
        raw_data = image_rect_raw.data
        buf = np.ndarray(shape=(1, len(raw_data)),
                         dtype=np.uint8, buffer=raw_data)
        img: np.ndarray = cv2.imdecode(buf, cv2.IMREAD_ANYDEPTH)
        print(img.shape, end='\r')
        self.color_compressed_frame = convert_depth_to_color(img)

    def preview(self):
        """プレビューを表示します."""

        for image_attr in self.images_attr:
            if not hasattr(self, image_attr):
                return

        tile_img = cv2.vconcat(
            [cv2.hconcat([self.depth_frame, self.depth_compressed_depth_frame]),
             cv2.hconcat([self.color_frame, self.color_compressed_frame])])
        cv2.imshow("Result", tile_img)
        cv2.waitKey(1)


def convert_depth_to_color(src: np.ndarray) -> np.ndarray:
    rounded_frame = frame_util.convert_frame_to_uint8(src.astype(np.float32), 4000)
    return cv2.applyColorMap(rounded_frame, cv2.COLORMAP_OCEAN)


def main(args=None):
    rclpy.init(args=args)

    realsense_preview_node = RealsensePreviewNode()

    try:
        rclpy.spin(realsense_preview_node)

    except KeyboardInterrupt:
        pass

    finally:
        # 終了処理
        realsense_preview_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
