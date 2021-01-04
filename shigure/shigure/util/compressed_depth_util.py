import cv2
import numpy as np

from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge


def convert_compressed_depth_img_to_cv2(src: CompressedImage) -> np.ndarray:
    if 'PNG' in src.data[:12]:
        depth_header_size = 0
    else:
        depth_header_size = 12
    raw_data = src.data[depth_header_size:]

    buf = np.ndarray(shape=(1, len(raw_data)),
                     dtype=np.uint8, buffer=raw_data)
    img: np.ndarray = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    return img


def convert_cv2_to_compressed_depth_img(src: np.ndarray, bridge: CvBridge) -> CompressedImage:
    msg: CompressedImage = bridge.cv2_to_compressed_imgmsg(src, dst_format='png')
    msg.format = '16UC1; compressedDepth'
    # refs https://github.com/ros-perception/image_transport_plugins/blob/indigo-devel
    # /compressed_depth_image_transport/src/codec.cpp
    msg.data = "\x00\x00\x00\x00\x88\x9c\x5c\xaa\x00\x40\x4b\xb7" + msg.data
    return msg
