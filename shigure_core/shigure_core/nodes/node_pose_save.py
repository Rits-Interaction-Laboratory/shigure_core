import datetime
import re
from typing import List, Tuple

import numpy as np
import rclpy
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage, CameraInfo
from shigure_core_msgs.msg import TrackedObjectList

from shigure_core.enum.contact_action_enum import ContactActionEnum
from shigure_core.enum.tracked_object_action_enum import TrackedObjectActionEnum
from shigure_core.nodes.node_image_preview import ImagePreviewNode
from shigure_core.nodes.object_detection.frame_object import FrameObject

class PoseSaveNode():
    def __init__(self):
        super().__init__('pose_save_node')

        self.subscription = self.create_subscription(
            TrackedObjectList,
            '/shigure/people_detection',
            self.callback,
            query_size=10
        )

class RecordingStartSignalNode():
    def __init__(self):
        super().__init__('recording_start_signal_node')

        self.subscrption = self.create_subscrption(
            String,
            '/HL2/pose_record_signal',
            self.callback,
            query_size=1
        )

def main(args=None):
    rclpy.init(args=args)

    pose_subscriber = PoseSaveNode()
    recording_start_signal_subscriber = RecordingStartSignalNode()

    rclpy.spin()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    pose_subscriber.destroy_node()
    recording_start_signal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
