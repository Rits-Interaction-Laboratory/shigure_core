import rclpy
from std_msgs.msg import String
from shigure_core_msgs.msg import PoseKeyPointsList

from shigure_core.db.event_repository import EventRepository

class RecordingStartSignalNode():
    def __init__(self):
        super().__init__('recording_start_signal_node')

        self.pose_record_signal: String = "None"

        self.subscrption = self.create_subscrption(
            String,
            '/HL2/pose_record_signal',
            self.callback,
            queue_size=10
        )

    def callback(self, msg: String):
        try:
            self.pose_record_signal = msg.data
        except Exception as err:
            self.get_logger().error(err)

class PoseSaveNode():
    def __init__(self):
        super().__init__('pose_save_node')

        self.subscription = self.create_subscription(
            PoseKeyPointsList,
            '/shigure/people_detection',
            self.callback,
            queue_size=10
        )

    def callback(self, pose_key_points_list: PoseKeyPointsList):
        try:
            if RecordingStartSignalNode.pose_record_signal == 'Start':
                print('記録開始')
                EventRepository.insert_pose_meta(pose_key_points_list)
            else:
                print('記録中断')
        except Exception as err:
            self.get_logger().error(err)

def main(args=None):
    rclpy.init(args=args)

    recording_start_signal_subscriber = RecordingStartSignalNode()
    pose_subscriber = PoseSaveNode()

    rclpy.spin()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    recording_start_signal_subscriber.destroy_node()
    pose_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
