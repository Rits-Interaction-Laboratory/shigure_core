import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from shigure_core_msgs.msg import PoseKeyPointsList

from shigure_core.db.event_repository import EventRepository

class PoseSaveNode(Node):
    def __init__(self):
        super().__init__('pose_save_node')

        self.signal = "None"
        self.start_flag = True
        self.wait_flag = True
        self.end_flag = True

        # QoS Settings
        shigure_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.signal_subscription = self.create_subscription(
            String, 
            '/HL2/pose_record_signal', 
            lambda msg: self.result_signal(msg),
            qos_profile=shigure_qos
        )
        self.pose_subscription = self.create_subscription(
            PoseKeyPointsList,
            '/shigure/people_detection',
            lambda msg: self.pose_save(msg),
            qos_profile=shigure_qos
        )

    def result_signal(self, msg):
        self.signal = msg.data

    def pose_save(self, pose_key_points_list: PoseKeyPointsList):
        try:
            if self.signal == 'Start':
                if self.start_flag:
                    print('記録開始')
                    self.start_flag = False
                EventRepository.insert_pose_meta(pose_key_points_list)
            elif self.signal == 'End':
                if self.end_flag:
                    print('記録終了')
                    self.end_flag = False   
            else:
                if self.wait_flag:
                    print('待機中')
                    self.wait_flag = False
        except Exception as err:
            self.get_logger().error(err)

def main(args=None):
    rclpy.init(args=args)

    pose_subscriber = PoseSaveNode()

    rclpy.spin(pose_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    pose_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()