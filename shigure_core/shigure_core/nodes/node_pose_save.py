import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from shigure_core_msgs.msg import HeaderString
from shigure_core_msgs.msg import PoseKeyPointsList

from shigure_core.db.event_repository import EventRepository
from shigure_core.db.convert_format import ConvertMsg

class PoseSaveNode(Node):
    def __init__(self):
        super().__init__('pose_save_node')

        self.signal = "None"
        self.start_flag = True
        self.wait_flag = True
        self.end_flag = True
        self.latest_sequence_id = 0
        self.frame_number = 1
        self.save_pose_data = []

        self.pub_id = 0

        # QoS Settings
        shigure_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # publisher, subscriber
        self._publisher = self.create_publisher(
            HeaderString, 
            '/shigure/current_pose_id', 
            10
        )
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
            if self.signal == 'None':
                if self.wait_flag:
                    print('待機中')
                    self.flag_controll(self.signal)
            elif self.signal == 'Start':
                if self.start_flag:
                    print('記録開始')
                    self.latest_sequence_id = EventRepository.get_pose_latest_sequence_id()
                    self.flag_controll(self.signal)

                sequence_id = self.latest_sequence_id + 1
                json_pose_key_points_list = ConvertMsg.message_to_json(pose_key_points_list)
                pose_column = (sequence_id, self.frame_number, json_pose_key_points_list)
                self.save_pose_data.append(pose_column)
                self.frame_number += 1
            elif self.signal == 'End':
                if self.end_flag:
                    EventRepository.insert_pose_meta(self.save_pose_data)
                    print('記録終了')
                    self.save_pose_data.clear()
                    self.frame_number = 1
                    self.flag_controll(self.signal)
            else:
                print('データが流れていません')

            publish_msg = HeaderString()
            publish_msg.header.stamp = pose_key_points_list.header.stamp
            publish_msg.data = str(self.pub_id)
            self._publisher.publish(publish_msg)
            self.pub_id += 1
                
        except Exception as err:
            self.get_logger().error(err)

    def flag_controll(self, state: str):
        # None
        if state == "None":
            self.wait_flag = False
            self.start_flag = True
            self.end_flag = True
        # Start
        elif state == "Start":
            self.wait_flag = True
            self.start_flag = False
            self.end_flag = True
        # End
        elif state == "End":
            self.wait_flag = True
            self.start_flag = True
            self.end_flag = False
        
def main(args=None):
    rclpy.init(args=args)

    pose_save_node = PoseSaveNode()

    rclpy.spin(pose_save_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    pose_save_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()