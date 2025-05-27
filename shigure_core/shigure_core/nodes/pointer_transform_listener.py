import sys
import math

from geometry_msgs.msg import Twist

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros import LookupException
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped

class PointerTransformListener(Node):

    def __init__(self):
        super().__init__('pointer_transform_listener_node')
        self.from_frame_name = "pointer"
        self.to_frame_name = "camera"
        self.get_logger().info("Transforming from {} to {}".format(self.from_frame_name, self.to_frame_name))
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.subscription = self.create_subscription(
            String,
            '/start_tf_listener',
            self.callback,
            10)
        self.subscription

        self.cmd_ = PointStamped()
        self.publisher_ = self.create_publisher(PointStamped, "/raycastHit", 10)
    
    def callback(self, msg):
        self.get_logger().info('recieve message from publish_pointer_tf')
        try:
            trans = self._tf_buffer.lookup_transform(self.to_frame_name, self.from_frame_name, rclpy.time.Time())
            
            self.cmd_.point.x = trans.transform.translation.x * 1000
            self.cmd_.point.y = trans.transform.translation.y * 1000
            self.cmd_.point.z = trans.transform.translation.z * 1000
            
            # タイムスタンプとフレーム名を追加
            self.cmd_.header.stamp = self.get_clock().now().to_msg()
            self.cmd_.header.frame_id = 'RaycastHitPointer'
            
            self.publisher_.publish(self.cmd_)
            self.get_logger().info('send message to node_raycast_hit_detection')

        except LookupException as e:
            self.get_logger().error('failed to get transform {} \n'.format(repr(e)))

def main(): # argv=sys.argv
    rclpy.init() # args=argv
    node = PointerTransformListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
