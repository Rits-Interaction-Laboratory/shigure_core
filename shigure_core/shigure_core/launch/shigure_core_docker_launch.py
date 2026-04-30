from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="shigure_core",
            executable="yolox_object_detection",
            prefix="xterm -T 'yolox_object_detection' -e",
            parameters=[
                {"is_debug_mode": True},
            ],
        ),
        Node(
            package="shigure_core",
            executable="object_tracking",
            prefix="xterm -T 'object_tracking' -e",
            parameters=[
                {"is_debug_mode": True},
            ],
        ),
        Node(
            package="shigure_core",
            executable="people_tracking",
            prefix="xterm -T 'people_tracking' -e",
            parameters=[
                {"is_debug_mode": False},
                {"focal_length": 1.0},
            ],
        ),
        Node(
            package="shigure_core",
            executable="contact_detection",
            prefix="xterm -T 'contact_detection' -e",
            parameters=[
                {"is_debug_mode": True},
            ],
        )
    ])
