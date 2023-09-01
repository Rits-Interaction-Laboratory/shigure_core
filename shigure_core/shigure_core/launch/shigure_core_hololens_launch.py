from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="shigure_core",
            executable="yolox_object_detection",
            prefix="gnome-terminal --tab -t 'yolox_object_detection' --",
            parameters=[
                {"is_debug_mode": True},
            ],
        ),
        Node(
            package="shigure_core",
            executable="object_tracking",
            prefix="gnome-terminal --tab -t 'object_tracking' --",
            parameters=[
                {"is_debug_mode": False},               
            ],
        ),
        Node(
            package="shigure_core",
            executable="people_tracking",
            prefix="gnome-terminal --tab -t 'people_tracking' --",
            parameters=[
                {"is_debug_mode": False},
                {"focal_length":1.0},
            ],
        ),
        Node(
            package="shigure_core",
            executable="contact_detection",
            prefix="gnome-terminal --tab -t 'contact_detection' --",
            parameters=[
                {"is_debug_mode": True},
            ],
        ),
        Node(
            package="shigure_core",
            executable="pose_save",
            prefix="gnome-terminal --tab -t 'pose_save' --",     
        ),
        Node(
            package="shigure_core",
            executable="record_event",
            prefix="gnome-terminal --tab -t 'record_event' --",
            parameters=[
                {"save_root_path":'/home/azuma/ros2_ws/src/shigure_core/events'},
                {"frame_num": 120},
                {"camera_id": 1},
                {"is_recording_depth_info": False},
            ],
        )
            ])
