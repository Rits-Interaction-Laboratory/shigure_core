from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="shigure_core",
            executable="bg_subtraction",
            prefix="gnome-terminal --tab -t 'bg_subtraction' --",
            parameters=[
                {"is_debug_mode": False},
                {"input_round":1500},
                {"avg_round":1500},
                {"sd_round":500},
            ],
        ),
        Node(
            package="shigure_core",
            executable="subtraction_analysis",
            prefix="gnome-terminal --tab -t 'subtraction_analysis' --",
            parameters=[
                {"is_debug_mode": False},
            ],
        ),
        Node(
            package="shigure_core",
            executable="object_detection",
            prefix="gnome-terminal --tab -t 'object_detection' --",
            parameters=[
                {"is_debug_mode": False},
            ],
        ),
        Node(
            package="shigure_core",
            executable="object_tracking",
            prefix="gnome-terminal --tab -t 'object_tracking' --",
            parameters=[
                {"is_debug_mode": True},               
            ],
        ),
        Node(
            package="shigure_core",
            executable="people_tracking",
            prefix="gnome-terminal --tab -t 'people_tracking' --",
            parameters=[
                {"is_debug_mode": True},
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
    ])