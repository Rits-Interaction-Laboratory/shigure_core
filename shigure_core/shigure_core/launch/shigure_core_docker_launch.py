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
                {"is_debug_mode": True},
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
        ),
        # 顔認識。辞書はSHIGURE_FACE_MODELS_DIR未指定時 ~/.shigure/face_models
        # (docker-composeでホストの~/.shigureをマウントして永続化している)
        Node(
            package="shigure_core",
            executable="people_recognition",
            prefix="xterm -T 'people_recognition' -e",
            parameters=[
                {"is_debug_mode": True},
            ],
        ),
        Node(
            package="shigure_core",
            executable="pose_save",
            prefix="xterm -T 'pose_save' -e",
        ),
        Node(
            package="shigure_core",
            executable="record_event",
            prefix="xterm -T 'record_event' -e",
            parameters=[
                # コンテナ内パス。save_root_pathは~を展開しないため絶対パスで指定する
                {"save_root_path": "/ros2_ws/events"},
            ],
        )
    ])
