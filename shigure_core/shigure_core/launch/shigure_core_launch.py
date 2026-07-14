"""YOLO11 + 顔認識パイプライン（既定 launch）。

launch 引数:
  debug_mode:=true/false    全ノードの is_debug_mode を制御（既定 false）。
                            true で各ノードの cv2 デバッグ窓を表示し、
                            people_recognition は自動登録ユーザーの .npy/.jpg をディスク保存する。
  save_image:=true/false    true で people_tracking が追跡デバッグ画像を
                            /shigure/tracking_debug_image へ配信・保存し、Web表示(shigure_api)を起動（既定 false）。
  enable_profile:=true/false true で横顔プロフィール特徴を /profile_feature_add に配信（既定 false）。

例:
  ros2 launch shigure_core shigure_core_launch.py
  ros2 launch shigure_core shigure_core_launch.py debug_mode:=true save_image:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    debug_mode = LaunchConfiguration('debug_mode')
    save_image = LaunchConfiguration('save_image')
    enable_profile = LaunchConfiguration('enable_profile')

    # 全ノード共通で使う is_debug_mode パラメータ（launch 引数 debug_mode で制御）
    is_debug = {"is_debug_mode": ParameterValue(debug_mode, value_type=bool)}

    return LaunchDescription([
        DeclareLaunchArgument(
            'debug_mode',
            default_value='false',
            description='true のとき全ノードの is_debug_mode を有効化（cv2デバッグ窓表示・顔データのディスク保存）。',
        ),
        DeclareLaunchArgument(
            'save_image',
            default_value='false',
            description='true のとき追跡デバッグ画像を保存・配信し、Web表示(shigure_api)を起動する。',
        ),
        DeclareLaunchArgument(
            'enable_profile',
            default_value='false',
            description='true のとき横顔プロフィール特徴を /profile_feature_add に配信する（横顔学習）。',
        ),
        Node(
            package="shigure_core",
            executable="yolox_object_detection",
            prefix="gnome-terminal --tab -t 'yolox_object_detection' --",
            parameters=[
                is_debug,
            ],
        ),
        Node(
            package="shigure_core",
            executable="object_tracking",
            prefix="gnome-terminal --tab -t 'object_tracking' --",
            parameters=[
                is_debug,
            ],
        ),
        Node(
            package="shigure_core",
            executable="people_tracking",
            prefix="gnome-terminal --tab -t 'people_tracking' --",
            parameters=[
                is_debug,
                {"focal_length": 1.0},
                {"save_image": ParameterValue(save_image, value_type=bool)},
                {"enable_profile_insightface": ParameterValue(enable_profile, value_type=bool)},
            ],
        ),
        # 顔認識（/face_recognition/results, /feature_info, /dictionary_update を配信）
        # is_debug_mode=true のとき自動登録ユーザーの特徴/画像をディスク保存する。
        Node(
            package="shigure_core",
            executable="people_recognition",
            prefix="gnome-terminal --tab -t 'people_recognition' --",
            parameters=[
                is_debug,
            ],
        ),
        Node(
            package="shigure_core",
            executable="contact_detection",
            prefix="gnome-terminal --tab -t 'contact_detection' --",
            parameters=[
                is_debug,
            ],
        ),
        # Web API（save_image:=true のときのみ起動）
        Node(
            package="shigure_api",
            executable="shigure_api",
            prefix="gnome-terminal --tab -t 'shigure_api' --",
            condition=IfCondition(save_image),
        ),
    ])
