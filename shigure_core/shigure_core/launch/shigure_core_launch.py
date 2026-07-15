"""YOLO11 + 顔認識パイプライン（既定 launch）。

launch 引数（いずれも「起動時の初期値」。稼働中は ros2 param set で切替可能）:
  debug_mode:=true/false    全ノードの is_debug_mode を制御（既定 false）。
                            true で各ノードの cv2 デバッグ窓を表示する（表示のみ。
                            顔データのディスク保存は save_registration で別管理）。
  save_image:=true/false    true で people_tracking が追跡デバッグ画像を
                            /shigure/tracking_debug_image へ配信・保存する（既定 false）。
  enable_profile:=true/false true で横顔プロフィール特徴を /profile_feature_add に配信（既定 false）。
  save_registration:=true/false true で people_recognition が新規/更新の顔特徴・画像・
                            PCAモデルをディスク保存する（既定 false。is_debug_mode とは独立）。

  ※ shigure_api（Web表示）は常時起動する。save_image が false の間は画像が配信されないため
    表示されないだけで、サーバ自体はストレージを消費しない。

実行中の切替（再起動不要）:
  ros2 param set /people_recognition_node save_registration false  # 顔データのディスク保存を停止
  ros2 param set /people_tracking_node    save_image        false  # 追跡画像の配信/保存を停止
  ros2 param set /people_tracking_node    enable_profile_insightface false  # 横顔特徴の配信を停止
  ros2 param set /people_tracking_node    is_debug_mode     true   # デバッグ窓の表示（保存はしない）

例:
  ros2 launch shigure_core shigure_core_launch.py
  ros2 launch shigure_core shigure_core_launch.py debug_mode:=true save_image:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    debug_mode = LaunchConfiguration('debug_mode')
    save_image = LaunchConfiguration('save_image')
    enable_profile = LaunchConfiguration('enable_profile')
    save_registration = LaunchConfiguration('save_registration')

    # 全ノード共通で使う is_debug_mode パラメータ（launch 引数 debug_mode で制御）
    is_debug = {"is_debug_mode": ParameterValue(debug_mode, value_type=bool)}

    return LaunchDescription([
        DeclareLaunchArgument(
            'debug_mode',
            default_value='false',
            description='true のとき全ノードの is_debug_mode を有効化（cv2デバッグ窓の表示のみ。'
                        '顔データのディスク保存は save_registration で別管理）。',
        ),
        DeclareLaunchArgument(
            'save_image',
            default_value='false',
            description='true のとき追跡デバッグ画像を /shigure/tracking_debug_image へ配信・保存する。',
        ),
        DeclareLaunchArgument(
            'enable_profile',
            default_value='false',
            description='true のとき横顔プロフィール特徴を /profile_feature_add に配信する（横顔学習）。',
        ),
        DeclareLaunchArgument(
            'save_registration',
            default_value='false',
            description='true のとき people_recognition が新規/更新の顔特徴・画像・PCAモデルを'
                        'ディスクへ保存する（is_debug_mode とは独立）。',
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
        # is_debug_mode は表示用。顔特徴/画像のディスク保存は save_registration で制御する。
        Node(
            package="shigure_core",
            executable="people_recognition",
            prefix="gnome-terminal --tab -t 'people_recognition' --",
            parameters=[
                is_debug,
                {"save_registration": ParameterValue(save_registration, value_type=bool)},
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
        # Web API（常駐）。実行中に people_tracking の save_image を切り替えれば
        # /shigure/tracking_debug_image の配信 ON/OFF で表示可否を制御できる。
        # サーバ自体はストレージを消費しないため常時起動しておく。
        Node(
            package="shigure_api",
            executable="shigure_api",
            prefix="gnome-terminal --tab -t 'shigure_api' --",
        ),
    ])
