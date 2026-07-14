"""YOLO11 + 顔認識パイプライン（既定 launch）。

launch 引数:
  debug_mode:=true/false    全ノードの is_debug_mode を制御（既定 false）。
                            true で各ノードの cv2 デバッグ窓を表示し、
                            people_recognition は自動登録ユーザーの .npy/.jpg をディスク保存する。
  save_image:=true/false    true で people_tracking が追跡デバッグ画像を
                            /shigure/tracking_debug_image へ配信・保存し、Web表示(shigure_api)を起動（既定 false）。
  enable_profile:=true/false true で横顔プロフィール特徴を /profile_feature_add に配信（既定 false）。
  terminal:=gnome-terminal/xterm/none
                            各ノードを開く端末エミュレータ（既定 gnome-terminal）。
                            Docker では xterm、ヘッドレス環境では none を指定する。
  record:=true/false        true で DB 保存系ノード（pose_save / record_event）も起動する（既定 false）。
  save_root_path:=<path>    record_event のイベント画像保存先（既定はノード側のデフォルト値）。

例:
  ros2 launch shigure_core shigure_core_launch.py
  ros2 launch shigure_core shigure_core_launch.py debug_mode:=true save_image:=true
  ros2 launch shigure_core shigure_core_launch.py terminal:=xterm record:=true  # Docker と同等の構成
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node


def _to_bool(value: str) -> bool:
    """launch引数の文字列をboolへ変換します."""
    return value.lower() in ('true', '1', 'yes')


def _make_prefix(terminal: str, title: str) -> str:
    """端末エミュレータ種別からノード起動prefixを作ります."""
    if terminal == 'gnome-terminal':
        return f"gnome-terminal --tab -t '{title}' --"
    if terminal == 'xterm':
        return f"xterm -T '{title}' -e"
    return ''


def _setup_nodes(context):
    """launch引数を解決してノード一覧を構築します."""
    debug_mode = _to_bool(context.launch_configurations['debug_mode'])
    save_image = _to_bool(context.launch_configurations['save_image'])
    enable_profile = _to_bool(context.launch_configurations['enable_profile'])
    record = _to_bool(context.launch_configurations['record'])
    terminal = context.launch_configurations['terminal']
    save_root_path = context.launch_configurations['save_root_path']

    is_debug = {'is_debug_mode': debug_mode}

    def make_node(executable, package='shigure_core', parameters=None):
        kwargs = {}
        prefix = _make_prefix(terminal, executable)
        if prefix:
            kwargs['prefix'] = prefix
        if parameters:
            kwargs['parameters'] = parameters
        return Node(package=package, executable=executable, **kwargs)

    nodes = [
        make_node('yolox_object_detection', parameters=[is_debug]),
        make_node('object_tracking', parameters=[is_debug]),
        make_node('people_tracking', parameters=[
            is_debug,
            {'focal_length': 1.0},
            {'save_image': save_image},
            {'enable_profile_insightface': enable_profile},
        ]),
        # 顔認識（/face_recognition/results, /feature_info, /dictionary_update を配信）
        # is_debug_mode=true のとき自動登録ユーザーの特徴/画像をディスク保存する。
        make_node('people_recognition', parameters=[is_debug]),
        make_node('contact_detection', parameters=[is_debug]),
    ]

    # Web API（save_image:=true のときのみ起動）
    if save_image:
        nodes.append(make_node('shigure_api', package='shigure_api'))

    # DB 保存系（record:=true のときのみ起動）
    if record:
        nodes.append(make_node('pose_save'))
        record_event_params = [is_debug]
        if save_root_path:
            record_event_params.append({'save_root_path': save_root_path})
        nodes.append(make_node('record_event', parameters=record_event_params))

    return nodes


def generate_launch_description():
    """launch定義を生成します."""
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
        DeclareLaunchArgument(
            'terminal',
            default_value='gnome-terminal',
            description='各ノードを開く端末エミュレータ（gnome-terminal / xterm / none）。',
        ),
        DeclareLaunchArgument(
            'record',
            default_value='false',
            description='true のとき DB 保存系ノード（pose_save / record_event）も起動する。',
        ),
        DeclareLaunchArgument(
            'save_root_path',
            default_value='',
            description='record_event のイベント画像保存先（未指定ならノード側のデフォルト値）。',
        ),
        OpaqueFunction(function=_setup_nodes),
    ])
