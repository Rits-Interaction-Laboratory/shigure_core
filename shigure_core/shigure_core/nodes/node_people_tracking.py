import glob
import os
import random
import re
from typing import List

import cv2
import message_filters
import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, SetParametersResult
from rcl_interfaces.msg import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from openpose_ros2_msgs.msg import PoseKeyPointsList as OpenPosePoseKeyPointsList
from sensor_msgs.msg import CompressedImage, CameraInfo
from shigure_core_msgs.msg import PoseKeyPointsList as ShigurePoseKeyPointsList, PoseKeyPoints
from shigure_core_msgs.msg import FaceRecognitionResult, RecognitionHistory, PeopleFaceInfo, RecInfo, DebugImage
from shigure_core_msgs.msg import ProfileFeatureAdd, DictionaryUpdate

from shigure_core.nodes.node_image_preview import ImagePreviewNode
from shigure_core.nodes.people_tracking.logic import PeopleTrackingLogic
from shigure_core.nodes.people_tracking.tracking_info import TrackingInfo
from shigure_core.util import compressed_depth_util, pose_key_points_util

# 横顔プロフィール特徴の保存間隔（フレーム）。過剰保存を防ぐ。
PROFILE_SAVE_INTERVAL_FRAMES = 3
# 顔辞書(face_models)ディレクトリ。node_face_models / people_recognition と一致させる。
# SHIGURE_FACE_MODELS_DIR 環境変数があれば最優先。無ければ固定の永続パス ~/.shigure/face_models。
_env_face_models_dir = os.environ.get('SHIGURE_FACE_MODELS_DIR')
DIRECTORY = (
    os.path.expanduser(_env_face_models_dir)
    if _env_face_models_dir
    else os.path.expanduser('~/.shigure/face_models')
)


class PeopleTrackingNode(ImagePreviewNode):

    def __init__(self):
        super().__init__('people_tracking_node')

        # QoS Settings
        shigure_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # 顔認識結果(体ID→顔ID)の累積履歴。 {people_id: {face_id: {"score", "features_num", "total_features"}}}
        # 暫定名・確定名ともこの累積のargmaxから決める（案B: latch。get_most_likely_face_id 参照）。
        self.recognition_history = {}

        # 横顔プロフィール保存用の状態
        self.confirmed_users = {}              # {people_id: 確定 user_name}（/dictionary_update で更新）
        self.existing_users_at_startup = set()  # 起動時点で登録済みの user_name 集合
        self._prev_tracked_people_ids = set()   # 直前フレームの追跡ID（消えたIDのクリーンアップ用）
        self._profile_last_save_frame = {}      # {people_id: 最後に横顔保存したframe_count}

        # ローカル画像保存モード。true のとき追跡デバッグ画像をローカルディスクに保存する
        # （手元で解析したい人向け）。shigure_api への配信(/shigure/tracking_debug_image)は
        # このフラグとは無関係に常時行う。launch の save_image 引数で切替。
        self.declare_parameter('save_image', False,
                               ParameterDescriptor(type=ParameterType.PARAMETER_BOOL,
                                                   description='true のとき追跡デバッグ画像をローカルディスクに保存する（Web配信は常時）。'))
        self.save_image: bool = self.get_parameter('save_image').get_parameter_value().bool_value

        # 横顔プロフィール学習。true のとき確定人物の@profile顔特徴を /profile_feature_add へ配信する。
        # 有効化には color 画像が必要なため、購読条件(is_debug_mode/enable_profile_insightface)にも含める。
        self.declare_parameter('enable_profile_insightface', False,
                               ParameterDescriptor(type=ParameterType.PARAMETER_BOOL,
                                                   description='true のとき横顔プロフィール特徴を /profile_feature_add に配信する。'))
        self.enable_profile_insightface: bool = \
            self.get_parameter('enable_profile_insightface').get_parameter_value().bool_value
        if self.enable_profile_insightface:
            self._load_existing_users_at_startup()

        # publisher, subscriber
        self._publisher = self.create_publisher(
            ShigurePoseKeyPointsList,
            '/shigure/people_detection',
            10
        )
        # 体IDごとの顔ID累積スコアを配信する。people_recognition が名前確定(辞書更新)に使用する
        self._recognition_history_publisher = self.create_publisher(
            RecognitionHistory,
            '/shigure/recognition_history',
            10
        )
        # 顔認識ノード(別ノード)の検出結果を購読し、頭部座標との対応付けで体IDに紐付ける
        self.face_recognition_sub = self.create_subscription(
            FaceRecognitionResult,
            '/face_recognition/results',
            self.face_recognition_callback,
            10
        )
        # 追跡デバッグ画像(オーバーレイ)の配信先。5フレーム毎に現フレームを配信する（shigure_api のWeb表示用）。
        self._debug_image_publisher = self.create_publisher(
            DebugImage,
            '/shigure/tracking_debug_image',
            10
        )
        # 横顔プロフィール特徴の配信先。people_recognition が購読して辞書へ追加する。
        self._profile_feature_publisher = self.create_publisher(
            ProfileFeatureAdd,
            '/profile_feature_add',
            10
        )
        # 名前確定通知。people_recognition が体ID→確定名を通知する。横顔保存の可否判定に使う。
        self.dictionary_update_sub = self.create_subscription(
            DictionaryUpdate,
            '/dictionary_update',
            self.update_recognition_history,
            10
        )
        depth_subscriber = message_filters.Subscriber(
            self, 
            CompressedImage,
            '/rs/aligned_depth_to_color/compressedDepth', 
            qos_profile=shigure_qos
        )
        depth_camera_info_subscriber = message_filters.Subscriber(
            self, 
            CameraInfo,
            '/rs/aligned_depth_to_color/cameraInfo', 
            qos_profile=shigure_qos
        )
        key_points_subscriber = message_filters.Subscriber(
            self, 
            OpenPosePoseKeyPointsList, 
            '/openpose/pose_key_points', 
            qos_profile=shigure_qos
        )

        # color 画像は Web配信(5フレーム毎)・描画(is_debug_mode)・横顔保存(enable_profile_insightface)
        # で必要になる。Web配信は常時行うため、常に color を購読して callback_debug を使う。
        # （color/depth/cameraInfo は同一 RealSense 由来でタイムスタンプが揃うため 4 topic
        # 同期でも取りこぼしは増えない。color 不要なフレームは callback_debug 側で復号しない。）
        color_subscriber = message_filters.Subscriber(
            self,
            CompressedImage,
            '/rs/color/compressed',
            qos_profile=shigure_qos
        )
        self.time_synchronizer = message_filters.TimeSynchronizer(
            [depth_subscriber, key_points_subscriber, depth_camera_info_subscriber, color_subscriber], 400000)
        self.time_synchronizer.registerCallback(self.callback_debug)

        # ros params
        self.declare_parameter('threshold_distance', 1000,
                               ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER,
                                                   description='同一人物と判定する最大移動距離(mm)。'))
        self.declare_parameter('threshold_person', 0.3,
                               ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
                                                   description='人物として認めるOpenPose平均信頼度のしきい値。'))
        self.declare_parameter('neck_index', 1,
                               ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER,
                                                   description='追跡基準点とする関節番号（OpenPoseインデックス）。'))
        self.declare_parameter('face_name_score_threshold', 3.0,
                               ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
                                                   description='face_name を確定として publish する累積スコアのしきい値（コサイン類似度の累積和）。未満は未確定として空文字を publish する。'))
        self.declare_parameter('tracking_max_age', 5,
                               ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER,
                                                   description='人物を見失っても people_id を保持する最大フレーム数。'
                                                               '再出現時に同じIDへ復帰させIDの振り直し（顔名累積のリセット）を抑制する。0で従来動作。'))

        self.threshold_distance: int = self.get_parameter('threshold_distance').get_parameter_value().integer_value
        self.threshold_person: float = self.get_parameter('threshold_person').get_parameter_value().double_value
        self.neck_index: int = self.get_parameter('neck_index').get_parameter_value().integer_value
        self.face_name_score_threshold: float = self.get_parameter('face_name_score_threshold').get_parameter_value().double_value
        self.tracking_max_age: int = self.get_parameter('tracking_max_age').get_parameter_value().integer_value

        self.add_on_set_parameters_callback(self._on_set_parameters_people_tracking)

        self.tracking_info = TrackingInfo(max_age=self.tracking_max_age)
        self.people_tracking_logic = PeopleTrackingLogic()

        self._colors = []
        for i in range(255):
            self._colors.append(tuple([random.randint(128, 192) for _ in range(3)]))

    def _on_set_parameters_people_tracking(self, params: List[Parameter]) -> SetParametersResult:
        """
        パラメータ変更時に呼び出されるコールバックです.

        :param params: 変更されたパラメータのリスト
        :return: パラメータ変更の成否
        """
        for param in params:
            if param.name == 'threshold_distance':
                self.threshold_distance = param.value
                self.get_logger().info('ThresholdDistance : ' + str(self.threshold_distance))
            elif param.name == 'threshold_person':
                self.threshold_person = param.value
                self.get_logger().info('ThresholdPerson : ' + str(self.threshold_person))
            elif param.name == 'neck_index':
                self.neck_index = param.value
                self.get_logger().info('NeckIndex : ' + str(self.neck_index))
            elif param.name == 'face_name_score_threshold':
                self.face_name_score_threshold = param.value
                self.get_logger().info('FaceNameScoreThreshold : ' + str(self.face_name_score_threshold))
            elif param.name == 'save_image':
                # ローカルディスク保存の有無のみを制御する（Web配信は常時で影響しない）。
                self.save_image = param.value
                self.get_logger().info('SaveImage : ' + str(self.save_image))
            elif param.name == 'enable_profile_insightface':
                self.enable_profile_insightface = param.value
                self.get_logger().info('EnableProfileInsightface : ' + str(self.enable_profile_insightface))
            elif param.name == 'tracking_max_age':
                self.tracking_max_age = param.value
                self.tracking_info.set_max_age(param.value)
                self.get_logger().info('TrackingMaxAge : ' + str(self.tracking_max_age))
        return SetParametersResult(successful=True)

    @staticmethod
    def is_point_in_box(point, box):
        """点(x, y)が矩形box=[x, y, w, h]の内側にあるか判定する."""
        x, y = point
        box_x, box_y, box_width, box_height = box
        return box_x <= x <= box_x + box_width and box_y <= y <= box_y + box_height

    def face_recognition_callback(self, msg: FaceRecognitionResult):
        """
        顔認識ノードの検出結果を受け取り、各顔と追跡中の人物(体ID)を頭部座標で対応付ける.

        頭部キーポイント(OpenPoseインデックス0)が顔boxに入っていれば、その体IDに顔IDのスコアを
        フレームをまたいで累積する。単発ではなく累積投票で確からしさを高める。
        """
        # 横顔プロフィール保存で参照するため、最新の顔検出結果を保持する。
        self.current_face_data = msg
        for face_info in msg.faces:
            face_id = face_info.id
            score = face_info.score
            feature_num = face_info.feature_num
            face_box = list(face_info.box)
            # box は [x, y, w, h] を想定。異常な長さは対応付け不能なのでスキップする。
            if len(face_box) != 4:
                continue

            for people_id, people in self.tracking_info.get_people_dict().items():
                openpose_pose_key_points = people[-1]
                if not openpose_pose_key_points.pose_key_points:
                    continue
                head = openpose_pose_key_points.pose_key_points[0]
                head_point = (head.x, head.y)

                if PeopleTrackingNode.is_point_in_box(head_point, face_box):
                    bucket = self.recognition_history.setdefault(people_id, {})
                    if face_id in bucket:
                        bucket[face_id]['score'] += score
                        bucket[face_id]['features_num'].append(feature_num)
                        bucket[face_id]['total_features'] += 1
                    else:
                        bucket[face_id] = {
                            'score': score,
                            'features_num': [feature_num],
                            'total_features': 1,
                        }

        self._recognition_history_publisher.publish(self.create_recognition_history_message())

    def create_recognition_history_message(self) -> RecognitionHistory:
        """内部の累積履歴(recognition_history)を RecognitionHistory メッセージへ変換する."""
        recognition_history_msg = RecognitionHistory()
        recognition_history_msg.users = []

        for people_id, faces in self.recognition_history.items():
            people_face_info = PeopleFaceInfo()
            people_face_info.people_id = people_id
            people_face_info.face_info = []
            for face_id, data in faces.items():
                rec_info = RecInfo()
                rec_info.id = face_id
                rec_info.features_num = data.get('features_num', [])
                rec_info.total_features = data['total_features']
                rec_info.accumulate_score = data['score']
                people_face_info.face_info.append(rec_info)
            recognition_history_msg.users.append(people_face_info)

        return recognition_history_msg

    def get_most_likely_face_id(self, people_id):
        """体IDに対し、累積スコアが最大の顔ID(名前)とそのスコアを返す。履歴が無ければ(None, 0)を返す."""
        faces = self.recognition_history.get(people_id)
        if not faces:
            return None, 0.0
        most_likely_id, data = max(faces.items(), key=lambda item: item[1]['score'])
        display_id = most_likely_id.replace('@profile', '')
        return display_id, data['score']

    def resolve_face_name(self, people_id):
        """
        体IDに対する face_name を決める.

        累積スコアが閾値以上なら確定名（例 'user_nakamura'）を返す。
        未確定でも現フレームの暫定Top1があれば末尾に '?' を付けた暫定名（例 'user_nakamura?'）を返す。
        どちらも無ければ空文字を返す。
        """
        # 暫定・確定とも累積(recognition_history)のargmaxを使う（案B: latch）。
        # 累積は追跡が続く限り残るため、顔が見えなくなっても・別人が一瞬映っても、
        # 累積で勝っている人物の名前が保持され続ける（instantaneousな上書きで消えない）。
        name, score = self.get_most_likely_face_id(people_id)
        if name is None or name.startswith('unknown'):
            return ''
        if score >= self.face_name_score_threshold:
            return name          # 確定：?なし
        return name + '?'        # 暫定：?付き（累積argmaxなのでlatchされる）

    def _cleanup_recognition_history(self):
        """現在追跡していない体IDの顔認識履歴・確定名・横顔保存状態を破棄する.

        coasting（見失い中だが max_age 以内）の体IDは生存扱いにして履歴を残す。
        こうしないと1フレームの取りこぼしで顔名累積が消えてしまう。
        """
        current_ids = self.tracking_info.get_alive_ids()
        for people_id in list(self.recognition_history.keys()):
            if people_id not in current_ids:
                del self.recognition_history[people_id]
        # 横顔プロフィール保存の状態も、追跡から外れた体IDについて破棄する。
        lost_ids = self._prev_tracked_people_ids - current_ids
        for people_id in lost_ids:
            self.confirmed_users.pop(people_id, None)
            self._profile_last_save_frame.pop(people_id, None)
        self._prev_tracked_people_ids = current_ids

    def _load_existing_users_at_startup(self):
        """起動時点で登録済みの user_* を読み込む."""
        user_dirs = glob.glob(os.path.join(DIRECTORY, 'user_*'))
        self.existing_users_at_startup = {os.path.basename(path) for path in user_dirs}

    def update_recognition_history(self, msg: DictionaryUpdate):
        """people_recognition からの確定名通知(/dictionary_update)を受け、確定名を保持する."""
        update_people_id = msg.people_id
        new_user_name = msg.name
        self.get_logger().info(f'Received user name: {new_user_name} in {update_people_id}')

        if new_user_name != 'none':
            self.confirmed_users[update_people_id] = new_user_name
            if new_user_name.startswith('user_new'):
                self.existing_users_at_startup.add(new_user_name)

    def can_save_profile(self, people_id, user_name):
        """骨格IDとユーザー名が confirmed_users で一致していれば横顔保存可."""
        if not user_name or user_name == 'unknown' or not user_name.startswith('user'):
            return False
        return self.confirmed_users.get(people_id) == user_name

    def _try_save_profile_feature(self, people_id, user_name, embedding, face_image=None):
        """保存間隔(PROFILE_SAVE_INTERVAL_FRAMES)を守りつつ横顔特徴を配信する."""
        last_frame = self._profile_last_save_frame.get(people_id, -PROFILE_SAVE_INTERVAL_FRAMES)
        if self.frame_count - last_frame < PROFILE_SAVE_INTERVAL_FRAMES:
            return
        self._publish_profile_feature_add(user_name, embedding, face_image)
        self._profile_last_save_frame[people_id] = self.frame_count

    def _publish_profile_feature_add(self, user_name, embedding, face_image=None):
        """横顔特徴(埋め込み＋任意で顔画像)を /profile_feature_add へ配信する."""
        msg = ProfileFeatureAdd()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.user_id = user_name
        msg.embedding = np.asarray(embedding, dtype=np.float32).reshape(-1).tolist()
        if face_image is not None and face_image.size > 0:
            msg.face_image = self.bridge.cv2_to_compressed_imgmsg(face_image)
        self._profile_feature_publisher.publish(msg)
        self.get_logger().info(f'Published profile feature add: user_id={user_name}')

    def _process_profile_save(self, published_msg: ShigurePoseKeyPointsList, color_img=None):
        """確定人物について、頭部と重なる @profile 顔の特徴を横顔として保存配信する."""
        current_face_data = getattr(self, 'current_face_data', None)
        if current_face_data is None:
            return

        people_dict = self.tracking_info.get_people_dict()
        for pose_key_points in published_msg.pose_key_points_list:
            people_id = pose_key_points.people_id
            if people_id not in people_dict:
                continue

            confirmed_user_id = self.confirmed_users.get(people_id)
            if not self.can_save_profile(people_id, confirmed_user_id or ''):
                continue

            openpose_pose_key_points = people_dict[people_id][-1]
            if not openpose_pose_key_points.pose_key_points:
                continue
            head_point = openpose_pose_key_points.pose_key_points[0]
            head = (head_point.x, head_point.y)

            for face_info in current_face_data.faces:
                if not face_info.id.endswith('@profile'):
                    continue
                face_box = list(face_info.box)
                if len(face_box) != 4:
                    continue
                if not PeopleTrackingNode.is_point_in_box(head, face_box):
                    continue
                if not face_info.embedding:
                    continue

                embedding = np.asarray(face_info.embedding, dtype=np.float32)
                face_crop = self._crop_face(color_img, face_box)
                self._try_save_profile_feature(people_id, confirmed_user_id, embedding, face_crop)
                break

    @staticmethod
    def _crop_face(color_img, box):
        """color_img から box=[x, y, w, h] で顔領域を切り出す。範囲外は None."""
        if color_img is None:
            return None
        height, width = color_img.shape[:2]
        x, y, w, h = [int(v) for v in box]
        left = max(0, x)
        top = max(0, y)
        right = min(width, x + w)
        bottom = min(height, y + h)
        if right <= left or bottom <= top:
            return None
        return color_img[top:bottom, left:right].copy()

    def callback(self, depth_src: CompressedImage, key_points_list: OpenPosePoseKeyPointsList, camera_info: CameraInfo):
        self.frame_count_up()

        depth_img = compressed_depth_util.convert_compressed_depth_img_to_cv2(depth_src)
        depth_img = depth_img.astype(np.float32)

        # 焦点距離取得
        #     [fx  0 cx]
        # K = [ 0 fy cy]
        #     [ 0  0  1]
        k = camera_info.k.reshape((3, 3))

        self.tracking_info = self.people_tracking_logic.execute(
            depth_img, key_points_list, self.tracking_info, k,
            self.threshold_distance, self.threshold_person, self.neck_index
        )

        # 追跡から外れた体IDの顔認識履歴を破棄（メモリ肥大・IDの使い回し対策）
        self._cleanup_recognition_history()

        # publish
        publish_msg = ShigurePoseKeyPointsList()
        publish_msg.header = key_points_list.header
        publish_msg.header.frame_id = camera_info.header.frame_id
        for people_id, people in self.tracking_info.get_people_dict().items():
            _, _, _, openpose_pose_key_points = people
            shigure_pose_key_points = pose_key_points_util.convert_openpose_to_shigure(openpose_pose_key_points,
                                                                                       depth_img, people_id, k)
            # 顔認識で対応が取れた名前を付与する。
            # 確定(累積≥閾値)なら 'user_xxx'、未確定でも暫定Top1があれば 'user_xxx?'、無ければ空。
            shigure_pose_key_points.face_name = self.resolve_face_name(people_id)
            publish_msg.pose_key_points_list.append(shigure_pose_key_points)
        self._publisher.publish(publish_msg)

        return publish_msg

    def callback_debug(self, depth_src: CompressedImage, key_points_list: OpenPosePoseKeyPointsList,
                       camera_info: CameraInfo, color_src: CompressedImage):
        published_msg: ShigurePoseKeyPointsList = self.callback(depth_src, key_points_list, camera_info)

        # 現フレームを shigure_api(/ws/tracking_debug)へ配信するフレームか（負荷軽減のため5フレーム毎）。
        # WebUI は常駐なので配信は常時行う。save_image=true のときは同じ画像をローカル保存もする。
        publish_web = (self.frame_count % 5 == 0)

        # color 画像が要るのは Web配信 / デバッグ窓表示 / 横顔保存 のいずれか。
        # どれも不要なフレームは復号せずに抜ける（CPU節約）。people_detection は上の callback で配信済み。
        # （destroyAllWindows は waitKey と同じトピックコールバックスレッドから呼ぶ必要がある。）
        if not publish_web and not self.is_debug_mode and not self.enable_profile_insightface:
            return

        color_img: np.ndarray = self.bridge.compressed_imgmsg_to_cv2(color_src)

        # 横顔プロフィール保存（描画前の生画像から切り出すため、_draw_overlay より先に行う）
        if self.enable_profile_insightface:
            self._process_profile_save(published_msg, color_img)

        # オーバーレイ描画が要るのは Web配信フレーム または デバッグ窓表示のとき。
        if not publish_web and not self.is_debug_mode:
            return

        self._draw_overlay(color_img, published_msg)
        self.print_fps(color_img)

        # デバッグウィンドウ表示（is_debug_mode のときのみ）
        if self.is_debug_mode:
            cv2.namedWindow('people_tracking', cv2.WINDOW_NORMAL)
            cv2.imshow('people_tracking', color_img)
            cv2.waitKey(1)
        else:
            cv2.destroyAllWindows()

        # WebUI(shigure_api /ws/tracking_debug)へ現フレームのオーバーレイ画像を常時配信する。
        # save_image=true のときのみ、同じ画像をローカルディスクにも保存する（手元解析用）。
        if publish_web:
            self._publish_tracking_debug_image(color_img)
            if self.save_image:
                self._save_tracking_debug_image(color_img)

    def _draw_overlay(self, color_img: np.ndarray, published_msg: ShigurePoseKeyPointsList) -> None:
        """追跡結果(バウンディングボックスとID/顔名ラベル)を color_img へ描画する."""
        height, width = color_img.shape[:2]

        pose_key_points: PoseKeyPoints
        for pose_key_points in published_msg.pose_key_points_list:
            people_id = pose_key_points.people_id
            _, _, _, current_pose_key_points = self.tracking_info.get_people_dict()[people_id]

            ref_point = current_pose_key_points.pose_key_points[self.neck_index]

            x = int(ref_point.x) if ref_point.x < width else width - 1
            y = int(ref_point.y) if ref_point.y < height else height - 1

            bounding_box = pose_key_points.bounding_box
            left = int(bounding_box.x) if bounding_box.x < width else width - 1
            top = int(bounding_box.y) if bounding_box.y < height else height - 1
            right = int(
                bounding_box.x + bounding_box.width) if bounding_box.x + bounding_box.width < width else width - 1
            bottom = int(
                bounding_box.y + bounding_box.height) if bounding_box.y + bounding_box.height < height else height - 1

            people_id_num = int(re.sub(".*_", "", people_id))
            # face_name の状態で枠色とラベルを決める（ミスリード防止のため色を厳密に固定）:
            #   確定('user_xxx')   → 青枠(255,0,0)  ＋ 名前
            #   暫定('user_xxx?')  → 赤枠(0,0,255)  ＋ 名前?
            #   未認識('')         → 灰枠(160,160,160) ＋ 'ID : n'（体IDの色は使わない）
            face_name = pose_key_points.face_name
            if not face_name:
                color = (160, 160, 160)  # 灰：顔と未結合
                label = f'ID : {people_id_num}'
            elif face_name.endswith('?'):
                color = (0, 0, 255)      # 赤(BGR)：暫定
                label = face_name
            else:
                color = (255, 0, 0)      # 青(BGR)：確定
                label = face_name
            cv2.circle(color_img, (x, y), 5, color, thickness=-1)
            cv2.rectangle(color_img, (left, top), (right, bottom), color, thickness=3)
            text_w, text_h = cv2.getTextSize(label,
                                             cv2.FONT_HERSHEY_PLAIN, 2.5, 3)[0]
            cv2.rectangle(color_img, (left, top - text_h - 6), (left + text_w, top), color, -1)
            cv2.putText(color_img, label, (left, top - 4),
                        cv2.FONT_HERSHEY_PLAIN, 2.5, (255, 255, 255), thickness=3)

        # ① 認識中の顔box（people_recognition の検出結果）を描画する。
        self._draw_face_boxes(color_img)

    def _draw_face_boxes(self, color_img: np.ndarray) -> None:
        """現フレームの顔検出box(/face_recognition/results)を描画する（①）."""
        face_data = getattr(self, 'current_face_data', None)
        if face_data is None:
            return
        height, width = color_img.shape[:2]
        for face_info in face_data.faces:
            box = list(face_info.box)
            if len(box) != 4:
                continue
            fx, fy, fw, fh = [int(v) for v in box]
            left = max(0, fx)
            top = max(0, fy)
            right = min(width - 1, fx + fw)
            bottom = min(height - 1, fy + fh)
            # 顔ID(faiss最近傍Top1)が実名なら緑、unknownなら灰の細枠。
            fid = face_info.id.replace('@profile', '')
            face_color = (128, 128, 128) if fid.startswith('unknown') else (0, 200, 0)
            cv2.rectangle(color_img, (left, top), (right, bottom), face_color, thickness=2)

    def _publish_tracking_debug_image(self, color_img: np.ndarray) -> None:
        """オーバーレイ画像を JPEG エンコードして /shigure/tracking_debug_image に配信する(Web表示用)."""
        ok, buf = cv2.imencode('.jpg', color_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return
        msg = DebugImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(self.frame_count)
        msg.format = 'jpeg'
        msg.data = buf.tobytes()
        self._debug_image_publisher.publish(msg)

    def _save_tracking_debug_image(self, color_img: np.ndarray) -> None:
        """オーバーレイ画像をローカルの debug_images ディレクトリへ保存する（save_image=true 時のみ）."""
        if not hasattr(self, '_debug_image_dir'):
            # shigure_core パッケージ直下の debug_images/people_tracking を保存先とする。
            base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'debug_images')
            self._debug_image_dir = os.path.abspath(os.path.join(base, 'people_tracking'))
            os.makedirs(self._debug_image_dir, exist_ok=True)
            self.get_logger().info(f'追跡デバッグ画像を保存: {self._debug_image_dir}')
        cv2.imwrite(os.path.join(self._debug_image_dir, 'people_tracking_latest.jpg'), color_img)
        cv2.imwrite(
            os.path.join(self._debug_image_dir, f'people_tracking_{self.frame_count:06d}.jpg'),
            color_img,
        )


def main(args=None):
    rclpy.init(args=args)

    people_tracking_node = PeopleTrackingNode()

    try:
        rclpy.spin(people_tracking_node)

    except KeyboardInterrupt:
        pass

    finally:
        # 終了処理
        people_tracking_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
