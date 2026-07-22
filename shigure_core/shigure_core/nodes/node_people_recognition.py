from typing import List

import cv2
import message_filters
import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, SetParametersResult, Parameter
from sensor_msgs.msg import CompressedImage
from shigure_core_msgs.msg import FaceRecognitionResult, FaceInfo, RecognitionHistory, DictionaryUpdate, FeatureInfo, ProfileFeatureAdd

import os
import glob
import time
from pathlib import Path
from PIL import Image

from shigure_core.nodes.node_image_preview import ImagePreviewNode
from shigure_core.nodes.profile_insightface import ProfileInsightFace, InsightFaceResult
from shigure_core.util.face_models_dir import get_face_models_dir
from shigure_core.util.pca_model import build_pca_model

from sklearn.semi_supervised import LabelPropagation
from collections import Counter

from std_msgs.msg import Header
import faiss

# COSINE_THRESHOLD = 0.363
COSINE_THRESHOLD = 0.4
PROFILE_COSINE_THRESHOLD = 0.4
NORML2_THRESHOLD = 1.128
LP_CONFIDENCE_THRESHOLD = 0.7
FAISS_FALLBACK_MIN_VOTE_RATIO = 0.5
MIN_DET_SCORE = 0.4  # 顔検出器(det_score)の採用しきい値の既定。param min_det_score で実行中変更可
# 新規ユーザーとして辞書登録する最低特徴数（この数を超えたら dictionary_renew で登録判定）。
MIN_FEATURES_FOR_NEW_USER = 10

# 顔辞書(face_models)。登録先・追跡ノード・APIの参照先と共通化する。
DIRECTORY = str(get_face_models_dir())

class PeopleRecognitionNode(ImagePreviewNode):

    def __init__(self):
        super().__init__('people_recognition_node')

        self.recognition_results = {}
        self.feature_num = 1
        self.last_renew_time = time.time()

        self.all_features = {}
        self.all_images = {}
        self.last_recognition_history = None  # 最後に受信したrecognition_history

        # 顔登録データ(.npy/.jpg/pca_model)をディスクへ永続化するか。
        # デバッグ窓表示用の is_debug_mode とは独立させる（表示だけしたい/保存だけしたいを分離）。
        # メモリ辞書とディスクを同期するため既定 true。実行中に切替可:
        #   ros2 param set /people_recognition_node save_registration false
        save_registration_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_BOOL,
            description='true のとき新規/更新した顔特徴・画像・PCAモデルをディスクへ保存する。')
        self.declare_parameter('save_registration', True, save_registration_descriptor)
        self.save_registration: bool = \
            self.get_parameter('save_registration').get_parameter_value().bool_value

        # 顔検出器(det_score)の採用しきい値。低いほど小さい/遠い顔も拾うが誤検出は増える
        # （誤検出は照合(COSINE_THRESHOLD)で unknown に落ちるため誤命名リスクは低い）。
        min_det_score_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,
            description='顔検出器の信頼度(det_score)の採用しきい値。これ未満の顔は照合対象外。')
        self.declare_parameter('min_det_score', MIN_DET_SCORE, min_det_score_descriptor)
        self.min_det_score: float = \
            self.get_parameter('min_det_score').get_parameter_value().double_value

        # 基底クラスの _on_set_parameters は上書きせず、専用コールバックを追加登録する。
        self.add_on_set_parameters_callback(self._on_set_parameters_people_recognition)
        self.get_logger().info('SaveRegistration : ' + str(self.save_registration))
        self.get_logger().info('MinDetScore : ' + str(self.min_det_score))

        # QoS Settings
        shigure_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # サブスクライバーとパブリッシャーの設定
        self.image_sub = message_filters.Subscriber(self, CompressedImage, '/rs/color/compressed', qos_profile=shigure_qos)
        self.sync = message_filters.TimeSynchronizer([self.image_sub], 10)
        self.sync.registerCallback(self.callback)

        self.recognition_history_sub = self.create_subscription(RecognitionHistory, '/shigure/recognition_history', self.callback_recognition_history, 10)
        self.profile_feature_add_sub = self.create_subscription(
            ProfileFeatureAdd, '/profile_feature_add', self.callback_profile_feature_add, 10
        )
        
        self.publisher = self.create_publisher(FaceRecognitionResult, '/face_recognition/results', 10)
        self.update_pub = self.create_publisher(DictionaryUpdate, '/dictionary_update', 10)
        # 描画ノード (node_feature_plot) 用に特徴ベクトルを配信
        self.feature_pub = self.create_publisher(FeatureInfo, '/feature_info', 10)

        # モデルロード #############################################################
        self.insightface = ProfileInsightFace()
        if not self.insightface.available:
            raise RuntimeError("insightface is not installed")

        # 特徴を読み込む
        self.dictionary = {}
        user_dirs = glob.glob(os.path.join(DIRECTORY, "user_*"))
        for user_dir in user_dirs:
            user_id = os.path.basename(user_dir)
            features = []
            for file in glob.glob(os.path.join(user_dir, "*.npy")):
                feature = np.load(file)
                features.append(np.squeeze(feature)) # 事前に形を整える
            if features:
                self.dictionary[user_id] = features

        self.profile_dictionary = {}
        for user_dir in user_dirs:
            user_id = os.path.basename(user_dir)
            profile_dir = os.path.join(user_dir, 'profile')
            features = []
            if os.path.isdir(profile_dir):
                for file in glob.glob(os.path.join(profile_dir, '*.npy')):
                    feature = np.load(file)
                    features.append(np.squeeze(feature).astype('float32'))
            if features:
                self.profile_dictionary[user_id] = features

        # people_idとuser_idの対応付け用辞書
        self.user_id_map = {}

    def _on_set_parameters_people_recognition(self, params: List[Parameter]) -> SetParametersResult:
        """save_registration の実行時変更を反映する（他パラメータは基底コールバックが処理）。"""
        for param in params:
            if param.name == 'save_registration':
                self.save_registration = param.value
                self.get_logger().info('SaveRegistration : ' + str(self.save_registration))
            elif param.name == 'min_det_score':
                self.min_det_score = param.value
                self.get_logger().info('MinDetScore : ' + str(self.min_det_score))
        return SetParametersResult(successful=True)

    def rebuild_pca_model_on_disk(self) -> None:
        """Rebuild pca_model.pkl from face_models (after new user saved to disk)."""
        if not self.save_registration:
            return
        result = build_pca_model(Path(DIRECTORY), Path(DIRECTORY) / 'pca_model.pkl')
        if result.success:
            self.get_logger().info(f'[PCA] {result.message}')
        else:
            self.get_logger().warn(f'[PCA] rebuild skipped: {result.message}')

    def _ensure_profile_user_entry(self, user_id):
        if user_id not in self.profile_dictionary:
            self.profile_dictionary[user_id] = []

    def add_profile_feature(self, user_name, embedding, face_image=None):
        """横顔特徴を profile_dictionary とディスクに追加（照合辞書の正）。"""
        if not user_name or not user_name.startswith('user'):
            return

        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        self._ensure_profile_user_entry(user_name)
        self.profile_dictionary[user_name].append(embedding)
        self.get_logger().info(
            f'Added profile feature: user_id={user_name}, '
            f'total={len(self.profile_dictionary[user_name])}'
        )

        if not self.save_registration:
            return

        profile_dir = os.path.join(DIRECTORY, user_name, 'profile')
        os.makedirs(profile_dir, exist_ok=True)
        existing = glob.glob(os.path.join(profile_dir, '*.npy'))
        idx = len(existing) + 1
        feature_path = os.path.join(profile_dir, f'{user_name}_profile_{idx}.npy')
        np.save(feature_path, embedding)
        self.get_logger().info(f'Saved profile feature: {feature_path}')

        if face_image is not None:
            image_path = os.path.join(profile_dir, f'{user_name}_profile_{idx}.jpg')
            cv2.imwrite(image_path, face_image)
            self.get_logger().info(f'Saved profile image: {image_path}')

    def callback_profile_feature_add(self, msg: ProfileFeatureAdd):
        face_image = None
        if msg.face_image.data:
            try:
                face_image = self.bridge.compressed_imgmsg_to_cv2(msg.face_image)
            except Exception as exc:
                self.get_logger().warn(f'Failed to decode profile face image: {exc}')
        self.add_profile_feature(msg.user_id, msg.embedding, face_image)

    # 特徴ベクトルを正規化
    def normalize_features(self, features):
        features = np.array(features).astype('float32')
        faiss.normalize_L2(features)  # Faissのnormalize_L2で正規化
        return features
    
    # Faissインデックス作成
    def create_faiss_index(self, features):
        # 特徴ベクトルを正規化
        features = self.normalize_features(features)
        
        # FaissのIndexFlatIP（内積）を使用
        index = faiss.IndexFlatIP(features.shape[1])  # IPは内積（コサイン類似度に対応）
        
        # 特徴ベクトルをインデックスに追加
        index.add(features)
        
        return index

    def matching(self, feature1, dictionary, threshold=COSINE_THRESHOLD):
        """Faissを使ってクエリ特徴と最も類似するユーザーを探す"""
        
        # 辞書内のすべての特徴ベクトルを取り出す
        features_list = []
        user_ids = []
        query_dim = np.asarray(feature1, dtype=np.float32).reshape(-1).shape[0]
        skipped_dim = 0

        for user_id, features in dictionary.items():
            for feature in features:
                feature_vec = np.squeeze(np.asarray(feature, dtype=np.float32))
                if feature_vec.shape[0] != query_dim:
                    skipped_dim += 1
                    continue
                features_list.append(feature_vec)
                user_ids.append(user_id)  # ユーザーIDを対応付けて追加

        if skipped_dim:
            self.get_logger().warn(
                f'Skipped {skipped_dim} dictionary feature(s) with dim != {query_dim}'
            )

        if not features_list:
            return "unknown", 0.0

        # Faissインデックス作成（正規化された特徴ベクトルを使う）
        index = self.create_faiss_index(features_list)
        
        # クエリ特徴ベクトルを用意
        query_feature = np.array(feature1).astype('float32').reshape(1, -1)  # クエリ特徴ベクトルの整形
        faiss.normalize_L2(query_feature)

        distances, indices = index.search(query_feature, 1)
        best_score = distances[0][0]
        best_index = indices[0][0]

        if best_score > threshold:
            best_user_id = user_ids[best_index]
            return best_user_id, float(best_score)
        else:
            return "unknown", 0.0

    def faiss_fallback_user(
        self,
        features: np.ndarray,
        people_id: str = '',
        face_id: str = '',
        threshold: float = COSINE_THRESHOLD,
        min_vote_ratio: float = FAISS_FALLBACK_MIN_VOTE_RATIO,
    ) -> str:
        """LP が unknown のとき、辞書を Faiss で再照合して既存ユーザーを探す。"""
        pid = people_id or '?'
        fid = face_id or '?'
        if not self.dictionary:
            return "unknown"

        features_list = []
        user_ids = []
        query_dim = np.asarray(features[0], dtype=np.float32).reshape(-1).shape[0]
        skipped_dim = 0

        for user_id, dict_features in self.dictionary.items():
            for feature in dict_features:
                feature_vec = np.squeeze(np.asarray(feature, dtype=np.float32))
                if feature_vec.shape[0] != query_dim:
                    skipped_dim += 1
                    continue
                features_list.append(feature_vec)
                user_ids.append(user_id)

        if skipped_dim:
            self.get_logger().warn(
                f'[Faiss] skipped {skipped_dim} dictionary feature(s) with dim != {query_dim}'
            )
        if not features_list:
            return "unknown"

        index = self.create_faiss_index(features_list)
        query_features = self.normalize_features(np.asarray(features, dtype=np.float32))
        distances, indices = index.search(query_features, 1)

        votes: Counter = Counter()
        n_features = len(query_features)
        for i in range(n_features):
            best_score = float(distances[i][0])
            best_index = int(indices[i][0])
            if best_score > threshold:
                votes[user_ids[best_index]] += 1

        if not votes:
            self.get_logger().info(
                f'[Faiss] people_id={pid} face_id={fid} LP=unknown, no match above {threshold:g}'
            )
            return "unknown"

        best_user, vote_count = votes.most_common(1)[0]
        if vote_count / n_features < min_vote_ratio:
            self.get_logger().info(
                f'[Faiss] people_id={pid} face_id={fid} LP=unknown, weak consensus '
                f'{best_user}={vote_count}/{n_features} (need >={min_vote_ratio:.0%})'
            )
            return "unknown"

        vote_parts = ', '.join(f'{uid}={cnt}' for uid, cnt in votes.most_common(5))
        self.get_logger().info(
            f'[Faiss] people_id={pid} face_id={fid} LP=unknown -> {best_user} '
            f'votes={vote_count}/{n_features} ({vote_parts})'
        )
        return best_user
        
    def generate_new_user_name(self, dictionary):
        current_users = [key for key in dictionary.keys() if key.startswith("user_")]
        
        # 現在のユーザー数をカウント
        user_count = len(current_users)

        # 新しいユーザー名を生成
        return f"user_new{user_count + 1}"

    def callback(self, color_img_src: CompressedImage):
        self.frame_count_up()

        color_img: np.ndarray = self.bridge.compressed_imgmsg_to_cv2(color_img_src)
        cap_height, cap_width = color_img.shape[:2]

        # 検出実施 #############################################################
        detected_faces = self.insightface.detect_faces(color_img)

        # Create header with timestamp
        header = Header()
        header.stamp = self.get_clock().now().to_msg()  # Current time
        header.frame_id = "room_camera1"  # Frame ID        
        
        self.recognition_results = FaceRecognitionResult()
        self.recognition_results.header = header

        for iface_face in detected_faces:

            # 顔の四角形の位置を取得
            x, y, w, h = iface_face.bbox

            # 顔の座標が画像範囲外にある場合、スキップ
            if x < 20 or y < 20 or x + w > cap_width -20 or y + h > cap_height -20:
                continue  # 画面外の顔を無視

            # 検出信頼度が低い顔は照合・辞書蓄積の対象外（param min_det_score で調整可）
            if iface_face.det_score < self.min_det_score:
                continue

            face_info = FaceInfo()
            face_info.id, face_info.score, face_info.box, face_info.embedding = self.identify_face(iface_face, color_img)
            face_info.feature_num = self.feature_num
            self.recognition_results.faces.append(face_info)
            self.feature_num = self.feature_num + 1

        self.publisher.publish(self.recognition_results)

        # 10秒ごとに辞書更新を実行
        if time.time() - self.last_renew_time > 2:
            if self.last_recognition_history is not None:  # 最後に受信したデータが存在する場合のみ更新
                self.dictionary_renew(self.last_recognition_history)
                self.last_recognition_history = None  # 更新後にリセット
            self.last_renew_time = time.time()

    def callback_recognition_history(self, msg: RecognitionHistory):
        # recognition_history を受け取るが、辞書更新時まで保持しておく
        self.last_recognition_history = msg  # 最後に受信したデータを保存
    
    def identify_face(self, iface_face: InsightFaceResult, image):
        embedding = iface_face.embedding.copy()
        box = [int(v) for v in iface_face.bbox]
        is_frontal = iface_face.is_frontal

        if is_frontal:
            user_id, score = self.matching(embedding, self.dictionary, COSINE_THRESHOLD)
            face_id = user_id

            # 特徴とfeature_numをall_featuresに追加
            self.all_features[self.feature_num] = {
                "feature": embedding,
                "user_id": user_id,
                "score": score
            }

            x, y, w, h = iface_face.bbox
            self.all_images[self.feature_num] = image[y:y + h, x:x + w].copy()
        else:
            user_id, score = self.matching(embedding, self.profile_dictionary, PROFILE_COSINE_THRESHOLD)
            face_id = f"{user_id}@profile" if user_id != "unknown" else "unknown@profile"

        # 描画/PCAプロット用に特徴を配信する（正面・非正面を問わず全ての検出顔）。
        # unlabeled プロットのリアルタイム表示は feature_info を元にするため、
        # 非正面(@profile)顔でも配信して現フレームの点を出せるようにする。
        feat_msg = FeatureInfo()
        feat_msg.feature_num = int(self.feature_num)
        feat_msg.feature = embedding.astype(np.float32).tolist()
        self.feature_pub.publish(feat_msg)

        return face_id, score, box, embedding.astype(np.float32).tolist()

    @staticmethod
    def max_disk_feature_index(user_dir: str, user_name: str) -> int:
        """ディスク上の正面特徴ファイルから最大連番を返す（無ければ0）."""
        prefix = f'{user_name}_'
        max_idx = 0
        for path in glob.glob(os.path.join(user_dir, f'{prefix}*.npy')):
            stem = os.path.splitext(os.path.basename(path))[0]
            suffix = stem[len(prefix):]
            if suffix.isdigit():
                max_idx = max(max_idx, int(suffix))
        return max_idx

    def save_features_for_user(
        self,
        user_name: str,
        features_num: list,
    ) -> None:
        """Persist feature_num list to dictionary memory and optionally disk."""
        # 辞書にユーザーを登録
        if user_name not in self.dictionary:
            self.dictionary[user_name] = []
        self._ensure_profile_user_entry(user_name)

        user_dir = None
        disk_idx = 1
        if self.save_registration:
            # 新しいディレクトリを作成
            user_dir = os.path.join(DIRECTORY, user_name)
            os.makedirs(user_dir, exist_ok=True)
            # 再起動後も既存ファイルの最大連番の続きから保存する
            # （セッション内 feature_num をファイル名に使うと小さい番号で上書きされるため）
            disk_idx = self.max_disk_feature_index(user_dir, user_name) + 1

        saved = 0
        # 特徴ベクトルを保存
        for num in features_num:
            if num not in self.all_features:
                continue
            feature = self.all_features[num]["feature"]
            self.dictionary[user_name].append(feature)
            if self.save_registration and user_dir is not None:
                # 保存処理（ディスク連番はユーザー単位で単調増加）
                feature_path = os.path.join(user_dir, f"{user_name}_{disk_idx}.npy")
                np.save(feature_path, feature)
                image = self.all_images.get(num)
                if image is not None:
                    # 顔画像保存
                    image_path = os.path.join(user_dir, f"{user_name}_{disk_idx}.jpg")
                    if isinstance(image, np.ndarray):
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        Image.fromarray(image).save(image_path, format="JPEG")
                    else:
                        image.save(image_path, format="JPEG")
                disk_idx += 1
            # all_features から削除
            del self.all_features[num]
            if num in self.all_images:
                del self.all_images[num]
            saved += 1
        if saved:
            print(
                f"Saved {saved} feature(s) to {user_name} "
                f"(total={len(self.dictionary[user_name])})"
            )

    def dictionary_renew(self, msg: RecognitionHistory):
        """
        10秒に1回呼び出される想定の関数。
        recognition_history.users に格納されている各ユーザーの features_num が
        MIN_FEATURES_FOR_NEW_USER 件を超えていたら、
        特徴を辞書に統合 (既存 or 新規) して、その後 all_features から削除する。
        ラベル伝播法を用いて未ラベルデータにラベルを付け、信頼度が低いものを新規ユーザーとして追加します。
        face_id ごとに評価し、同一 people_id 内の別 face_id とは混ぜない。
        """
        print("辞書の更新を開始します")
        recognition_history = msg

        # 特徴数とスコアの計算（face_id バケット単位）
        for user in recognition_history.users:
            people_id = user.people_id

            for face in user.face_info:
                # 横顔は数えない
                if face.id.endswith('@profile'):
                    continue
                # 一定数の特徴がないと辞書に登録しない
                if face.total_features <= MIN_FEATURES_FOR_NEW_USER:
                    continue

                face_id = face.id
                features_num = [int(fn) for fn in face.features_num]
                # features_num の各番号に対応する特徴が存在するかをチェック
                features_list = [
                    self.all_features[num]["feature"]
                    for num in features_num
                    if num in self.all_features
                ]
                # features_list が空でないことを確認してから np.stack を呼び出す
                if not features_list:
                    print(
                        f"{people_id} face_id={face_id}: "
                        "all_features に該当特徴がありませんでした"
                    )
                    continue

                features = np.stack(features_list)
                best_user = self.label_propagation(
                    features, people_id=people_id, face_id=face_id
                )
                if not best_user:
                    continue

                faiss_rescued = False
                if best_user.startswith("unknown"):
                    faiss_user = self.faiss_fallback_user(
                        features, people_id=people_id, face_id=face_id
                    )
                    if not faiss_user.startswith("unknown"):
                        best_user = faiss_user
                        faiss_rescued = True

                print(
                    f"Selected user: {best_user} "
                    f"(people_id={people_id}, face_id={face_id}, n={len(features_num)}, "
                    f"faiss_rescued={faiss_rescued})"
                )

                # 新しいユーザー登録の条件を満たす場合
                if best_user.startswith("unknown"):
                    if face_id != "unknown":
                        self.get_logger().info(
                            f'[dict_renew] skip new-user: people_id={people_id} '
                            f'face_id={face_id} LP=unknown (only unknown bucket registers)'
                        )
                        continue

                    new_user_name = self.generate_new_user_name(self.dictionary)
                    print(f"New user detected: {new_user_name}")
                    self.save_features_for_user(new_user_name, features_num)
                    self.rebuild_pca_model_on_disk()
                    # 辞書内容をトピックで配信
                    self.update_dictionary(people_id, new_user_name)
                    continue

                # 既存の user の場合、face_id が LP 結果と一致するときのみ特徴を辞書に追加
                # Faiss で unknown から救済した場合は face_id=unknown でも既存ユーザーとして扱う
                if face_id != best_user and not faiss_rescued:
                    self.get_logger().info(
                        f'[dict_renew] skip existing-user: people_id={people_id} '
                        f'face_id={face_id} LP={best_user} (face_id must match)'
                    )
                    continue

                average_score = face.accumulate_score / face.total_features
                print("average_score: ", average_score)
                if face.accumulate_score > 3 and average_score > 0.6:
                    self.save_features_for_user(best_user, features_num)
                    # 辞書内容をトピックで配信
                    self.update_dictionary(people_id, best_user)
                else:
                    self.update_dictionary(people_id, "none")

    def label_propagation(
        self, unlabeled_features, people_id: str = '', face_id: str = ''
    ) -> str:
        labeled_features = []    # 全ラベル付き特徴量
        labeled_labels = []      # ラベル(数値)

        # ---------- 1) ラベル付きデータの読み込み ----------
        # self.dictionary からユーザーIDと特徴を抽出
        user_ids = sorted(self.dictionary.keys())  # すべてのユーザーIDを取得
        # ユーザーIDから数値ラベルへのマッピングを作る
        # 例: user_yoki -> 0, user_john -> 1, ...
        user_id_to_label = {uid: idx for idx, uid in enumerate(sorted(user_ids))}
        # あとで表示時に数字→名前へ戻すマップ
        label_to_user_id = {v: k for k, v in user_id_to_label.items()}
        # 実際の特徴量とラベルを集める
        for user_id in user_ids:
            label_value = user_id_to_label[user_id]
            features = self.dictionary[user_id]
            for feature in features:
                labeled_features.append(feature)  # 各ユーザーの特徴ベクトルを追加
                labeled_labels.append(label_value)  # ラベルを追加

        # 辞書が空のときは LabelPropagation せず新規ユーザー扱いにする
        if not labeled_features:
            self.get_logger().warn(
                f'[LP] people_id={people_id} face_id={face_id} '
                f'dictionary empty -> unknown (new user)'
            )
            return "unknown"

        # ---------- 3) LabelPropagation 用のデータ準備 ----------
        #  labeled_features + unlabeled_features を連結し、ラベル配列を作る
        labeled_features = np.vstack(labeled_features).astype(np.float32)
        all_features = np.vstack([labeled_features, unlabeled_features])
        num_labeled = len(labeled_features)
        num_unlabeled = len(unlabeled_features)
        # 既知ラベル: labeled_labels, 未知ラベル: -1
        all_labels = np.array(labeled_labels + [-1]*num_unlabeled, dtype=int)

        # ---------- 4) LabelPropagation の実行 ----------
        # n_neighbors はサンプル数を超えられない（超えると sklearn が ValueError）。
        # 辞書が小さいうちでも落ちないよう、サンプル数-1 で上限クリップする。
        n_samples = len(all_features)
        n_neighbors = max(1, min(100, n_samples - 1))
        label_prop = LabelPropagation(kernel='knn', n_neighbors=n_neighbors)
        label_prop.fit(all_features, all_labels)
        final_labels = label_prop.transduction_
        # label_distributions_ (各サンプルが各クラスに属する確率分布)
        label_probs = label_prop.label_distributions_
                
        # ---------- 5) 「確信度が低い」サンプルを新規クラスに再割り当て ----------
        threshold = LP_CONFIDENCE_THRESHOLD
        existing_max_label = final_labels.max()  # 既存の最大ラベルID
        new_label_id = existing_max_label + 1    # 新規クラスのラベルID
        unlabeled_max_probs: list[float] = []
        low_confidence_count = 0
        for i, probs in enumerate(label_probs):
            if all_labels[i] == -1:  # もともと未ラベル
                max_prob = float(np.max(probs))
                unlabeled_max_probs.append(max_prob)
                if max_prob < threshold:
                    low_confidence_count += 1
                    final_labels[i] = new_label_id

        # 伝播後のラベルをカウントして最も多いラベルを決定
        # ここで最も頻繁に現れるラベルを返します
        label_counts = Counter(final_labels[num_labeled:])  # ラベルなしデータに対応する部分
        most_common_label, most_common_count = label_counts.most_common(1)[0]  # 最も頻繁に現れるラベルとその数

        # 辞書から指定したキー（ここでは most_common_label）に対応する値を取得します。
        # もしそのキーが辞書に存在しない場合、2番目の引数（ここでは "unknown"）を返します。
        most_common_user_id = label_to_user_id.get(most_common_label, "unknown")

        self.log_label_propagation_confidence(
            people_id=people_id,
            face_id=face_id,
            threshold=threshold,
            num_labeled=num_labeled,
            unlabeled_max_probs=unlabeled_max_probs,
            low_confidence_count=low_confidence_count,
            new_label_id=new_label_id,
            label_counts=label_counts,
            label_to_user_id=label_to_user_id,
            most_common_label=most_common_label,
            most_common_count=most_common_count,
            most_common_user_id=most_common_user_id,
        )
        return most_common_user_id

    def log_label_propagation_confidence(
        self,
        *,
        people_id: str,
        face_id: str,
        threshold: float,
        num_labeled: int,
        unlabeled_max_probs: list,
        low_confidence_count: int,
        new_label_id: int,
        label_counts: Counter,
        label_to_user_id: dict,
        most_common_label: int,
        most_common_count: int,
        most_common_user_id: str,
    ) -> None:
        pid = people_id or '?'
        fid = face_id or '?'
        n = len(unlabeled_max_probs)
        if n == 0:
            self.get_logger().info(f'[LP] people_id={pid} face_id={fid} no unlabeled samples')
            return

        arr = np.asarray(unlabeled_max_probs, dtype=np.float32)
        self.get_logger().info(
            f'[LP] people_id={pid} face_id={fid} threshold={threshold:g} '
            f'unlabeled={n} dictionary_refs={num_labeled} '
            f'max_prob min={arr.min():.3f} mean={arr.mean():.3f} max={arr.max():.3f} '
            f'below_threshold={low_confidence_count}/{n} '
            f'-> result={most_common_user_id} '
            f'(votes={most_common_count}/{n}, label_id={most_common_label})'
        )

        # 確信度ヒストグラム（0.1 刻み）
        buckets = {f'{lo:.1f}-{hi:.1f}': 0 for lo, hi in zip(np.arange(0, 1, 0.1), np.arange(0.1, 1.1, 0.1))}
        for p in unlabeled_max_probs:
            idx = min(int(p * 10), 9)
            lo = idx / 10
            hi = lo + 0.1
            buckets[f'{lo:.1f}-{hi:.1f}'] += 1
        hist = ' '.join(f'{k}:{v}' for k, v in buckets.items() if v > 0)
        self.get_logger().info(f'[LP] people_id={pid} face_id={fid} max_prob histogram: {hist}')

        # 投票内訳（上位5）
        vote_parts = []
        for label_id, count in label_counts.most_common(5):
            uid = label_to_user_id.get(label_id, f'new_label_{label_id}')
            vote_parts.append(f'{uid}={count}')
        self.get_logger().info(f'[LP] people_id={pid} face_id={fid} label votes: {", ".join(vote_parts)}')

        if low_confidence_count > 0:
            self.get_logger().info(
                f'[LP] people_id={pid} face_id={fid} {low_confidence_count} sample(s) '
                f'max_prob<{threshold:g} -> reassigned to new_label_id={new_label_id} (maps to unknown)'
            )
    

    def update_dictionary(self, people_id: str, user_name: str):
        # user_name が文字列か確認
        if not isinstance(user_name, str):
            print(f"Invalid user_name type: {type(user_name)}. Expected 'str'.")
            return
        
        if user_name == "none":
            print("新しいユーザーは登録されませんでした")
            
        # Create header with timestamp
        header = Header()
        header.stamp = self.get_clock().now().to_msg()  # Current time
        header.frame_id = "room_camera1"  # Frame ID

        update_msg = DictionaryUpdate()
        update_msg.header = header
        update_msg.people_id = people_id
        update_msg.name = user_name
        self.update_pub.publish(update_msg)

def main(args=None):
    rclpy.init(args=args)

    people_recognition_node = PeopleRecognitionNode()

    try:
        rclpy.spin(people_recognition_node)

    except KeyboardInterrupt:
        pass

    finally:
        # 終了処理
        people_recognition_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
