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
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from PIL import Image

from shigure_core.nodes.node_image_preview import ImagePreviewNode
from shigure_core.nodes.face_analyzer import (
    DetectedFace,
    FaceAnalyzer,
    FRONTAL_PITCH_THRESHOLD,
    FRONTAL_ROLL_THRESHOLD,
    FRONTAL_YAW_THRESHOLD,
)
from shigure_core.util.face_models_dir import (
    face_models_signature,
    get_face_models_dir,
    sync_all_user_file_prefixes,
)
from shigure_core.util.pca_model import build_pca_model

from sklearn.semi_supervised import LabelPropagation
from collections import Counter

from std_msgs.msg import Header
import faiss

from shigure_core.nodes.people_recognition.unknown_enrollment import (
    DBSCAN_MIN_SAMPLES,
    GALLERY_KNN_K,
    GALLERY_RESCUE_MIN_VOTE_RATIO,
    dbscan_cluster_features,
    find_gallery_user_by_knn,
    select_dense_clusters,
)

# COSINE_THRESHOLD = 0.363
# AdaFace (L2 正規化 512次元) の初期しきい値。スコア分布が ArcFace と違うので
# 実データで取り直すこと。
COSINE_THRESHOLD = 0.4
PROFILE_COSINE_THRESHOLD = 0.4
NORML2_THRESHOLD = 1.128
LP_CONFIDENCE_THRESHOLD = 0.7
MIN_DET_SCORE = 0.4  # 顔検出器(det_score)の採用しきい値の既定。param min_det_score で実行中変更可
# 新規ユーザー候補として未ラベル池へ送る最低特徴数（即 new user にはしない）。
MIN_FEATURES_FOR_NEW_USER = 10
# 1ユーザーあたりの正面特徴・画像の上限。到達後は dictionary_renew で追加保存しない。
MAX_FEATURES_PER_USER = 100
# 未ラベル池の特徴数上限（古いものから破棄してメモリ肥大を防ぐ）。
MAX_UNLABELED_POOL_SIZE = 500

# 顔辞書(face_models)。登録先・追跡ノード・APIの参照先と共通化する。
DIRECTORY = str(get_face_models_dir())
REGISTRATION_MEMO_PATH = os.path.join(DIRECTORY, 'registration_memo.jsonl')


class PeopleRecognitionNode(ImagePreviewNode):

    def __init__(self):
        super().__init__('people_recognition_node')

        self.recognition_results = {}
        self.feature_num = 1
        self.last_renew_time = time.time()

        self.all_features = {}
        self.all_images = {}
        self.last_recognition_history = None  # 最後に受信したrecognition_history
        # 保存済み特徴・画像のハッシュ集合（クロスユーザー重複登録防止）
        self.saved_feature_hashes = set()
        self.saved_image_hashes = set()
        # unknown バケットの未ラベル池。
        # {feature_num: {'people_id', 'face_id', 'added_at'}}
        # 実体ベクトルは all_features / all_images 側に残し、DBSCAN 後に保存 or 破棄する。
        self.unlabeled_pool = {}

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
        self.face_analyzer = FaceAnalyzer()
        if not self.face_analyzer.available:
            raise RuntimeError("insightface or onnxruntime is not installed")

        # 特徴を読み込む（手動でディスクの user_* を消した場合は照合前に再同期する）
        self.dictionary = {}
        self.profile_dictionary = {}
        # ディスク由来のユーザー集合。ディレクトリ削除時にメモリから落とす判定に使う。
        self._disk_backed_users = set()
        self._face_models_signature = None
        self.reload_dictionary_from_disk(force=True)

        # people_idとuser_idの対応付け用辞書
        self.user_id_map = {}

    @staticmethod
    def _load_dictionaries_from_disk():
        """face_models から正面・横顔辞書を読み込む."""
        dictionary = {}
        profile_dictionary = {}
        user_dirs = glob.glob(os.path.join(DIRECTORY, 'user_*'))
        for user_dir in user_dirs:
            user_id = os.path.basename(user_dir)
            features = []
            for file in glob.glob(os.path.join(user_dir, '*.npy')):
                feature = np.load(file)
                features.append(np.squeeze(feature))
            if features:
                dictionary[user_id] = features

            profile_dir = os.path.join(user_dir, 'profile')
            profile_features = []
            if os.path.isdir(profile_dir):
                for file in glob.glob(os.path.join(profile_dir, '*.npy')):
                    feature = np.load(file)
                    profile_features.append(np.squeeze(feature).astype('float32'))
            if profile_features:
                profile_dictionary[user_id] = profile_features
        return dictionary, profile_dictionary

    def reload_dictionary_from_disk(self, force: bool = False) -> bool:
        """ディスクの face_models をメモリ辞書へ反映する.

        - ディスク上の user_* は内容をディスク基準で置き換える
        - ディスクから消えたユーザーはメモリからも削除する
        - save_registration=false などでディスクに無いメモリ専用ユーザーは残す
        - ディレクトリ名とファイル接頭辞の不一致（手動リネーム後の user_newN_*）を揃える
        """
        signature = face_models_signature(DIRECTORY)
        if not force and signature == self._face_models_signature:
            return False

        # user_new1 → user_aono のようにディレクトリだけ変えた場合、中の
        # user_new1_*.npy/.jpg もディレクトリ名に合わせてリネームする。
        renamed = sync_all_user_file_prefixes(DIRECTORY)
        if renamed:
            self.get_logger().info(
                f'[dict_sync] renamed {renamed} file(s) to match user dir names'
            )
            signature = face_models_signature(DIRECTORY)

        disk_dictionary, disk_profile = self._load_dictionaries_from_disk()
        disk_users = set(disk_dictionary.keys()) | set(disk_profile.keys())
        previous_disk_users = set(self._disk_backed_users)
        removed_users = previous_disk_users - disk_users

        # ディスク由来だったが今はディレクトリが無いユーザーをメモリから除去する
        for user_id in removed_users:
            self.dictionary.pop(user_id, None)
            self.profile_dictionary.pop(user_id, None)

        # ディスク上のユーザーはディスク内容で上書き（追加含む）
        for user_id, features in disk_dictionary.items():
            self.dictionary[user_id] = features
        for user_id, features in disk_profile.items():
            self.profile_dictionary[user_id] = features
        # 正面が無く横顔だけ残っているケース向けに profile も掃除
        for user_id in list(self.profile_dictionary.keys()):
            if (
                user_id in previous_disk_users
                and user_id not in disk_profile
                and user_id not in disk_dictionary
            ):
                self.profile_dictionary.pop(user_id, None)

        self._disk_backed_users = disk_users
        self._face_models_signature = signature

        if removed_users:
            self.get_logger().info(
                f'[dict_sync] removed from memory: {sorted(removed_users)}'
            )
        self.get_logger().info(
            f'[dict_sync] disk users={sorted(self._disk_backed_users)} '
            f'memory users={sorted(self.dictionary.keys())}'
        )
        self.rebuild_saved_hashes()
        return True

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
        self._disk_backed_users.add(user_name)
        self._face_models_signature = face_models_signature(DIRECTORY)

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
        # 手動でディスク上の user_* を削除/変更した場合にメモリへ反映する
        self.reload_dictionary_from_disk(force=False)

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

    def generate_new_user_name(self, dictionary):
        """未使用の user_newN 名を返す（最大番号+1。欠番があっても衝突しない）."""
        max_idx = 0
        for key in dictionary.keys():
            if not key.startswith('user_new'):
                continue
            suffix = key[len('user_new'):]
            if suffix.isdigit():
                max_idx = max(max_idx, int(suffix))
        # 未ラベル池処理中に連続採番するため、メモリ上の最大+1を使う
        return f'user_new{max_idx + 1}'

    def gallery_rescue_decide(
        self,
        features: np.ndarray,
        people_id: str = '',
        face_id: str = '',
        threshold: float = COSINE_THRESHOLD,
        k: int = GALLERY_KNN_K,
        min_vote_ratio: float = GALLERY_RESCUE_MIN_VOTE_RATIO,
    ) -> str:
        """ギャラリー救済判定。閾値超え票があれば既存 user_id を返す."""
        pid = people_id or '?'
        fid = face_id or '?'
        user_id, info = find_gallery_user_by_knn(
            features,
            self.dictionary,
            threshold=threshold,
            k=k,
            min_vote_ratio=min_vote_ratio,
        )
        n_queries = int(info.get('n_queries', 0))
        vote_ratio = float(info.get('vote_ratio', 0.0))
        if user_id is None:
            self.get_logger().info(
                f'[GalleryRescue] people_id={pid} face_id={fid} reject '
                f'(best_score={info.get("best_score", 0):.3f}, '
                f'vote_ratio={vote_ratio:.1%} need>={min_vote_ratio:.0%}, '
                f'votes={info.get("votes", {})}, n={n_queries})'
            )
            return 'unknown'

        self.get_logger().info(
            f'[GalleryRescue] people_id={pid} face_id={fid} -> {user_id} '
            f'votes={info.get("vote_count", 0)}/{n_queries} '
            f'({vote_ratio:.1%} >= {min_vote_ratio:.0%}) '
            f'best_score={info.get("best_score", 0):.3f} '
            f'votes={info.get("votes", {})}'
        )
        return user_id

    @staticmethod
    def is_frontal_feature_entry(entry: dict) -> bool:
        """all_features エントリが正面登録可能か判定する.

        pose 欠損や閾値超えは False。横顔・傾きの新規登録・辞書混入を防ぐ。
        """
        if not entry.get('pose_valid', False):
            return False
        yaw = float(entry.get('yaw', 0.0))
        pitch = float(entry.get('pitch', 0.0))
        roll = float(entry.get('roll', 0.0))
        return FaceAnalyzer.is_frontal(
            yaw,
            pitch,
            roll,
            yaw_threshold=FRONTAL_YAW_THRESHOLD,
            pitch_threshold=FRONTAL_PITCH_THRESHOLD,
            roll_threshold=FRONTAL_ROLL_THRESHOLD,
        )

    def add_features_to_unlabeled_pool(
        self,
        features_num: list,
        people_id: str = '',
        face_id: str = '',
    ) -> int:
        """unknown 特徴を未ラベル池へ追加する（即 new user しない）."""
        added = 0
        skipped_profile = 0
        now = time.time()
        for num in features_num:
            if num not in self.all_features:
                continue
            if num in self.unlabeled_pool:
                continue
            entry = self.all_features[num]
            # 横顔・pose 欠損は新規登録候補にしない
            if not self.is_frontal_feature_entry(entry):
                self.get_logger().info(
                    f'[UnlabeledPool] skip non-frontal: feature_num={num} '
                    f'yaw={entry.get("yaw", "?")} pitch={entry.get("pitch", "?")} '
                    f'roll={entry.get("roll", "?")} '
                    f'pose_valid={entry.get("pose_valid", False)}'
                )
                self.release_pending_feature(num)
                skipped_profile += 1
                continue
            # 既に辞書へ保存済みの完全一致特徴は池に入れない
            feature = entry['feature']
            if self.is_feature_duplicate(feature):
                self.release_pending_feature(num)
                continue
            self.unlabeled_pool[num] = {
                'people_id': people_id,
                'face_id': face_id,
                'added_at': now,
            }
            added += 1

        # 上限超過時は古い順に破棄
        overflow = len(self.unlabeled_pool) - MAX_UNLABELED_POOL_SIZE
        if overflow > 0:
            oldest = sorted(
                self.unlabeled_pool.items(),
                key=lambda item: item[1].get('added_at', 0.0),
            )[:overflow]
            for num, _meta in oldest:
                self.unlabeled_pool.pop(num, None)
                self.release_pending_feature(num)
            self.get_logger().warn(
                f'[UnlabeledPool] dropped {overflow} oldest feature(s) '
                f'(limit={MAX_UNLABELED_POOL_SIZE})'
            )

        if skipped_profile:
            self.get_logger().info(
                f'[UnlabeledPool] skipped {skipped_profile} non-frontal '
                f'feature(s) (people_id={people_id or "?"})'
            )
        if added:
            self.get_logger().info(
                f'[UnlabeledPool] added={added} pool_size={len(self.unlabeled_pool)} '
                f'(people_id={people_id or "?"}, face_id={face_id or "?"})'
            )
        return added

    def process_unlabeled_pool_dbscan(self) -> int:
        """未ラベル池を DBSCAN し、密集クラスタだけを既存マージ or new user する.

        既存マージはクラスタ全体の kNN（票が半数以上）のみ。重心照合は使わない。

        Returns:
            今回保存した特徴数の合計。
        """
        # 実体が消えたエントリを掃除
        stale = [num for num in self.unlabeled_pool if num not in self.all_features]
        for num in stale:
            self.unlabeled_pool.pop(num, None)

        pool_nums = sorted(self.unlabeled_pool.keys())
        min_samples = max(DBSCAN_MIN_SAMPLES, MIN_FEATURES_FOR_NEW_USER)
        if len(pool_nums) < min_samples:
            return 0

        features_list = [self.all_features[num]['feature'] for num in pool_nums]
        features = np.stack(features_list)
        eps = 1.0 - COSINE_THRESHOLD
        labels = dbscan_cluster_features(
            features,
            eps=eps,
            min_samples=min_samples,
        )
        clusters = select_dense_clusters(labels)
        noise_count = int(np.sum(labels < 0))
        self.get_logger().info(
            f'[DBSCAN] pool={len(pool_nums)} clusters={len(clusters)} '
            f'noise={noise_count} eps={eps:g} min_samples={min_samples}'
        )
        if not clusters:
            return 0

        total_saved = 0
        for label, member_idx in clusters:
            member_nums = [pool_nums[i] for i in member_idx.tolist()]
            member_feats = features[member_idx]
            people_ids = sorted({
                self.unlabeled_pool[n].get('people_id', '')
                for n in member_nums
                if n in self.unlabeled_pool
            })
            face_ids = sorted({
                self.unlabeled_pool[n].get('face_id', '')
                for n in member_nums
                if n in self.unlabeled_pool
            })
            people_id = people_ids[0] if len(people_ids) == 1 else ','.join(people_ids)
            face_id = face_ids[0] if len(face_ids) == 1 else ','.join(face_ids)

            # クラスタ全体の kNN（半数以上）だけで既存マージする。重心照合はしない。
            merge_user, knn_info = find_gallery_user_by_knn(
                member_feats,
                self.dictionary,
                threshold=COSINE_THRESHOLD,
                k=GALLERY_KNN_K,
                min_vote_ratio=GALLERY_RESCUE_MIN_VOTE_RATIO,
            )

            if merge_user is not None:
                self.get_logger().info(
                    f'[DBSCAN] cluster={label} n={len(member_nums)} '
                    f'-> merge {merge_user} (best_score={knn_info.get("best_score", 0):.3f})'
                )
                saved = self.save_features_for_user(
                    merge_user,
                    member_nums,
                    route='unlabeled_pool(DBSCAN -> gallery merge) -> '
                          'save_features_for_user',
                    source='unlabeled_pool_dbscan',
                    people_id=people_id,
                    face_id=face_id,
                )
                is_existing_merge = True
            else:
                # ハッシュ一致の最終ガード
                existing_user = self.find_existing_user_for_features(member_feats)
                if existing_user:
                    merge_user = existing_user
                    is_existing_merge = True
                    self.get_logger().info(
                        f'[DBSCAN] cluster={label} n={len(member_nums)} '
                        f'-> hash merge {merge_user}'
                    )
                    saved = self.save_features_for_user(
                        merge_user,
                        member_nums,
                        route='unlabeled_pool(DBSCAN -> hash merge) -> '
                              'save_features_for_user',
                        source='unlabeled_pool_dbscan',
                        people_id=people_id,
                        face_id=face_id,
                    )
                else:
                    is_existing_merge = False
                    new_user_name = self.generate_new_user_name(self.dictionary)
                    self.get_logger().info(
                        f'[DBSCAN] cluster={label} n={len(member_nums)} '
                        f'-> new user {new_user_name}'
                    )
                    print(f'New user detected: {new_user_name}')
                    saved = self.save_features_for_user(
                        new_user_name,
                        member_nums,
                        route='unlabeled_pool(DBSCAN -> new_user) -> '
                              'save_features_for_user',
                        source='unlabeled_pool_dbscan',
                        people_id=people_id,
                        face_id=face_id,
                    )
                    merge_user = new_user_name if saved > 0 else None

            # 池から除去（save 済みは release 済み、未保存分も池からは外す）
            for num in member_nums:
                self.unlabeled_pool.pop(num, None)
                # 保存されなかった残りは破棄して再登録ループを防ぐ
                if num in self.all_features:
                    self.release_pending_feature(num)

            if saved > 0:
                total_saved += saved
                self.rebuild_pca_model_on_disk()

            # 既存マージ: 保存条件を満たしたので saved=0 でも青枠
            # 新規: 実際に作成できたときだけ青枠
            should_confirm = False
            if is_existing_merge and merge_user:
                should_confirm = True
            elif merge_user and saved > 0:
                should_confirm = True
            if should_confirm and people_ids:
                if len(people_ids) == 1 and people_ids[0]:
                    self.update_dictionary(people_ids[0], merge_user)

        if total_saved:
            self.get_logger().info(
                f'[DBSCAN] saved_total={total_saved} remaining_pool={len(self.unlabeled_pool)}'
            )
        return total_saved

    def callback(self, color_img_src: CompressedImage):
        self.frame_count_up()

        color_img: np.ndarray = self.bridge.compressed_imgmsg_to_cv2(color_img_src)
        cap_height, cap_width = color_img.shape[:2]

        # 検出実施 #############################################################
        detected_faces = self.face_analyzer.detect_faces(color_img)

        # Create header with timestamp
        header = Header()
        header.stamp = self.get_clock().now().to_msg()  # Current time
        header.frame_id = "room_camera1"  # Frame ID        
        
        self.recognition_results = FaceRecognitionResult()
        self.recognition_results.header = header

        for detected_face in detected_faces:

            # 顔の四角形の位置を取得
            x, y, w, h = detected_face.bbox

            # 顔の座標が画像範囲外にある場合、スキップ
            if x < 20 or y < 20 or x + w > cap_width -20 or y + h > cap_height -20:
                continue  # 画面外の顔を無視

            # 検出信頼度が低い顔は照合・辞書蓄積の対象外（param min_det_score で調整可）
            if detected_face.det_score < self.min_det_score:
                continue

            face_info = FaceInfo()
            face_info.id, face_info.score, face_info.box, face_info.embedding = self.identify_face(detected_face, color_img)
            face_info.feature_num = self.feature_num
            self.recognition_results.faces.append(face_info)
            self.feature_num = self.feature_num + 1

        self.publisher.publish(self.recognition_results)

        # 数秒ごとに辞書更新 + 未ラベル池の DBSCAN 登録判定
        if time.time() - self.last_renew_time > 2:
            if self.last_recognition_history is not None:  # 最後に受信したデータが存在する場合のみ更新
                self.dictionary_renew(self.last_recognition_history)
                self.last_recognition_history = None  # 更新後にリセット
            # recognition_history が無くても池に溜まっていれば DBSCAN を回す
            self.process_unlabeled_pool_dbscan()
            self.last_renew_time = time.time()

    def callback_recognition_history(self, msg: RecognitionHistory):
        # recognition_history を受け取るが、辞書更新時まで保持しておく
        self.last_recognition_history = msg  # 最後に受信したデータを保存
    
    def identify_face(self, detected_face: DetectedFace, image):
        embedding = detected_face.embedding.copy()
        box = [int(v) for v in detected_face.bbox]
        is_frontal = detected_face.is_frontal

        if is_frontal:
            user_id, score = self.matching(embedding, self.dictionary, COSINE_THRESHOLD)
            face_id = user_id

            # 特徴とfeature_numをall_featuresに追加（角度は保存時に再検証する）
            self.all_features[self.feature_num] = {
                "feature": embedding,
                "user_id": user_id,
                "score": score,
                "yaw": float(detected_face.yaw),
                "pitch": float(detected_face.pitch),
                "roll": float(detected_face.roll),
                "pose_valid": bool(detected_face.pose_valid),
            }

            x, y, w, h = detected_face.bbox
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

    def compute_feature_hash(self, feature) -> str:
        """正規化済み特徴ベクトルの内容ハッシュを返す。"""
        vec = np.asarray(feature, dtype=np.float32).reshape(-1).copy()
        faiss.normalize_L2(vec.reshape(1, -1))
        return hashlib.md5(vec.tobytes()).hexdigest()

    def compute_image_hash(self, image) -> str:
        """顔画像(JPG相当)の内容ハッシュを返す。"""
        if isinstance(image, np.ndarray):
            arr = np.ascontiguousarray(image)
        else:
            arr = np.ascontiguousarray(np.array(image))
        return hashlib.md5(arr.tobytes()).hexdigest()

    def rebuild_saved_feature_hashes(self) -> None:
        """辞書内の全正面特徴から保存済み特徴ハッシュ集合を再構築する。"""
        self.saved_feature_hashes = set()
        for features in self.dictionary.values():
            for feature in features:
                self.saved_feature_hashes.add(self.compute_feature_hash(feature))

    def rebuild_saved_image_hashes(self) -> None:
        """ディスク上の正面顔JPGから保存済み画像ハッシュ集合を再構築する。"""
        self.saved_image_hashes = set()
        for user_dir in glob.glob(os.path.join(DIRECTORY, 'user_*')):
            for image_path in glob.glob(os.path.join(user_dir, '*.jpg')):
                image = cv2.imread(image_path)
                if image is None:
                    continue
                self.saved_image_hashes.add(self.compute_image_hash(image))

    def rebuild_saved_hashes(self) -> None:
        """保存済み特徴・画像ハッシュ集合を辞書/ディスクから再構築する。"""
        self.rebuild_saved_feature_hashes()
        self.rebuild_saved_image_hashes()

    def is_feature_duplicate(self, feature) -> bool:
        """特徴ハッシュが既存辞書と一致するか判定する。"""
        return self.compute_feature_hash(feature) in self.saved_feature_hashes

    def is_image_duplicate(self, image) -> bool:
        """画像内容ハッシュが既に保存済みか判定する。"""
        return self.compute_image_hash(image) in self.saved_image_hashes

    def find_user_for_feature_hash(self, feature_hash: str) -> str | None:
        """特徴ハッシュに一致する既存ユーザーを返す。無ければ None。"""
        for user_id, features in self.dictionary.items():
            for feature in features:
                if self.compute_feature_hash(feature) == feature_hash:
                    return user_id
        return None

    def find_existing_user_for_features(self, features: np.ndarray) -> str | None:
        """保存候補特徴群のうち、ハッシュ一致する特徴があればその user_id を返す。"""
        if not self.dictionary:
            return None
        votes: Counter = Counter()
        for feature in features:
            feature_hash = self.compute_feature_hash(feature)
            if feature_hash not in self.saved_feature_hashes:
                continue
            user_id = self.find_user_for_feature_hash(feature_hash)
            if user_id is not None:
                votes[user_id] += 1
        if not votes:
            return None
        return votes.most_common(1)[0][0]

    def release_pending_feature(self, num: int) -> None:
        """all_features / all_images から未保存の feature_num を破棄する。"""
        self.all_features.pop(num, None)
        self.all_images.pop(num, None)

    def register_saved_hashes(self, feature, image=None) -> None:
        """保存した特徴・画像のハッシュを集合へ登録する。"""
        self.saved_feature_hashes.add(self.compute_feature_hash(feature))
        if image is not None:
            self.saved_image_hashes.add(self.compute_image_hash(image))

    def count_user_features(self, user_name: str) -> int:
        """ユーザーの正面特徴数を返す（メモリとディスクの大きい方）."""
        mem_count = len(self.dictionary.get(user_name, []))
        if not self.save_registration:
            return mem_count
        user_dir = os.path.join(DIRECTORY, user_name)
        if not os.path.isdir(user_dir):
            return mem_count
        disk_count = len(glob.glob(os.path.join(user_dir, f'{user_name}_*.npy')))
        return max(mem_count, disk_count)

    def append_registration_memo(
        self,
        *,
        user_name: str,
        route: str,
        source: str,
        people_id: str,
        face_id: str,
        requested_feature_nums: list,
        saved_items: list,
        skipped_feature_nums: list,
    ) -> None:
        """登録履歴を jsonl へ追記する。route には通過関数の経路を入れる。"""
        memo = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'user_name': user_name,
            'route': route,
            'source': source,
            'people_id': people_id,
            'face_id': face_id,
            'requested_feature_nums': requested_feature_nums,
            'saved_feature_nums': [item['feature_num'] for item in saved_items],
            'skipped_feature_nums': skipped_feature_nums,
            'saved_count': len(saved_items),
            'items': saved_items,
        }
        os.makedirs(os.path.dirname(REGISTRATION_MEMO_PATH), exist_ok=True)
        with open(REGISTRATION_MEMO_PATH, 'a', encoding='utf-8') as memo_file:
            memo_file.write(json.dumps(memo, ensure_ascii=False) + '\n')

    def save_features_for_user(
        self,
        user_name: str,
        features_num: list,
        route: str = 'unknown -> save_features_for_user',
        source: str = 'unknown',
        people_id: str = '',
        face_id: str = '',
    ) -> int:
        """Persist feature_num list to dictionary memory and optionally disk.

        1ユーザーあたり MAX_FEATURES_PER_USER 件を超える正面特徴・画像は追加しない。
        既存辞書と重複する特徴/画像はスキップする。
        pose 欠損・横顔・傾き（yaw/pitch/roll しきい値外）は保存しない。
        """
        # 辞書にユーザーを登録
        if user_name not in self.dictionary:
            self.dictionary[user_name] = []
        self._ensure_profile_user_entry(user_name)

        current_count = self.count_user_features(user_name)
        remaining = MAX_FEATURES_PER_USER - current_count
        if remaining <= 0:
            self.get_logger().info(
                f'[dict_renew] skip save: {user_name} already has '
                f'{current_count} feature(s) (limit={MAX_FEATURES_PER_USER})'
            )
            return 0

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
        skipped = 0
        skipped_profile = 0
        saved_items = []
        skipped_feature_nums = []
        # 特徴ベクトルを保存（上限まで）
        for num in features_num:
            if saved >= remaining:
                break
            if num not in self.all_features:
                continue
            entry = self.all_features[num]
            # 横顔・傾き・pose 欠損は正面辞書へ保存しない
            if not self.is_frontal_feature_entry(entry):
                self.get_logger().info(
                    f'[dict_renew] skip non-frontal: feature_num={num} '
                    f'yaw={entry.get("yaw", "?")} pitch={entry.get("pitch", "?")} '
                    f'roll={entry.get("roll", "?")} '
                    f'pose_valid={entry.get("pose_valid", False)}'
                )
                self.release_pending_feature(num)
                skipped_profile += 1
                skipped_feature_nums.append(num)
                continue
            feature = entry["feature"]
            image = self.all_images.get(num)

            if self.is_feature_duplicate(feature):
                self.get_logger().info(
                    f'[dict_renew] skip duplicate feature: feature_num={num}'
                )
                self.release_pending_feature(num)
                skipped += 1
                skipped_feature_nums.append(num)
                continue
            if image is not None and self.is_image_duplicate(image):
                self.get_logger().info(
                    f'[dict_renew] skip duplicate image: feature_num={num}'
                )
                self.release_pending_feature(num)
                skipped += 1
                skipped_feature_nums.append(num)
                continue

            self.dictionary[user_name].append(feature)
            feature_hash = self.compute_feature_hash(feature)
            image_hash = self.compute_image_hash(image) if image is not None else None
            feature_path = None
            image_path = None
            if self.save_registration and user_dir is not None:
                # 保存処理（ディスク連番はユーザー単位で単調増加）
                feature_path = os.path.join(user_dir, f"{user_name}_{disk_idx}.npy")
                np.save(feature_path, feature)
                if image is not None:
                    # 顔画像保存
                    image_path = os.path.join(user_dir, f"{user_name}_{disk_idx}.jpg")
                    if isinstance(image, np.ndarray):
                        image_to_save = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        Image.fromarray(image_to_save).save(image_path, format="JPEG")
                    else:
                        image.save(image_path, format="JPEG")
                disk_idx += 1
            saved_items.append({
                'feature_num': num,
                'feature_path': feature_path,
                'image_path': image_path,
                'feature_hash': feature_hash,
                'image_hash': image_hash,
            })
            self.register_saved_hashes(feature, image)
            self.release_pending_feature(num)
            saved += 1
        if skipped:
            self.get_logger().info(
                f'[dict_renew] skipped {skipped} duplicate feature(s) for {user_name}'
            )
        if skipped_profile:
            self.get_logger().info(
                f'[dict_renew] skipped {skipped_profile} non-frontal '
                f'feature(s) for {user_name}'
            )
        if saved:
            print(
                f"Saved {saved} feature(s) to {user_name} "
                f"(total={len(self.dictionary[user_name])})"
            )
            self.append_registration_memo(
                user_name=user_name,
                route=route,
                source=source,
                people_id=people_id,
                face_id=face_id,
                requested_feature_nums=features_num,
                saved_items=saved_items,
                skipped_feature_nums=skipped_feature_nums,
            )
            if self.save_registration:
                # 直後のディスク同期タイマーで再読込されないよう署名を更新する
                self._disk_backed_users.add(user_name)
                self._face_models_signature = face_models_signature(DIRECTORY)
        return saved

    @staticmethod
    def meets_existing_feature_add_gate(face) -> bool:
        """通常経路で既存ユーザーへ特徴追加する累積スコア条件.

        accumulate_score > 3 かつ 平均スコア > 0.6 のときだけ True。
        ギャラリー救済・ハッシュ一致・DBSCAN 既存マージでは使わない。
        """
        if face.total_features <= 0:
            return False
        average_score = face.accumulate_score / face.total_features
        return face.accumulate_score > 3 and average_score > 0.6

    def save_and_confirm_existing_user(
        self,
        people_id: str,
        user_name: str,
        features_num: list,
        *,
        route: str,
        face_id: str = '',
        source: str = 'dictionary_renew',
    ) -> int:
        """既存ユーザーへ保存を試し、保存条件を満たしたとして確定名を通知する.

        重複・上限で saved=0 でも青枠（/dictionary_update で名前）にする。

        Returns:
            保存件数。
        """
        saved = self.save_features_for_user(
            user_name,
            features_num,
            route=route,
            source=source,
            people_id=people_id,
            face_id=face_id,
        )
        if saved > 0:
            self.rebuild_pca_model_on_disk()
        else:
            self.get_logger().info(
                f'[dict_renew] confirm without new files: saved=0 for {user_name} '
                f'(people_id={people_id}, face_id={face_id or "?"})'
            )
        self.update_dictionary(people_id, user_name)
        return saved

    def dictionary_renew(self, msg: RecognitionHistory):
        """recognition_history に基づき辞書を更新する.

        face_id ごとに評価し、同一 people_id 内の別 face_id とは混ぜない。
        LP が unknown のときは常にギャラリー kNN で既存救済を試みる。
        それでも unknown のバケットは即 new user せず、次の2段で扱う:

        1. ハッシュ一致なら既存へマージ
        2. 未ヒットは未ラベル池へ送り、定期 DBSCAN で密集クラスタだけ登録

        既存ユーザー確定（青枠）は、既存へ保存する条件を満たしたときに出す
        （保存件数0でも可）:

        - 通常: face_id 一致 + 累積>3 かつ平均>0.6
        - ギャラリー救済: LP=unknown → ギャラリー kNN ヒット（スコアゲートなし）
        - ハッシュ一致: 既存と同一特徴（スコアゲートなし）
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
                features_num = [
                    int(fn) for fn in face.features_num
                    if int(fn) in self.all_features
                ]
                # features_num の各番号に対応する特徴が存在するかをチェック
                features_list = [
                    self.all_features[num]["feature"]
                    for num in features_num
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

                # LP が unknown ならギャラリー救済（閾値超え票が半数以上なら既存へ）
                gallery_rescued = False
                if best_user.startswith("unknown"):
                    gallery_user = self.gallery_rescue_decide(
                        features,
                        people_id=people_id,
                        face_id=face_id,
                        threshold=COSINE_THRESHOLD,
                    )
                    if not gallery_user.startswith('unknown'):
                        best_user = gallery_user
                        gallery_rescued = True

                print(
                    f"Selected user: {best_user} "
                    f"(people_id={people_id}, face_id={face_id}, n={len(features_num)}, "
                    f"gallery_rescued={gallery_rescued})"
                )

                # ギャラリーでも unknown: face_id=unknown のみ池行き（即 new user 禁止）
                if best_user.startswith("unknown"):
                    if face_id != "unknown":
                        self.get_logger().info(
                            f'[dict_renew] skip pool: people_id={people_id} '
                            f'face_id={face_id} LP=unknown gallery=miss '
                            f'(only unknown bucket enters unlabeled pool)'
                        )
                        continue

                    existing_user = self.find_existing_user_for_features(features)
                    if existing_user:
                        # ハッシュ一致 → 既存へ保存条件満たす → 青枠
                        self.get_logger().info(
                            f'[dict_renew] duplicate bucket -> merge to {existing_user} '
                            f'(people_id={people_id}, face_id={face_id})'
                        )
                        self.save_and_confirm_existing_user(
                            people_id,
                            existing_user,
                            features_num,
                            route='dictionary_renew(unknown -> existing hash match) -> '
                                  'save_features_for_user',
                            face_id=face_id,
                        )
                        continue

                    # 未ヒットは未ラベル池へ（DBSCAN 後にのみ new user）
                    self.add_features_to_unlabeled_pool(
                        features_num, people_id=people_id, face_id=face_id
                    )
                    self.update_dictionary(people_id, "none")
                    continue

                # 既存の user の場合、face_id が LP 結果と一致するときのみ特徴を辞書に追加
                # ギャラリー救済時は face_id 不問（バケット全体を保存）
                if face_id != best_user and not gallery_rescued:
                    self.get_logger().info(
                        f'[dict_renew] skip existing-user: people_id={people_id} '
                        f'face_id={face_id} LP={best_user} (face_id must match)'
                    )
                    continue

                # ギャラリー救済: バケット全特徴を保存＋青枠
                if gallery_rescued:
                    self.save_and_confirm_existing_user(
                        people_id,
                        best_user,
                        features_num,
                        route='dictionary_renew(gallery_rescued) -> '
                              'save_features_for_user',
                        face_id=face_id,
                    )
                    continue

                # 通常の既存追加: 累積>3 かつ 平均>0.6 が保存条件 → 満たせば青枠
                average_score = face.accumulate_score / face.total_features
                print("average_score: ", average_score)
                if not self.meets_existing_feature_add_gate(face):
                    self.get_logger().info(
                        f'[dict_renew] existing gate fail -> no confirm '
                        f'(people_id={people_id}, face_id={face_id}, '
                        f'user={best_user}, accumulate={face.accumulate_score:.3f}, '
                        f'avg={average_score:.3f})'
                    )
                    self.update_dictionary(people_id, "none")
                    continue

                self.save_and_confirm_existing_user(
                    people_id,
                    best_user,
                    features_num,
                    route='dictionary_renew(existing_user) -> save_features_for_user',
                    face_id=face_id,
                )

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
