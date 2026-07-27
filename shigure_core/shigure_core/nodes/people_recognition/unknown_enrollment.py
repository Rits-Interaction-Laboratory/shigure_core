"""unknown 顔特徴の登録判定（ギャラリー kNN マージ + DBSCAN 新規判定）.

Node 層から ROS2 非依存のアルゴリズムだけを切り出している。
新規ユーザー判定は次の2段:

1. 既存ギャラリーへ閾値付き kNN / max cos 照合 → ヒットなら既存へ強制マージ
2. 未ヒットは未ラベル池へ。DBSCAN で密集クラスタだけを候補人物にし、
   クラスタ中心を再度ギャラリー照合してから new user にする
"""

from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np
from sklearn.cluster import DBSCAN

# ギャラリー照合・クラスタのコサイン類似度しきい値（Node の COSINE_THRESHOLD と揃える）.
GALLERY_COSINE_THRESHOLD = 0.4
# Faiss kNN の近傍数（分裂 ID への票割れ耐性用）.
GALLERY_KNN_K = 10
# DBSCAN: cosine 距離 = 1 - cos。cos>=閾値を近傍とみなす.
DBSCAN_EPS = 1.0 - GALLERY_COSINE_THRESHOLD
# DBSCAN の最小点数（従来の MIN_FEATURES_FOR_NEW_USER 相当）.
DBSCAN_MIN_SAMPLES = 10


def normalize_l2(features: np.ndarray) -> np.ndarray:
    """特徴行列を L2 正規化する（破壊的でない）."""
    feats = np.asarray(features, dtype=np.float32)
    if feats.ndim == 1:
        feats = feats.reshape(1, -1)
    out = feats.copy()
    faiss.normalize_L2(out)
    return out


def flatten_dictionary(
    dictionary: Dict[str, Sequence[np.ndarray]],
    query_dim: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], List[str]]:
    """辞書を (N, D) 行列と user_id リストへ平坦化する."""
    rows: List[np.ndarray] = []
    user_ids: List[str] = []
    for user_id, features in dictionary.items():
        for feature in features:
            vec = np.squeeze(np.asarray(feature, dtype=np.float32))
            if vec.ndim != 1:
                continue
            if query_dim is not None and vec.shape[0] != query_dim:
                continue
            rows.append(vec)
            user_ids.append(user_id)
    if not rows:
        return None, []
    return np.stack(rows).astype(np.float32), user_ids


def feature_centroid(features: np.ndarray) -> np.ndarray:
    """L2 正規化後の平均を再正規化した重心ベクトルを返す."""
    norms = normalize_l2(features)
    centroid = norms.mean(axis=0)
    centroid = centroid.astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(centroid)
    return centroid.reshape(-1)


def find_gallery_user_by_knn(
    query_features: np.ndarray,
    dictionary: Dict[str, Sequence[np.ndarray]],
    threshold: float = GALLERY_COSINE_THRESHOLD,
    k: int = GALLERY_KNN_K,
) -> Tuple[Optional[str], dict]:
    """クエリ特徴群をギャラリーへ kNN 照合し、ヒットすれば user_id を返す.

    閾値以上の近傍があれば必ず既存ユーザーを返す（投票比率不足でも new user にしない）。
    返り値: (user_id or None, デバッグ情報 dict)
    """
    info = {
        'hit': False,
        'best_user': None,
        'best_score': 0.0,
        'vote_count': 0,
        'n_queries': 0,
        'votes': {},
    }
    if not dictionary:
        return None, info

    queries = np.asarray(query_features, dtype=np.float32)
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)
    query_dim = queries.shape[1]
    gallery, user_ids = flatten_dictionary(dictionary, query_dim=query_dim)
    if gallery is None:
        return None, info

    index_feats = normalize_l2(gallery)
    index = faiss.IndexFlatIP(index_feats.shape[1])
    index.add(index_feats)

    q = normalize_l2(queries)
    info['n_queries'] = int(q.shape[0])
    knn = min(k, index_feats.shape[0])
    distances, indices = index.search(q, knn)

    votes: Counter = Counter()
    score_sums: Counter = Counter()
    best_score = -1.0
    best_user_for_score = None

    for i in range(q.shape[0]):
        for j in range(knn):
            score = float(distances[i][j])
            idx = int(indices[i][j])
            if idx < 0:
                continue
            if score <= threshold:
                continue
            user_id = user_ids[idx]
            votes[user_id] += 1
            score_sums[user_id] += score
            if score > best_score:
                best_score = score
                best_user_for_score = user_id

    info['best_score'] = float(best_score) if best_score >= 0 else 0.0
    info['votes'] = dict(votes)

    if not votes:
        return None, info

    # 票数優先、同票ならスコア合計が大きいユーザー
    best_user, vote_count = max(
        votes.items(),
        key=lambda item: (item[1], score_sums[item[0]]),
    )
    info['hit'] = True
    info['best_user'] = best_user
    info['vote_count'] = int(vote_count)
    # max cos のユーザーと票数が食い違う場合は票の勝者を採用（分裂耐性）
    if best_user_for_score is not None and best_user_for_score != best_user:
        info['max_score_user'] = best_user_for_score
    return best_user, info


def match_centroid_to_gallery(
    centroid: np.ndarray,
    dictionary: Dict[str, Sequence[np.ndarray]],
    threshold: float = GALLERY_COSINE_THRESHOLD,
    k: int = GALLERY_KNN_K,
) -> Tuple[Optional[str], dict]:
    """クラスタ重心をギャラリーへ照合する."""
    return find_gallery_user_by_knn(
        centroid.reshape(1, -1),
        dictionary,
        threshold=threshold,
        k=k,
    )


def dbscan_cluster_features(
    features: np.ndarray,
    eps: float = DBSCAN_EPS,
    min_samples: int = DBSCAN_MIN_SAMPLES,
) -> np.ndarray:
    """未ラベル特徴を cosine 距離 DBSCAN でクラスタリングする.

    Returns:
        各サンプルのラベル。-1 はノイズ。
    """
    feats = np.asarray(features, dtype=np.float32)
    if feats.ndim == 1:
        feats = feats.reshape(1, -1)
    if feats.shape[0] == 0:
        return np.array([], dtype=np.int64)
    # 正規化済みでも cosine 距離は 1-cos。DBSCAN は密度連結で人物候補を切る。
    norms = normalize_l2(feats)
    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric='cosine',
        n_jobs=1,
    )
    labels = clustering.fit_predict(norms)
    return labels.astype(np.int64)


def select_dense_clusters(
    labels: np.ndarray,
) -> List[Tuple[int, np.ndarray]]:
    """ノイズ以外のクラスタについて (label, メンバーindex配列) を返す."""
    clusters: List[Tuple[int, np.ndarray]] = []
    if labels.size == 0:
        return clusters
    for label in sorted(set(labels.tolist())):
        if label < 0:
            continue
        indices = np.where(labels == label)[0]
        clusters.append((int(label), indices))
    return clusters
