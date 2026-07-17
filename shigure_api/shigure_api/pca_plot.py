"""PCA feature-map state for /ws/pca_plot WebSocket clients."""

from __future__ import annotations

import asyncio
import pickle
import queue
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from pydantic import BaseModel

from shigure_api.config import PRESENCE_TIMEOUT_SEC
from shigure_api.feature_images import feature_num_from_stem

Point2D = Tuple[float, float]


class PcaPlotPoint(BaseModel):
    x: float
    y: float
    feature_num: int


class PreConfirmTrajectory(BaseModel):
    people_id: str
    face_id: Literal['unknown'] = 'unknown'
    points: List[Tuple[float, float]]


class PcaPlotState(BaseModel):
    type: Literal['pca_plot_state'] = 'pca_plot_state'
    timestamp: str
    sequence: int
    dictionary: Dict[str, List[PcaPlotPoint]]
    labeled_new: Dict[str, List[PcaPlotPoint]]
    unlabeled: List[PcaPlotPoint]
    trajectories_pre_confirm: List[PreConfirmTrajectory]


class PcaUnlabeledUpdate(BaseModel):
    type: Literal['pca_unlabeled_update'] = 'pca_unlabeled_update'
    timestamp: str
    sequence: int
    unlabeled: List[PcaPlotPoint]


class PcaLabeledUpdate(BaseModel):
    """確定後ユーザーの点が unlabeled と同様に増えていくときの差分配信。"""

    type: Literal['pca_labeled_update'] = 'pca_labeled_update'
    timestamp: str
    sequence: int
    labeled_new: Dict[str, List[PcaPlotPoint]]


class PcaTrajectoryUpdate(BaseModel):
    type: Literal['pca_trajectory_update'] = 'pca_trajectory_update'
    timestamp: str
    trajectories_pre_confirm: List[PreConfirmTrajectory]


class PcaSegmentClosed(BaseModel):
    type: Literal['pca_segment_closed'] = 'pca_segment_closed'
    timestamp: str
    people_id: str
    user_id: str
    promoted_feature_nums: List[int]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PcaTransformer:
    """512-dim embedding -> 2D using pca_model.pkl (same rules as feature_plot node)."""

    def __init__(self, pca_path: Path) -> None:
        self._pca_path = pca_path
        self._pca = None
        self._expected_dim: Optional[int] = None
        self.reload(pca_path)

    def reload(self, pca_path: Optional[Path] = None) -> None:
        path = pca_path or self._pca_path
        self._pca_path = path
        self._pca = None
        self._expected_dim = None
        if path.is_file():
            with open(path, 'rb') as f:
                self._pca = pickle.load(f)
            self._expected_dim = getattr(self._pca, 'n_features_in_', None)

    def transform(self, feature: List[float]) -> Optional[Point2D]:
        import numpy as np

        feat = np.asarray(feature, dtype=np.float32).reshape(1, -1)
        try:
            if self._pca is not None and (
                self._expected_dim is None or feat.shape[1] == self._expected_dim
            ):
                pt = self._pca.transform(feat)[0]
            else:
                import faiss

                faiss.normalize_L2(feat)
                pt = feat[0, :2]
            return float(pt[0]), float(pt[1])
        except Exception:
            return None


def load_dictionary_points(
    face_models_dir: Path, transformer: PcaTransformer
) -> Dict[str, List[PcaPlotPoint]]:
    out: Dict[str, List[PcaPlotPoint]] = {}
    if not face_models_dir.is_dir():
        return out
    for user_dir in sorted(face_models_dir.glob('user_*')):
        if not user_dir.is_dir():
            continue
        uid = user_dir.name
        points: List[PcaPlotPoint] = []
        for npy_path in sorted(user_dir.glob('*.npy')):
            import numpy as np

            fn = feature_num_from_stem(uid, npy_path.stem)
            if fn is None:
                continue
            raw = np.squeeze(np.load(npy_path)).astype(np.float32).reshape(-1).tolist()
            pt = transformer.transform(raw)
            if pt is not None:
                points.append(PcaPlotPoint(x=pt[0], y=pt[1], feature_num=fn))
        if points:
            out[uid] = points
    return out


def should_promote_face_for_user(face_id: str, confirmed_user_id: str) -> bool:
    """Promote PCA points only for the face_id bucket that matches confirmation."""
    if face_id.endswith('@profile'):
        return False
    if face_id == confirmed_user_id:
        return True
    # New user_new* registration uses the unknown bucket at confirm time.
    if face_id == 'unknown' and confirmed_user_id.startswith('user_new'):
        return True
    return False


class PcaPlotStateBuilder:
    """Thread-safe PCA plot state (mirrors feature_plot logic, pre_confirm trajectories only)."""

    def __init__(
        self,
        face_models_dir: Path,
        pca_path: Path,
        *,
        redraw_every: int = 5,
        trajectory_max_points: int = 50,
        presence_timeout: float = PRESENCE_TIMEOUT_SEC,
    ) -> None:
        self._lock = threading.Lock()
        self._redraw_every = max(1, redraw_every)
        self._trajectory_max_points = max(2, trajectory_max_points)
        # 今映っているユーザーのみプロットするための在室追跡（user_id -> 最終認識時刻）。
        self._presence_timeout = max(0.0, presence_timeout)
        self._present_last_seen: Dict[str, float] = {}
        # 未確定ユーザー（people_id 単位）の在室追跡と、その people が持つ unlabeled 特徴番号。
        self._present_people_last_seen: Dict[str, float] = {}
        self._people_feature_nums: Dict[str, Set[int]] = {}
        self._sequence = 0
        self._feature_counter = 0
        self._face_models_dir = face_models_dir
        self._pca_path = pca_path

        self._transformer = PcaTransformer(pca_path)
        self._dictionary: Dict[str, List[PcaPlotPoint]] = load_dictionary_points(
            face_models_dir, self._transformer
        )
        self._labeled_new: Dict[str, List[PcaPlotPoint]] = defaultdict(list)
        # 確定時に一括追加せず、unlabeled と同様に少しずつ labeled_new へ移す待ち行列。
        self._pending_labeled: Dict[str, List[PcaPlotPoint]] = defaultdict(list)
        # feature_num -> 確定済み user_id（確定後の新規点を labeled に振り分ける用）。
        self._feature_to_user: Dict[int, str] = {}
        self._unlabeled: Dict[int, Point2D] = {}
        self._feature_map: Dict[int, Point2D] = {}
        self._raw_features: Dict[int, List[float]] = {}
        self._confirmed_people: Dict[str, str] = {}
        self._promoted_feature_nums: Set[int] = set()
        self._latest_history = None
        self._known_user_count = sum(
            1 for p in face_models_dir.glob('user_*') if p.is_dir()
        ) if face_models_dir.is_dir() else 0

    def on_feature_info(self, feature_num: int, feature: List[float]) -> Tuple[bool, bool]:
        """特徴を取り込み、配信が必要なら (unlabeled要配信, labeled要配信) を返す。"""
        pt = self._transformer.transform(feature)
        if pt is None:
            return False, False
        with self._lock:
            self._raw_features[feature_num] = list(feature)
            self._feature_map[feature_num] = pt
            push_unlabeled = False
            push_labeled = False
            owner = self._feature_to_user.get(feature_num)
            if owner:
                self._append_labeled_point_locked(
                    owner, PcaPlotPoint(x=pt[0], y=pt[1], feature_num=feature_num)
                )
            elif feature_num not in self._promoted_feature_nums:
                self._unlabeled[feature_num] = pt
            self._feature_counter += 1
            # unlabeled と同様、redraw_every ごとに現在の一覧を配信する。
            if self._feature_counter % self._redraw_every != 0:
                return False, False
            drained = self._drain_pending_labeled_locked(self._redraw_every)
            has_labeled = drained or any(self._labeled_new.values())
            return True, has_labeled

    def tick_labeled_drip(self) -> bool:
        """タイマー用: pending の点を labeled_new へ移す。配信が必要なら True。"""
        with self._lock:
            return self._drain_pending_labeled_locked(self._redraw_every)

    def _labeled_feature_nums_locked(self, user_id: str) -> Set[int]:
        fns = {p.feature_num for p in self._labeled_new.get(user_id, [])}
        fns |= {p.feature_num for p in self._pending_labeled.get(user_id, [])}
        return fns

    def _append_labeled_point_locked(self, user_id: str, point: PcaPlotPoint) -> bool:
        """labeled_new に未登録の点を追加する。追加したら True。"""
        if point.feature_num in self._labeled_feature_nums_locked(user_id):
            return False
        self._labeled_new[user_id].append(point)
        self._feature_to_user[point.feature_num] = user_id
        self._promoted_feature_nums.add(point.feature_num)
        self._unlabeled.pop(point.feature_num, None)
        return True

    def _enqueue_pending_labeled_locked(self, user_id: str, point: PcaPlotPoint) -> None:
        """確定時の一括点を pending に積む（即 labeled_new には入れない）。"""
        if point.feature_num in self._labeled_feature_nums_locked(user_id):
            return
        self._pending_labeled[user_id].append(point)
        self._feature_to_user[point.feature_num] = user_id
        self._promoted_feature_nums.add(point.feature_num)
        self._unlabeled.pop(point.feature_num, None)

    def _drain_pending_labeled_locked(self, max_points: int) -> bool:
        """pending から最大 max_points 個を labeled_new へ移す。"""
        if max_points <= 0:
            return False
        moved = 0
        for user_id in list(self._pending_labeled.keys()):
            queue = self._pending_labeled[user_id]
            while queue and moved < max_points:
                point = queue.pop(0)
                existing = {p.feature_num for p in self._labeled_new[user_id]}
                if point.feature_num not in existing:
                    self._labeled_new[user_id].append(point)
                    moved += 1
            if not queue:
                self._pending_labeled.pop(user_id, None)
            if moved >= max_points:
                break
        return moved > 0

    def rebuild_pca_from_disk(self, registered_user_id: Optional[str] = None) -> bool:
        """Rebuild pca_model.pkl, reload coordinates, and refresh dictionary from disk."""
        from shigure_core.util.pca_model import build_pca_model, count_user_dirs

        face_models_dir = self._face_models_dir
        user_count = count_user_dirs(face_models_dir)
        if registered_user_id is None and user_count <= self._known_user_count:
            return False

        result = build_pca_model(face_models_dir, self._pca_path)
        if not result.success:
            return False

        with self._lock:
            self._known_user_count = user_count
            self._transformer.reload(self._pca_path)
            self._dictionary = load_dictionary_points(face_models_dir, self._transformer)

            for fn, raw in list(self._raw_features.items()):
                pt = self._transformer.transform(raw)
                if pt is None:
                    continue
                self._feature_map[fn] = pt
                if fn in self._unlabeled:
                    self._unlabeled[fn] = pt

            for uid, points in list(self._labeled_new.items()):
                rebuilt: List[PcaPlotPoint] = []
                for point in points:
                    raw = self._raw_features.get(point.feature_num)
                    if raw is None:
                        rebuilt.append(point)
                        continue
                    pt = self._transformer.transform(raw)
                    if pt is not None:
                        rebuilt.append(
                            PcaPlotPoint(
                                x=pt[0], y=pt[1], feature_num=point.feature_num
                            )
                        )
                if rebuilt:
                    self._labeled_new[uid] = rebuilt

            for uid, points in list(self._pending_labeled.items()):
                rebuilt_pending: List[PcaPlotPoint] = []
                for point in points:
                    raw = self._raw_features.get(point.feature_num)
                    if raw is None:
                        rebuilt_pending.append(point)
                        continue
                    pt = self._transformer.transform(raw)
                    if pt is not None:
                        rebuilt_pending.append(
                            PcaPlotPoint(
                                x=pt[0], y=pt[1], feature_num=point.feature_num
                            )
                        )
                if rebuilt_pending:
                    self._pending_labeled[uid] = rebuilt_pending
                else:
                    self._pending_labeled.pop(uid, None)

            if registered_user_id:
                # ディスク辞書に新ユーザーが載るが、一括表示せず pending 経由で増分表示する。
                self._labeled_new.pop(registered_user_id, None)
                disk_points = list(self._dictionary.pop(registered_user_id, []))
                self._pending_labeled.pop(registered_user_id, None)
                for point in disk_points:
                    self._enqueue_pending_labeled_locked(registered_user_id, point)

        return True

    def reload_dictionary_from_disk(self) -> None:
        """Reload dictionary points from face_models without rebuilding PCA."""
        with self._lock:
            self._dictionary = load_dictionary_points(
                self._face_models_dir, self._transformer
            )

    def clear_labeled_new(self, user_id: str) -> None:
        with self._lock:
            self._labeled_new.pop(user_id, None)

    def mark_present(self, user_id: str) -> None:
        """ユーザーが認識されたことを記録する（在室扱いにする）。認識フレームごとに呼ぶ。"""
        if not user_id or user_id in ('none', 'unknown'):
            return
        with self._lock:
            self._present_last_seen[user_id] = time.monotonic()

    def _forget_user_plot_locked(self, user_id: str) -> None:
        """退出した確定ユーザーの蓄積プロットを破棄する（再登場時に前回分を再表示しない）。"""
        self._labeled_new.pop(user_id, None)
        self._pending_labeled.pop(user_id, None)
        owned = [fn for fn, uid in self._feature_to_user.items() if uid == user_id]
        for fn in owned:
            self._feature_to_user.pop(fn, None)
            self._promoted_feature_nums.discard(fn)
            self._feature_map.pop(fn, None)
            self._raw_features.pop(fn, None)
            self._unlabeled.pop(fn, None)
        # 確定関係も解除し、再入室時は改めてリアルタイム点だけ積む。
        stale_people = [
            people_id
            for people_id, name in self._confirmed_people.items()
            if name == user_id
        ]
        for people_id in stale_people:
            self._confirmed_people.pop(people_id, None)

    def present_user_ids_locked(self) -> Set[str]:
        """在室タイムアウトを超えたユーザーを除去し、今映っている user_id 集合を返す（要ロック済み）。"""
        now = time.monotonic()
        expired = [
            user_id
            for user_id, seen in self._present_last_seen.items()
            if now - seen > self._presence_timeout
        ]
        for user_id in expired:
            self._present_last_seen.pop(user_id, None)
            # 退出したら蓄積プロットを捨て、再登場時に前回分が復活しないようにする。
            self._forget_user_plot_locked(user_id)
        return set(self._present_last_seen.keys())

    def present_user_ids(self) -> Set[str]:
        """今映っている user_id 集合を返す（退室分は除去済み）。"""
        with self._lock:
            return self.present_user_ids_locked()

    def present_people_ids_locked(self) -> Set[str]:
        """在室タイムアウトを超えた未確定 people を除去し、今映っている people_id 集合を返す（要ロック済み）。"""
        now = time.monotonic()
        expired = [
            people_id
            for people_id, seen in self._present_people_last_seen.items()
            if now - seen > self._presence_timeout
        ]
        for people_id in expired:
            self._present_people_last_seen.pop(people_id, None)
            self._people_feature_nums.pop(people_id, None)
        return set(self._present_people_last_seen.keys())

    def present_unlabeled_fns_locked(self) -> Set[int]:
        """今映っている未確定 people が持つ unlabeled 特徴番号の集合を返す（要ロック済み）。"""
        present_people = self.present_people_ids_locked()
        fns: Set[int] = set()
        for people_id in present_people:
            fns |= self._people_feature_nums.get(people_id, set())
        return fns

    def presence_signature(self) -> frozenset:
        """確定ユーザーと未確定 people を合わせた在室シグネチャ（変化検知用）。"""
        with self._lock:
            users = self.present_user_ids_locked()
            people = self.present_people_ids_locked()
            return frozenset(users) | frozenset(f'people:{p}' for p in people)

    def on_people_tracking(self, msg) -> None:
        """骨格追跡(/shigure/people_detection)から在室を更新する。

        顔が検出されなくても、確定済みユーザーは同じ骨格(people_id)が
        追跡されている限りプロットを維持する。
        """
        now = time.monotonic()
        with self._lock:
            for pose in msg.pose_key_points_list:
                people_id = pose.people_id
                # 辞書確定済み: 骨格が残っている限り在室（顔認識不要）。
                if people_id in self._confirmed_people:
                    self._present_last_seen[self._confirmed_people[people_id]] = now
                    continue
                # pose 側の確定名（末尾?なし）も在室扱い。
                face_name = (pose.face_name or '').strip()
                if (
                    face_name
                    and not face_name.endswith('?')
                    and face_name.startswith('user')
                    and face_name not in ('none', 'unknown')
                ):
                    self._present_last_seen[face_name] = now

    def on_recognition_history(self, msg) -> None:
        now = time.monotonic()
        with self._lock:
            self._latest_history = msg
            for user in msg.users:
                people_id = user.people_id
                if people_id in self._confirmed_people:
                    # 確定済み: 在室更新しつつ、新しい特徴を labeled 側へ紐付ける。
                    user_id = self._confirmed_people[people_id]
                    self._present_last_seen[user_id] = now
                    for face in user.face_info:
                        if face.id.endswith('@profile'):
                            continue
                        if not should_promote_face_for_user(face.id, user_id):
                            continue
                        for fn in face.features_num:
                            fn_int = int(fn)
                            self._feature_to_user[fn_int] = user_id
                            self._promoted_feature_nums.add(fn_int)
                            self._unlabeled.pop(fn_int, None)
                            pt = self._feature_map.get(fn_int)
                            if pt is not None:
                                # 履歴上の新規点も pending 経由で増分表示する。
                                self._enqueue_pending_labeled_locked(
                                    user_id,
                                    PcaPlotPoint(
                                        x=pt[0], y=pt[1], feature_num=fn_int
                                    ),
                                )
                    continue
                fns: Set[int] = set()
                for face in user.face_info:
                    # 非正面(@profile)も含め、未認識(unknown)の顔だけを unlabeled 対象にする。
                    # 登録ユーザーに一致した顔（'user_x' / 'user_x@profile'）は除外する。
                    base_id = (
                        face.id[: -len('@profile')]
                        if face.id.endswith('@profile')
                        else face.id
                    )
                    if base_id != 'unknown':
                        continue
                    fns.update(int(fn) for fn in face.features_num)
                # 未確定 people を在室として記録し、その unlabeled 特徴番号を更新する。
                self._present_people_last_seen[people_id] = now
                self._people_feature_nums[people_id] = fns

    def on_dictionary_update(self, people_id: str, name: str) -> Optional[PcaSegmentClosed]:
        if not name or name == 'none':
            return None
        promoted: List[int] = []
        with self._lock:
            self._confirmed_people[people_id] = name
            # 確定直後から在室扱い（骨格追跡が来るまでの空白を埋める）。
            self._present_last_seen[name] = time.monotonic()
            # 確定した people は未確定の在室追跡から外す（未確定プロットとしては消す）。
            self._present_people_last_seen.pop(people_id, None)
            self._people_feature_nums.pop(people_id, None)
            if self._latest_history is not None:
                for user in self._latest_history.users:
                    if user.people_id != people_id:
                        continue
                    for face in user.face_info:
                        if not should_promote_face_for_user(face.id, name):
                            continue
                        for fn in face.features_num:
                            fn_int = int(fn)
                            promoted.append(fn_int)
                            pt = self._feature_map.get(fn_int)
                            if pt is not None:
                                # 一気に labeled_new へは入れず、pending 経由で増分表示する。
                                self._enqueue_pending_labeled_locked(
                                    name,
                                    PcaPlotPoint(
                                        x=pt[0], y=pt[1], feature_num=fn_int
                                    ),
                                )
                            else:
                                self._feature_to_user[fn_int] = name
                                self._promoted_feature_nums.add(fn_int)
                                self._unlabeled.pop(fn_int, None)
            promoted = sorted(set(promoted))
        return PcaSegmentClosed(
            timestamp=_now_iso(),
            people_id=people_id,
            user_id=name,
            promoted_feature_nums=promoted,
        )

    def build_unlabeled_update(self) -> PcaUnlabeledUpdate:
        with self._lock:
            self._sequence += 1
            present_fns = self.present_unlabeled_fns_locked()
            unlabeled = [
                PcaPlotPoint(feature_num=fn, x=pt[0], y=pt[1])
                for fn, pt in sorted(self._unlabeled.items())
                if fn in present_fns
            ]
            # デバッグ: FeatureInfo 到着ごと（5フレーム毎）の点数。顔が映っている間に出る。
            print(
                f'[pca][B:unlabeled_update] seq={self._sequence} '
                f'unlabeled_total={len(self._unlabeled)} present_fns={len(present_fns)} '
                f'unlabeled_shown={len(unlabeled)} '
                f'present_people={sorted(self._present_people_last_seen.keys())}',
                flush=True,
            )
            return PcaUnlabeledUpdate(
                timestamp=_now_iso(),
                sequence=self._sequence,
                unlabeled=unlabeled,
            )

    def build_labeled_update(self) -> PcaLabeledUpdate:
        with self._lock:
            self._sequence += 1
            present = self.present_user_ids_locked()
            labeled = {
                k: list(v) for k, v in self._labeled_new.items() if k in present
            }
            pending_counts = {
                k: len(v) for k, v in self._pending_labeled.items() if v
            }
            print(
                f'[pca][C:labeled_update] seq={self._sequence} '
                f'labeled_users={len(labeled)} '
                f'labeled_points={sum(len(v) for v in labeled.values())} '
                f'pending={pending_counts} present_users={sorted(present)}',
                flush=True,
            )
            return PcaLabeledUpdate(
                timestamp=_now_iso(),
                sequence=self._sequence,
                labeled_new=labeled,
            )

    def build_trajectory_update(self) -> PcaTrajectoryUpdate:
        with self._lock:
            return PcaTrajectoryUpdate(
                timestamp=_now_iso(),
                trajectories_pre_confirm=self._build_trajectories_pre_confirm(),
            )

    def build_state(self) -> PcaPlotState:
        with self._lock:
            self._sequence += 1
            # 未確定 people の在室（＋その unlabeled 特徴番号）を先に確定させる。
            present_fns = self.present_unlabeled_fns_locked()
            trajectories = self._build_trajectories_pre_confirm()
            unlabeled = [
                PcaPlotPoint(feature_num=fn, x=pt[0], y=pt[1])
                for fn, pt in sorted(self._unlabeled.items())
                if fn in present_fns
            ]
            # 確定ユーザーは、この実行中に取得した labeled_new の点だけを表示する。
            # ディスク上の辞書特徴は PCA 変換には使うが、プロットには含めない。
            present = self.present_user_ids_locked()
            labeled_shown = sum(1 for k in self._labeled_new if k in present)
            # デバッグ: 点の生成有無と在室フィルタ通過後の件数を並べて出す。
            # unlabeled_total>0 かつ unlabeled_shown==0 なら「点はあるがフィルタで全除外」。
            print(
                f'[pca][A:build_state] seq={self._sequence} '
                f'unlabeled_total={len(self._unlabeled)} present_fns={len(present_fns)} '
                f'unlabeled_shown={len(unlabeled)} '
                f'dict_total={len(self._dictionary)} dict_shown=0 '
                f'labeled_total={len(self._labeled_new)} labeled_shown={labeled_shown} '
                f'present_users={sorted(present)} '
                f'present_people={sorted(self._present_people_last_seen.keys())}',
                flush=True,
            )
            return PcaPlotState(
                timestamp=_now_iso(),
                sequence=self._sequence,
                # API 互換性のためフィールドは残し、過去の辞書特徴は常に非表示にする。
                dictionary={},
                labeled_new={
                    k: list(v) for k, v in self._labeled_new.items() if k in present
                },
                unlabeled=unlabeled,
                trajectories_pre_confirm=trajectories,
            )

    def _build_trajectories_pre_confirm(self) -> List[PreConfirmTrajectory]:
        if self._latest_history is None:
            return []
        out: List[PreConfirmTrajectory] = []
        for user in self._latest_history.users:
            people_id = user.people_id
            if people_id in self._confirmed_people:
                continue
            # 退室（在室タイムアウト超過）した未確定 people の軌跡は出さない。
            if people_id not in self._present_people_last_seen:
                continue
            feature_nums: List[int] = []
            for face in user.face_info:
                if face.id.endswith('@profile'):
                    continue
                if face.id != 'unknown':
                    continue
                feature_nums.extend(int(fn) for fn in face.features_num)
            feature_nums = sorted(set(feature_nums))
            points: List[Point2D] = []
            for fn in feature_nums:
                pt = self._feature_map.get(fn)
                if pt is not None:
                    points.append(pt)
            if len(points) > self._trajectory_max_points:
                points = points[-self._trajectory_max_points :]
            if len(points) >= 2:
                out.append(
                    PreConfirmTrajectory(
                        people_id=people_id,
                        points=points,
                    )
                )
        return out


class PcaPlotHub:
    """Broadcast PCA plot JSON to WebSocket clients."""

    def __init__(self) -> None:
        self._clients: Set[Any] = set()
        self._lock = asyncio.Lock()
        self._thread_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
        self._latest_state: Optional[Dict[str, Any]] = None

    async def start(self) -> None:
        asyncio.create_task(self._broadcast_loop())

    def enqueue(self, payload: Dict[str, Any]) -> None:
        self._thread_queue.put_nowait(payload)

    async def _broadcast_loop(self) -> None:
        while True:
            payload = await asyncio.to_thread(self._thread_queue.get)
            if payload.get('type') == 'pca_plot_state':
                self._latest_state = payload
            elif (
                payload.get('type') == 'pca_unlabeled_update'
                and self._latest_state is not None
            ):
                self._latest_state = {
                    **self._latest_state,
                    'unlabeled': payload.get('unlabeled', []),
                    'sequence': payload.get('sequence', self._latest_state.get('sequence', 0)),
                    'timestamp': payload.get(
                        'timestamp', self._latest_state.get('timestamp', '')
                    ),
                }
            elif (
                payload.get('type') == 'pca_labeled_update'
                and self._latest_state is not None
            ):
                self._latest_state = {
                    **self._latest_state,
                    'labeled_new': payload.get('labeled_new', {}),
                    'sequence': payload.get(
                        'sequence', self._latest_state.get('sequence', 0)
                    ),
                    'timestamp': payload.get(
                        'timestamp', self._latest_state.get('timestamp', '')
                    ),
                }
            elif (
                payload.get('type') == 'pca_trajectory_update'
                and self._latest_state is not None
            ):
                self._latest_state = {
                    **self._latest_state,
                    'trajectories_pre_confirm': payload.get(
                        'trajectories_pre_confirm', []
                    ),
                    'timestamp': payload.get('timestamp', self._latest_state['timestamp']),
                }
            async with self._lock:
                dead = []
                for ws in self._clients:
                    try:
                        await ws.send_json(payload)
                    except Exception:
                        dead.append(ws)
                for ws in dead:
                    self._clients.discard(ws)

    async def connect(self, websocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._clients.add(websocket)
        if self._latest_state is not None:
            await websocket.send_json(self._latest_state)

    async def disconnect(self, websocket) -> None:
        async with self._lock:
            self._clients.discard(websocket)


def state_to_dict(state: PcaPlotState) -> Dict[str, Any]:
    return state.model_dump()


def unlabeled_update_to_dict(msg: PcaUnlabeledUpdate) -> Dict[str, Any]:
    return msg.model_dump()


def labeled_update_to_dict(msg: PcaLabeledUpdate) -> Dict[str, Any]:
    return msg.model_dump()


def trajectory_update_to_dict(msg: PcaTrajectoryUpdate) -> Dict[str, Any]:
    return msg.model_dump()


def segment_closed_to_dict(msg: PcaSegmentClosed) -> Dict[str, Any]:
    return msg.model_dump()
