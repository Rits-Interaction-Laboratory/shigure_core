"""InsightFace 検出 + AdaFace 埋め込みのラッパー.

検出・yaw/pitch は InsightFace (buffalo_s の SCRFD / 3D landmark)、
認識ベクトルだけ AdaFace ONNX で計算する。buffalo の ArcFace は読まない。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from shigure_core.nodes.people_recognition.adaface import (
    AdaFaceEmbedder,
    select_onnx_providers,
)

logger = logging.getLogger(__name__)

FRONTAL_YAW_THRESHOLD = 30.0
FRONTAL_PITCH_THRESHOLD = 20.0
MIN_FACE_SIZE = 40


@dataclass
class DetectedFace:
    """検出顔。埋め込みは AdaFace、姿勢は InsightFace 由来."""

    bbox: Tuple[int, int, int, int]  # x, y, w, h in full image coordinates
    embedding: np.ndarray
    det_score: float
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    @property
    def is_frontal(self) -> bool:
        """yaw/pitch が正面しきい値内なら True."""
        return FaceAnalyzer.is_frontal(self.yaw, self.pitch)


class FaceAnalyzer:
    """InsightFace で検出・姿勢、AdaFace で埋め込みを返す."""

    def __init__(self, det_size: Tuple[int, int] = (640, 640)):
        """検出入力サイズを保持する。モデルは初回 detect 時に読む."""
        self._det_size = det_size
        self._app = None
        self._norm_crop = None
        self._embedder = AdaFaceEmbedder()
        self._available = None

    @property
    def available(self) -> bool:
        """検出器と推論ランタイムが import できれば True."""
        if self._available is None:
            try:
                import insightface  # noqa: F401
            except ImportError:
                self._available = False
                return False
            self._available = self._embedder.available
        return self._available

    def _ensure_app(self):
        if self._app is not None:
            return
        if not self.available:
            raise RuntimeError(
                'insightface or onnxruntime is not installed'
            )
        from insightface.app import FaceAnalysis

        providers, use_gpu = select_onnx_providers()
        # 認識(ArcFace)は読まない。検出と 3D ランドマーク（pose）だけ使う。
        self._app = FaceAnalysis(
            name='buffalo_s',
            allowed_modules=['detection', 'landmark_3d_68'],
            providers=providers,
        )
        ctx_id = 0 if use_gpu else -1
        self._app.prepare(ctx_id=ctx_id, det_size=self._det_size)
        from insightface.utils.face_align import norm_crop
        self._norm_crop = norm_crop
        logger.info(
            'InsightFace detector ready: gpu=%s det_size=%s',
            use_gpu,
            self._det_size,
        )

    @staticmethod
    def is_frontal(
        yaw: float,
        pitch: float = 0.0,
        yaw_threshold: float = FRONTAL_YAW_THRESHOLD,
        pitch_threshold: float = FRONTAL_PITCH_THRESHOLD,
    ) -> bool:
        """yaw/pitch がしきい値未満なら正面とみなす."""
        return abs(yaw) < yaw_threshold and abs(pitch) < pitch_threshold

    @staticmethod
    def is_point_in_box(
            point: Tuple[float, float],
            box: Tuple[int, int, int, int]) -> bool:
        """点が bbox (x, y, w, h) の内側なら True."""
        x, y = point
        box_x, box_y, box_width, box_height = box
        return (box_x <= x <= box_x + box_width
                and box_y <= y <= box_y + box_height)

    def _embed_face(self, image: np.ndarray, face) -> Optional[np.ndarray]:
        """5点ランドマークで 112x112 に整列し、AdaFace 埋め込みを返す."""
        kps = getattr(face, 'kps', None)
        if kps is None:
            return None
        aligned = self._norm_crop(image, landmark=kps, image_size=112)
        if aligned is None or aligned.size == 0:
            return None
        return self._embedder.embed(aligned)

    def detect_faces(
            self,
            image: np.ndarray,
            min_size: int = MIN_FACE_SIZE,
            det_thresh: Optional[float] = None) -> List[DetectedFace]:
        """全画面で顔検出し、bbox と AdaFace embedding の一覧を返す."""
        if image is None or image.size == 0:
            return []

        self._ensure_app()
        old_thresh = None
        if det_thresh is not None:
            old_thresh = self._app.det_model.det_thresh
            self._app.det_model.det_thresh = det_thresh
        try:
            faces = self._app.get(image)
        finally:
            if old_thresh is not None:
                self._app.det_model.det_thresh = old_thresh
        results: List[DetectedFace] = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            if w < min_size or h < min_size:
                continue
            embedding = self._embed_face(image, face)
            if embedding is None:
                continue
            det_score = float(getattr(face, 'det_score', 0.0))
            pitch, yaw, roll = 0.0, 0.0, 0.0
            pose = getattr(face, 'pose', None)
            if pose is not None and len(pose) >= 3:
                pitch, yaw, roll = (
                    float(pose[0]), float(pose[1]), float(pose[2])
                )
            results.append(DetectedFace(
                bbox=(x1, y1, w, h),
                embedding=embedding,
                det_score=det_score,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
            ))
        return results

    def embed_bgr_crop(self, crop: np.ndarray) -> np.ndarray:
        """整列できないクロップを 112x112 にリサイズして埋め込む."""
        return self._embedder.embed(crop)

    def find_face_for_head(
        self,
        faces: List[DetectedFace],
        head_point,
    ) -> Optional[DetectedFace]:
        """
        検出済みの顔 bbox 一覧から、頭部座標が bbox 内にある顔を返す.

        複数候補がある場合は det_score が最も高い顔を採用する。
        """
        head = (head_point.x, head_point.y)
        best: Optional[DetectedFace] = None
        best_score = -1.0
        for candidate in faces:
            if not self.is_point_in_box(head, candidate.bbox):
                continue
            if candidate.det_score > best_score:
                best = candidate
                best_score = candidate.det_score
        return best
