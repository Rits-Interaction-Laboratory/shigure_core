"""InsightFace (SCRFD + ArcFace) wrapper for profile-face detection and embedding."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

FRONTAL_YAW_THRESHOLD = 30.0
FRONTAL_PITCH_THRESHOLD = 20.0


@dataclass
class InsightFaceResult:
    """Detected profile face with ArcFace embedding."""

    bbox: Tuple[int, int, int, int]  # x, y, w, h in full image coordinates
    embedding: np.ndarray
    det_score: float
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    @property
    def is_frontal(self) -> bool:
        return ProfileInsightFace.is_frontal(self.yaw, self.pitch)


class ProfileInsightFace:
    """Lazy-loaded InsightFace detector for YuNet fallback."""

    def __init__(self, det_size: Tuple[int, int] = (640, 640)):
        self._det_size = det_size
        self._app = None
        self._available = None

    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                import insightface  # noqa: F401
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def _ensure_app(self):
        if self._app is not None:
            return
        if not self.available:
            raise RuntimeError("insightface is not installed")
        from insightface.app import FaceAnalysis

        # 実行デバイスを選択する。
        #   SHIGURE_INSIGHTFACE_DEVICE=cpu/gpu で明示指定可。未指定なら CUDA が使えれば GPU、
        #   無ければ CPU に自動フォールバックする（GPU 必須ではない）。
        device = os.environ.get('SHIGURE_INSIGHTFACE_DEVICE', 'auto').lower()
        use_gpu = False
        if device == 'gpu':
            use_gpu = True
        elif device == 'cpu':
            use_gpu = False
        else:
            try:
                import onnxruntime as ort
                use_gpu = 'CUDAExecutionProvider' in ort.get_available_providers()
            except Exception:
                use_gpu = False

        if use_gpu:
            self._app = FaceAnalysis(
                name='buffalo_s',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
            )
            self._app.prepare(ctx_id=0, det_size=self._det_size)
        else:
            self._app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
            self._app.prepare(ctx_id=-1, det_size=self._det_size)

    @staticmethod
    def is_frontal(
        yaw: float,
        pitch: float = 0.0,
        yaw_threshold: float = FRONTAL_YAW_THRESHOLD,
        pitch_threshold: float = FRONTAL_PITCH_THRESHOLD,
    ) -> bool:
        return abs(yaw) < yaw_threshold and abs(pitch) < pitch_threshold

    @staticmethod
    def is_point_in_box(point: Tuple[float, float], box: Tuple[int, int, int, int]) -> bool:
        x, y = point
        box_x, box_y, box_width, box_height = box
        return box_x <= x <= box_x + box_width and box_y <= y <= box_y + box_height

    def detect_faces(self, image: np.ndarray) -> List[InsightFaceResult]:
        """全画面で顔検出し、bbox と embedding の一覧を返す（YuNet と同じ起点）。"""
        if image is None or image.size == 0:
            return []

        self._ensure_app()
        faces = self._app.get(image)
        results: List[InsightFaceResult] = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            if w < 40 or h < 40:
                continue
            embedding = np.asarray(face.embedding, dtype=np.float32).reshape(-1)
            det_score = float(getattr(face, "det_score", 0.0))
            pitch, yaw, roll = 0.0, 0.0, 0.0
            pose = getattr(face, "pose", None)
            if pose is not None and len(pose) >= 3:
                pitch, yaw, roll = float(pose[0]), float(pose[1]), float(pose[2])
            results.append(InsightFaceResult(
                bbox=(x1, y1, w, h),
                embedding=embedding,
                det_score=det_score,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
            ))
        return results

    def find_face_for_head(
        self,
        faces: List[InsightFaceResult],
        head_point,
    ) -> Optional[InsightFaceResult]:
        """
        検出済みの顔 bbox 一覧から、頭部座標が bbox 内にある顔を返す（YuNet 側の紐付けと同じ）。
        複数候補がある場合は det_score が最も高い顔を採用する。
        """
        head = (head_point.x, head_point.y)
        best: Optional[InsightFaceResult] = None
        best_score = -1.0
        for candidate in faces:
            if not self.is_point_in_box(head, candidate.bbox):
                continue
            if candidate.det_score > best_score:
                best = candidate
                best_score = candidate.det_score
        return best
