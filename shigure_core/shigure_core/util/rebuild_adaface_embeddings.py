"""既存 face_models の jpg から AdaFace 埋め込みを再生成する.

ArcFace 時代の .npy は空間が違うため照合できない。対応する .jpg が
ある番号だけ検出し直して上書きし、jpg 無しの .npy は削除する。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from shigure_core.nodes.face_analyzer import FaceAnalyzer
from shigure_core.util.face_models_dir import get_face_models_dir
from shigure_core.util.pca_model import build_pca_model, default_pca_model_path

CROP_PAD_RATIO = 0.3
REBUILD_MIN_SIDE = 256
REBUILD_DET_THRESH = 0.2


@dataclass
class RebuildStats:
    """再生成の件数集計."""

    saved: int = 0
    skipped: int = 0
    removed: int = 0
    users: int = 0
    messages: List[str] = field(default_factory=list)


def pad_face_crop(image: np.ndarray, ratio: float = CROP_PAD_RATIO) -> np.ndarray:
    """タイトな顔クロップに余白を足して検出を安定させる."""
    height, width = image.shape[:2]
    pad_y = max(1, int(height * ratio))
    pad_x = max(1, int(width * ratio))
    return cv2.copyMakeBorder(
        image,
        pad_y,
        pad_y,
        pad_x,
        pad_x,
        cv2.BORDER_REPLICATE,
    )


def prepare_crop_for_detection(image: np.ndarray) -> np.ndarray:
    """小さい顔クロップを拡大して余白を足し、SCRFD が拾いやすくする."""
    height, width = image.shape[:2]
    min_side = min(height, width)
    if min_side < REBUILD_MIN_SIDE:
        scale = REBUILD_MIN_SIDE / max(min_side, 1)
        image = cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_CUBIC,
        )
    return pad_face_crop(image)


def detect_best_embedding(
        detector: FaceAnalyzer,
        image: np.ndarray) -> Optional[np.ndarray]:
    """クロップ画像から最も det_score の高い顔の埋め込みを返す."""
    candidates = []
    for source in (prepare_crop_for_detection(image), image):
        faces = detector.detect_faces(
            source,
            min_size=1,
            det_thresh=REBUILD_DET_THRESH,
        )
        if faces:
            candidates.extend(faces)
    if candidates:
        best = max(candidates, key=lambda face: face.det_score)
        return best.embedding
    # 検出不能なタイトクロップは未整列 112x112 でフォールバックする
    return detector.embed_bgr_crop(image)


def _iter_user_dirs(face_models_dir: Path) -> List[Path]:
    if not face_models_dir.is_dir():
        return []
    return sorted(
        path for path in face_models_dir.glob('user_*') if path.is_dir()
    )


def rebuild_directory(
        detector: FaceAnalyzer,
        target_dir: Path,
        stats: RebuildStats,
        *,
        dry_run: bool,
        delete_orphans: bool) -> None:
    """1ディレクトリ（正面 or profile/）の jpg から npy を作り直す."""
    jpgs = sorted(target_dir.glob('*.jpg'))
    npys = {path.stem: path for path in target_dir.glob('*.npy')}
    used_stems = set()

    for jpg_path in jpgs:
        used_stems.add(jpg_path.stem)
        image = cv2.imread(str(jpg_path))
        if image is None:
            stats.skipped += 1
            stats.messages.append(f'skip unreadable: {jpg_path}')
            continue
        embedding = detect_best_embedding(detector, image)
        npy_path = target_dir / f'{jpg_path.stem}.npy'
        if embedding is None:
            stats.skipped += 1
            stats.messages.append(f'skip no-face: {jpg_path}')
            if delete_orphans and npy_path.exists() and not dry_run:
                npy_path.unlink()
                stats.removed += 1
            continue
        if not dry_run:
            np.save(str(npy_path), embedding)
        stats.saved += 1

    if not delete_orphans:
        return
    for stem, npy_path in npys.items():
        if stem in used_stems:
            continue
        stats.messages.append(f'remove orphan npy: {npy_path}')
        if not dry_run:
            npy_path.unlink()
        stats.removed += 1


def rebuild_face_models(
        face_models_dir: Path,
        *,
        dry_run: bool = False,
        include_profile: bool = True,
        delete_orphans: bool = True,
        rebuild_pca: bool = True) -> RebuildStats:
    """face_models 配下の全ユーザーを AdaFace 埋め込みへ作り直す."""
    stats = RebuildStats()
    detector = FaceAnalyzer()
    if not detector.available:
        raise RuntimeError('insightface or onnxruntime is not installed')

    user_dirs = _iter_user_dirs(face_models_dir)
    stats.users = len(user_dirs)
    for user_dir in user_dirs:
        rebuild_directory(
            detector,
            user_dir,
            stats,
            dry_run=dry_run,
            delete_orphans=delete_orphans,
        )
        profile_dir = user_dir / 'profile'
        if include_profile and profile_dir.is_dir():
            rebuild_directory(
                detector,
                profile_dir,
                stats,
                dry_run=dry_run,
                delete_orphans=delete_orphans,
            )

    if rebuild_pca and not dry_run:
        result = build_pca_model(
            face_models_dir,
            default_pca_model_path(face_models_dir),
        )
        stats.messages.append(result.message)
    return stats


def main() -> None:
    """コマンドラインから顔辞書を AdaFace 埋め込みへ作り直す."""
    parser = argparse.ArgumentParser(
        description='Rebuild face_models/*.npy with AdaFace embeddings',
    )
    parser.add_argument(
        '--face-models-dir',
        type=Path,
        default=get_face_models_dir(),
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='書き込みせず検出可否だけ確認する',
    )
    parser.add_argument(
        '--no-profile',
        action='store_true',
        help='横顔(profile/)は再生成しない',
    )
    parser.add_argument(
        '--keep-orphans',
        action='store_true',
        help='jpg が無い / 検出失敗の .npy を残す',
    )
    parser.add_argument(
        '--skip-pca',
        action='store_true',
        help='pca_model.pkl を再学習しない',
    )
    args = parser.parse_args()

    stats = rebuild_face_models(
        args.face_models_dir,
        dry_run=args.dry_run,
        include_profile=not args.no_profile,
        delete_orphans=not args.keep_orphans,
        rebuild_pca=not args.skip_pca,
    )
    print(
        f'users={stats.users} saved={stats.saved} '
        f'skipped={stats.skipped} removed={stats.removed}'
    )
    for message in stats.messages:
        print(message)


if __name__ == '__main__':
    main()
