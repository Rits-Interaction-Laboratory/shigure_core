"""Resolve feature_num to saved face crop images under face_models/."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

FACE_THUMBNAIL_SIZE = 128


def feature_num_from_stem(user_id: str, file_stem: str) -> Optional[int]:
    """Parse feature_num from e.g. user_new5_12628 (user_id + feature_num)."""
    prefix = f'{user_id}_'
    if not file_stem.startswith(prefix):
        return None
    suffix = file_stem[len(prefix) :]
    if suffix.isdigit():
        return int(suffix)
    return None


def find_face_image_path(
    face_models_dir: Path, user_id: str, feature_num: int
) -> Optional[Path]:
    """Return user_*_{feature_num}.jpg under face_models/{user_id}/."""
    if not face_models_dir.is_dir() or not user_id.startswith('user_'):
        return None
    path = face_models_dir / user_id / f'{user_id}_{feature_num}.jpg'
    if not path.is_file():
        return None
    if feature_num_from_stem(user_id, path.stem) != feature_num:
        return None
    return path


def load_face_thumbnail(path: Path, size: int = FACE_THUMBNAIL_SIZE) -> bytes:
    """Resize face crop to a square JPEG thumbnail."""
    from PIL import Image

    with Image.open(path) as img:
        rgb = img.convert('RGB')
        rgb.thumbnail((size, size))
        buf = io.BytesIO()
        rgb.save(buf, format='JPEG', quality=85)
        return buf.getvalue()
