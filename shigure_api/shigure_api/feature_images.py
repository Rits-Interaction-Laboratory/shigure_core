"""Resolve feature_num to saved face crop images under face_models/."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

FACE_THUMBNAIL_SIZE = 128


def feature_num_from_stem(user_id: str, file_stem: str) -> Optional[int]:
    """
    ファイル名(stem)から feature_num を取り出す.

    2種類の命名に対応する:
      - 自動登録(people_recognition): '{user_id}_{num}'  例 'user_new5_12628'
      - 手動登録(node_face_models):   '{name}{num}'       例 'nakamura12'
        （user_id='user_nakamura' に対し 'user_' を除いた 'nakamura' + 数字）
    どちらにも合わなければ None。
    """
    # 1) 自動登録形式 '{user_id}_{num}'
    prefix = f'{user_id}_'
    if file_stem.startswith(prefix):
        suffix = file_stem[len(prefix):]
        if suffix.isdigit():
            return int(suffix)
    # 2) 手動登録形式 '{name}{num}'（user_id から 'user_' を外した name + 末尾数字）
    name = user_id[len('user_'):] if user_id.startswith('user_') else user_id
    if name and file_stem.startswith(name):
        suffix = file_stem[len(name):]
        if suffix.isdigit():
            return int(suffix)
    return None


def find_face_image_path(
    face_models_dir: Path, user_id: str, feature_num: int
) -> Optional[Path]:
    """
    face_models/{user_id}/ から feature_num に対応する顔画像(.jpg)を返す.

    2種類の命名に対応: '{user_id}_{num}.jpg'（自動登録）/ '{name}{num}.jpg'（手動登録）。
    """
    if not face_models_dir.is_dir() or not user_id.startswith('user_'):
        return None
    name = user_id[len('user_'):]
    candidates = [
        face_models_dir / user_id / f'{user_id}_{feature_num}.jpg',  # 自動登録形式
        face_models_dir / user_id / f'{name}{feature_num}.jpg',      # 手動登録形式
    ]
    for path in candidates:
        if path.is_file() and feature_num_from_stem(user_id, path.stem) == feature_num:
            return path
    return None


def load_face_thumbnail(path: Path, size: int = FACE_THUMBNAIL_SIZE) -> bytes:
    """Resize face crop to a square JPEG thumbnail."""
    from PIL import Image

    with Image.open(path) as img:
        rgb = img.convert('RGB')
        rgb.thumbnail((size, size))
        buf = io.BytesIO()
        rgb.save(buf, format='JPEG', quality=85)
        return buf.getvalue()
