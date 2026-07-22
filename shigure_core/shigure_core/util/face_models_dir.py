"""顔モデルの保存先を全ノードで共通化するユーティリティ."""

from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

# 正面: '{任意接頭辞}_{連番}' / 横顔: '{任意接頭辞}_profile_{連番}'
_FRONTAL_STEM_RE = re.compile(r'^(?P<prefix>.+)_(?P<idx>\d+)$')
_PROFILE_STEM_RE = re.compile(r'^(?P<prefix>.+)_profile_(?P<idx>\d+)$')


def get_face_models_dir() -> Path:
    """環境変数、またはソース内の既定ディレクトリから保存先を返す."""
    configured_dir = os.environ.get('SHIGURE_FACE_MODELS_DIR')
    if configured_dir:
        return Path(configured_dir).expanduser()

    module_root = Path(__file__).resolve().parent.parent
    return module_root / 'nodes' / 'face_models'


def face_models_signature(face_models_dir: Path | str | None = None) -> Tuple:
    """face_models 配下の user_* 構成を表すシグネチャを返す.

    ディレクトリ削除・追加・中の .npy 増減を検知するために使う。
    """
    base = Path(face_models_dir) if face_models_dir is not None else get_face_models_dir()
    if not base.is_dir():
        return ()

    entries = []
    for user_dir in sorted(base.glob('user_*')):
        if not user_dir.is_dir():
            continue
        frontal = list(user_dir.glob('*.npy'))
        profile_dir = user_dir / 'profile'
        profile = list(profile_dir.glob('*.npy')) if profile_dir.is_dir() else []
        mtimes = []
        for path in frontal + profile:
            try:
                mtimes.append(path.stat().st_mtime_ns)
            except OSError:
                continue
        try:
            dir_mtime = user_dir.stat().st_mtime_ns
        except OSError:
            dir_mtime = 0
        entries.append(
            (
                user_dir.name,
                len(frontal),
                len(profile),
                max(mtimes) if mtimes else dir_mtime,
            )
        )
    return tuple(entries)


def _sync_dir_file_prefixes(dir_path: Path, user_id: str, *, profile: bool) -> int:
    """1ディレクトリ内のファイル接頭辞を user_id に揃える.

    :return: リネームしたファイル数
    """
    if not dir_path.is_dir():
        return 0

    by_stem: Dict[str, List[Path]] = defaultdict(list)
    for path in dir_path.iterdir():
        if path.is_file():
            by_stem[path.stem].append(path)

    occupied: Set[int] = set()
    mismatched: List[Tuple[str, int, List[Path]]] = []
    pattern = _PROFILE_STEM_RE if profile else _FRONTAL_STEM_RE

    for stem, paths in by_stem.items():
        matched = pattern.match(stem)
        if matched is None:
            continue
        idx = int(matched.group('idx'))
        prefix = matched.group('prefix')
        if prefix == user_id:
            occupied.add(idx)
            continue
        mismatched.append((stem, idx, paths))

    if not mismatched:
        return 0

    next_idx = (max(occupied) + 1) if occupied else 1
    renamed = 0
    for stem, old_idx, paths in sorted(mismatched, key=lambda item: item[1]):
        if old_idx not in occupied:
            new_idx = old_idx
        else:
            while next_idx in occupied:
                next_idx += 1
            new_idx = next_idx
            next_idx += 1
        occupied.add(new_idx)

        new_stem = (
            f'{user_id}_profile_{new_idx}' if profile else f'{user_id}_{new_idx}'
        )
        if new_stem == stem:
            continue

        for path in paths:
            dest = path.with_name(new_stem + path.suffix)
            if dest.exists():
                # 占有管理の不整合時はスキップしてデータ破壊を避ける
                continue
            path.rename(dest)
            renamed += 1
    return renamed


def sync_user_file_prefixes(user_dir: Path | str) -> int:
    """ユーザーディレクトリ名に合わせて特徴・画像ファイル名の接頭辞を更新する.

    例: face_models/user_aono/user_new1_1.npy → user_aono_1.npy
    同名衝突時は既存の最大連番の続きへ振り直す。
    profile/ 配下の横顔ファイルも同様に処理する。

    :return: リネームしたファイル数
    """
    user_path = Path(user_dir)
    if not user_path.is_dir():
        return 0
    user_id = user_path.name
    if not user_id.startswith('user_'):
        return 0

    renamed = _sync_dir_file_prefixes(user_path, user_id, profile=False)
    renamed += _sync_dir_file_prefixes(user_path / 'profile', user_id, profile=True)
    return renamed


def sync_all_user_file_prefixes(face_models_dir: Path | str | None = None) -> int:
    """face_models 配下の全 user_* についてファイル接頭辞をディレクトリ名に揃える.

    ディレクトリを手動リネームしたあと、中の user_newN_* が古いまま残るのを解消する。

    :return: リネームしたファイル総数
    """
    base = Path(face_models_dir) if face_models_dir is not None else get_face_models_dir()
    if not base.is_dir():
        return 0
    total = 0
    for user_dir in sorted(base.glob('user_*')):
        if user_dir.is_dir():
            total += sync_user_file_prefixes(user_dir)
    return total
