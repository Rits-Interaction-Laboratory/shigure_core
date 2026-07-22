"""顔モデルの保存先を全ノードで共通化するユーティリティ."""

import os
from pathlib import Path
from typing import Tuple


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
