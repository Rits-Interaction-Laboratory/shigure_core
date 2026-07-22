"""顔モデルの保存先を全ノードで共通化するユーティリティ."""

import os
from pathlib import Path


def get_face_models_dir() -> Path:
    """環境変数、またはソース内の既定ディレクトリから保存先を返す."""
    configured_dir = os.environ.get('SHIGURE_FACE_MODELS_DIR')
    if configured_dir:
        return Path(configured_dir).expanduser()

    module_root = Path(__file__).resolve().parent.parent
    return module_root / 'nodes' / 'face_models'
