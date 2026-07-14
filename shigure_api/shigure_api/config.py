"""Configuration for shigure_api."""

from __future__ import annotations

import os
from pathlib import Path


def _default_face_models_dir() -> Path:
    # SHIGURE_FACE_MODELS_DIR 環境変数があれば最優先。
    # 無ければ固定の永続パス ~/.shigure/face_models（shigure_core 側ノードの既定と一致させること）。
    env = os.environ.get('SHIGURE_FACE_MODELS_DIR')
    if env:
        return Path(env).expanduser()
    return Path('~/.shigure/face_models').expanduser()


FACE_MODELS_DIR = _default_face_models_dir()

RECOGNITION_SCORE_THRESHOLD = float(os.environ.get('SHIGURE_RECOGNITION_SCORE_THRESHOLD', '0.4'))
RECOGNITION_DEBOUNCE_SEC = float(os.environ.get('SHIGURE_RECOGNITION_DEBOUNCE_SEC', '30'))

API_HOST = os.environ.get('SHIGURE_API_HOST', '0.0.0.0')
API_PORT = int(os.environ.get('SHIGURE_API_PORT', '8765'))
API_KEY = os.environ.get('SHIGURE_API_KEY', '')

DICTIONARY_UPDATE_TOPIC = os.environ.get('SHIGURE_DICTIONARY_UPDATE_TOPIC', '/dictionary_update')
FACE_RECOGNITION_TOPIC = os.environ.get('SHIGURE_FACE_RECOGNITION_TOPIC', '/face_recognition/results')
FEATURE_INFO_TOPIC = os.environ.get('SHIGURE_FEATURE_INFO_TOPIC', '/feature_info')
RECOGNITION_HISTORY_TOPIC = os.environ.get(
    'SHIGURE_RECOGNITION_HISTORY_TOPIC', '/shigure/recognition_history'
)
TRACKING_DEBUG_IMAGE_TOPIC = os.environ.get(
    'SHIGURE_TRACKING_DEBUG_IMAGE_TOPIC', '/shigure/tracking_debug_image'
)

PCA_REDRAW_EVERY = int(os.environ.get('SHIGURE_PCA_REDRAW_EVERY', '5'))
PCA_TRAJECTORY_MAX_POINTS = int(os.environ.get('SHIGURE_PCA_TRAJECTORY_MAX_POINTS', '50'))
PCA_REBUILD_ON_NEW_USER = os.environ.get('SHIGURE_PCA_REBUILD_ON_NEW_USER', 'true').lower() in (
    '1',
    'true',
    'yes',
)


def default_pca_model_path(face_models_dir: Path | None = None) -> Path:
    env = os.environ.get('SHIGURE_PCA_MODEL_PATH')
    if env:
        return Path(env).expanduser()
    base = face_models_dir or FACE_MODELS_DIR
    return base / 'pca_model.pkl'
