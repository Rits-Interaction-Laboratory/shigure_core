"""AdaFace ONNX による顔埋め込み抽出.

検出・姿勢推定は InsightFace 側に任せ、認識だけを差し替える。
公式 AdaFace と同じ前処理（BGR・112x112・mean/std=127.5）で推論する。
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# yakhyo/adaface-onnx が公式重みから書き出した ONNX。
# ir18: 軽量（リアルタイム向け）。ir101: 高精度（GPU 推奨）。
ADAFACE_MODELS: Dict[str, Dict[str, str]] = {
    'ir18': {
        'filename': 'adaface_ir_18.onnx',
        'url': (
            'https://github.com/yakhyo/adaface-onnx/releases/download/'
            'weights/adaface_ir_18.onnx'
        ),
    },
    'ir101': {
        'filename': 'adaface_ir_101.onnx',
        'url': (
            'https://github.com/yakhyo/adaface-onnx/releases/download/'
            'weights/adaface_ir_101.onnx'
        ),
    },
}
DEFAULT_ADAFACE_MODEL = 'ir18'
ADAFACE_INPUT_SIZE = (112, 112)
ADAFACE_MEAN = 127.5
ADAFACE_STD = 127.5


def resolve_adaface_model_name() -> str:
    """環境変数 SHIGURE_ADAFACE_MODEL からモデル名を返す."""
    raw = os.environ.get('SHIGURE_ADAFACE_MODEL', DEFAULT_ADAFACE_MODEL)
    name = str(raw).strip().lower().replace('_', '')
    aliases = {
        'ir18': 'ir18',
        '18': 'ir18',
        'ir101': 'ir101',
        '101': 'ir101',
    }
    if name not in aliases:
        supported = ', '.join(sorted(ADAFACE_MODELS.keys()))
        raise RuntimeError(
            f'Unknown SHIGURE_ADAFACE_MODEL={raw!r}. '
            f'Supported: {supported}'
        )
    return aliases[name]


def default_adaface_model_dir() -> Path:
    """ONNX の既定キャッシュ先を返す.

    ~/.insightface 配下に置く。Docker の ~/.shigure マウントで消えないようにする。
    """
    return Path.home() / '.insightface' / 'models' / 'adaface'


def resolve_adaface_model_path(model_name: Optional[str] = None) -> Path:
    """ダウンロード済み（またはこれから置く）ONNX パスを返す."""
    configured = os.environ.get('SHIGURE_ADAFACE_MODEL_PATH')
    if configured:
        return Path(configured).expanduser()
    name = model_name or resolve_adaface_model_name()
    filename = ADAFACE_MODELS[name]['filename']
    return default_adaface_model_dir() / filename


def ensure_adaface_model(model_path: Optional[Path] = None) -> Path:
    """ONNX が無ければ GitHub Releases から取得する."""
    path = model_path or resolve_adaface_model_path()
    if path.is_file() and path.stat().st_size > 0:
        return path

    name = resolve_adaface_model_name()
    url = ADAFACE_MODELS[name]['url']
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.download')
    logger.info('Downloading AdaFace model: %s -> %s', url, path)
    try:
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            f'Failed to download AdaFace model from {url}'
        ) from None
    return path


def ensure_onnx_cuda_lib_path() -> List[str]:
    """pip の nvidia-* パッケージ配下の lib を LD_LIBRARY_PATH へ通す.

    onnxruntime-gpu は libcublasLt.so.12 等を
    site-packages/nvidia/*/lib に置くが、未 export だと
    CUDAExecutionProvider のロードに失敗して CPU へ落ちる。
    InferenceSession 作成前に呼ぶこと。
    """
    try:
        import site
    except ImportError:
        return []

    candidates: List[Path] = []
    site_dirs = []
    try:
        site_dirs.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        user_site = site.getusersitepackages()
        if user_site:
            site_dirs.append(user_site)
    except Exception:
        pass
    # 一部環境では sys.path 上にしか nvidia が無い
    import sys
    site_dirs.extend(sys.path)

    seen = set()
    for sp in site_dirs:
        if not sp:
            continue
        nvidia_root = Path(sp) / 'nvidia'
        if not nvidia_root.is_dir():
            continue
        for lib_dir in sorted(nvidia_root.glob('*/lib')):
            key = str(lib_dir.resolve()) if lib_dir.exists() else str(lib_dir)
            if key in seen or not lib_dir.is_dir():
                continue
            seen.add(key)
            candidates.append(lib_dir)

    if not candidates:
        return []

    added = [str(p) for p in candidates]
    current = os.environ.get('LD_LIBRARY_PATH', '')
    current_parts = [p for p in current.split(':') if p]
    # 既存より前に挿し、今回追加分を優先する
    merged = added + [p for p in current_parts if p not in added]
    os.environ['LD_LIBRARY_PATH'] = ':'.join(merged)
    return added


def select_onnx_providers() -> Tuple[List[str], bool]:
    """ONNX Runtime の実行プロバイダと GPU 使用可否を返す.

    SHIGURE_INSIGHTFACE_DEVICE=cpu/gpu で明示指定可。未指定なら CUDA が
    使えれば GPU、無ければ CPU にフォールバックする。
    """
    device = os.environ.get('SHIGURE_INSIGHTFACE_DEVICE', 'auto').lower()
    use_gpu = False
    if device == 'cpu':
        return ['CPUExecutionProvider'], False

    # CUDA EP ロード前に nvidia pip の .so パスを通す
    added = ensure_onnx_cuda_lib_path()
    if added:
        logger.info('ONNX CUDA libs prepended to LD_LIBRARY_PATH (%d dirs)', len(added))

    if device == 'gpu':
        use_gpu = True
    else:
        try:
            import onnxruntime as ort
            use_gpu = 'CUDAExecutionProvider' in ort.get_available_providers()
        except Exception:
            use_gpu = False

    if use_gpu:
        return ['CUDAExecutionProvider', 'CPUExecutionProvider'], True
    if device == 'gpu':
        logger.warning(
            'SHIGURE_INSIGHTFACE_DEVICE=gpu だが CUDAExecutionProvider が'
            '使えないため CPU にフォールバックする'
        )
    return ['CPUExecutionProvider'], False


class AdaFaceEmbedder:
    """AdaFace ONNX セッション. 整列済み BGR 112x112 から 512 次元埋め込みを返す."""

    def __init__(self, model_path: Optional[Path] = None):
        """ONNX パスを保持する。セッションは初回推論時に開く."""
        self._session = None
        self._input_name = None
        self._output_name = None
        self._model_path = model_path
        self._providers: Optional[List[str]] = None
        self.use_gpu = False

    def _ensure_session(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort

        path = ensure_adaface_model(self._model_path)
        providers, use_gpu = select_onnx_providers()
        self.use_gpu = use_gpu
        self._providers = providers
        self._session = ort.InferenceSession(str(path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        logger.info(
            'AdaFace loaded: path=%s providers=%s',
            path,
            self._session.get_providers(),
        )

    @property
    def available(self) -> bool:
        """推論ランタイムが import できれば True."""
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return False
        return True

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """整列済み BGR 画像から L2 正規化埋め込みを返す."""
        if aligned_bgr is None or aligned_bgr.size == 0:
            raise ValueError('aligned_bgr is empty')
        self._ensure_session()
        height, width = aligned_bgr.shape[:2]
        if (width, height) != ADAFACE_INPUT_SIZE:
            aligned_bgr = cv2.resize(aligned_bgr, ADAFACE_INPUT_SIZE)
        blob = cv2.dnn.blobFromImage(
            aligned_bgr,
            scalefactor=1.0 / ADAFACE_STD,
            size=ADAFACE_INPUT_SIZE,
            mean=(ADAFACE_MEAN, ADAFACE_MEAN, ADAFACE_MEAN),
            swapRB=False,
        )
        raw = self._session.run(
            [self._output_name],
            {self._input_name: blob},
        )[0]
        vec = np.asarray(raw, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)
