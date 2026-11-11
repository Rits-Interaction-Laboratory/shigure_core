"""PCA の重い処理（辞書再読込・再学習・フル state 配信）を ROS コールバックから切り離す."""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

from shigure_api.config import FACE_MODELS_DIR, PCA_REBUILD_ON_NEW_USER
from shigure_api.perf_confirm import perf_log
from shigure_core.util.face_models_dir import face_models_signature, sync_all_user_file_prefixes


class PcaHeavyWorker:
    """バックグラウンドで PCA reload / rebuild / フル state 配信を直列実行する."""

    _COALESCE_KINDS = frozenset({'reload_publish', 'rebuild_publish', 'publish_state'})

    def __init__(
        self,
        *,
        pca_builder: Any,
        publish_state: Callable[[], None],
        on_error: Callable[[str, Exception], None],
    ) -> None:
        self._pca_builder = pca_builder
        self._publish_state = publish_state
        self._on_error = on_error
        self._queue: queue.Queue[Optional[Tuple[str, Dict[str, Any]]]] = queue.Queue()
        self._stop = threading.Event()
        self._face_models_signature = face_models_signature(FACE_MODELS_DIR)
        self._thread = threading.Thread(
            target=self._run,
            name='shigure_api_pca_worker',
            daemon=True,
        )
        self._thread.start()

    @property
    def face_models_signature(self) -> Any:
        return self._face_models_signature

    @face_models_signature.setter
    def face_models_signature(self, value: Any) -> None:
        self._face_models_signature = value

    def enqueue(self, kind: str, **kwargs: Any) -> None:
        """重い PCA ジョブを投入する。同種の未処理ジョブは最新1件に畳む."""
        if kind in self._COALESCE_KINDS:
            pending: list[Tuple[str, Dict[str, Any]]] = []
            try:
                while True:
                    item = self._queue.get_nowait()
                    if item is None:
                        pending.append(item)
                        continue
                    if item[0] not in self._COALESCE_KINDS:
                        pending.append(item)
            except queue.Empty:
                pass
            for item in pending:
                self._queue.put(item)
        self._queue.put((kind, kwargs))

    def shutdown(self, timeout_sec: float = 2.0) -> None:
        self._stop.set()
        self._queue.put(None)
        self._thread.join(timeout=timeout_sec)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                continue
            kind, kwargs = item
            t0 = time.perf_counter()
            try:
                self._handle(kind, kwargs)
            except Exception as exc:
                self._on_error(kind, exc)
            else:
                perf_log(
                    'pca_worker_done',
                    worker_kind=kind,
                    ms=(time.perf_counter() - t0) * 1000.0,
                )

    def _handle(self, kind: str, kwargs: Dict[str, Any]) -> None:
        if self._pca_builder is None:
            return
        if kind == 'reload_publish':
            self._pca_builder.reload_dictionary_from_disk()
            self._face_models_signature = face_models_signature(FACE_MODELS_DIR)
            self._publish_state()
        elif kind == 'rebuild_publish':
            registered_user_id = kwargs.get('registered_user_id')
            if PCA_REBUILD_ON_NEW_USER:
                self._pca_builder.rebuild_pca_from_disk(
                    registered_user_id=registered_user_id,
                )
            self._face_models_signature = face_models_signature(FACE_MODELS_DIR)
            self._publish_state()
        elif kind == 'publish_state':
            self._publish_state()
        elif kind == 'sync_reload_publish':
            renamed = sync_all_user_file_prefixes(FACE_MODELS_DIR)
            if renamed:
                self._face_models_signature = face_models_signature(FACE_MODELS_DIR)
            self._pca_builder.reload_dictionary_from_disk()
            self._face_models_signature = face_models_signature(FACE_MODELS_DIR)
            self._publish_state()
        else:
            raise ValueError(f'unknown PCA worker kind: {kind}')

    def check_disk_signature_changed(self) -> bool:
        """ROS スレッド側で face_models の変化だけを軽く判定する."""
        signature = face_models_signature(FACE_MODELS_DIR)
        return signature != self._face_models_signature
