"""FastAPI application: WebSocket events + REST user list."""

from __future__ import annotations

import asyncio
import queue
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, List, Optional, Set

from fastapi import FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from shigure_api.config import (
    API_KEY,
    FACE_MODELS_DIR,
    PCA_REDRAW_EVERY,
    PCA_SHOW_FULL_DICTIONARY,
    PCA_TRAJECTORY_MAX_POINTS,
    default_pca_model_path,
)
from shigure_api.events import ContactCandidate, UserEvent, UserSummary, contact_row_to_model, user_event_to_dict

from shigure_api.feature_images import find_face_image_path, load_face_thumbnail
from shigure_api.pca_plot import PcaPlotHub, PcaPlotStateBuilder, state_to_dict
from shigure_api.tracking_debug import TrackingDebugHub


class EventHub:
    """Broadcast UserEvent to all connected WebSocket clients."""

    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._thread_queue: queue.Queue[object] = queue.Queue()
        self._history: List[UserEvent] = []
        self._history_limit = 100
        # 直近の累積スコア配信（cumulative_scores）。新規接続時のスナップショット用。
        self._latest_recognition_scores: Optional[dict] = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        asyncio.create_task(self._broadcast_loop())

    def enqueue(self, event: UserEvent) -> None:
        """Thread-safe enqueue from ROS callback thread."""
        self._thread_queue.put_nowait(event)

    def enqueue_recognition_scores(self, payload: dict) -> None:
        """Thread-safe: RecognitionHistory 由来の累積スコア配信を投入する（ROSスレッドから）。"""
        self._thread_queue.put_nowait(payload)

    async def handle_event(self, event: UserEvent) -> List[dict]:
        self._history.append(event)
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit:]
        return [user_event_to_dict(event)]

    async def broadcast(self, payloads: List[dict]) -> None:
        async with self._lock:
            dead: List[WebSocket] = []
            for ws in self._clients:
                try:
                    for payload in payloads:
                        await ws.send_json(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)

    async def _broadcast_loop(self) -> None:
        while True:
            item = await asyncio.to_thread(self._thread_queue.get)
            if isinstance(item, dict):
                # ROS由来の完成済みペイロード（cumulative_scores 等）はそのまま配信する。
                if item.get('type') == 'cumulative_scores':
                    self._latest_recognition_scores = item
                payloads = [item]
            else:
                payloads = await self.handle_event(item)
            if payloads:
                await self.broadcast(payloads)

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._clients.add(websocket)
        for item in self.recent_events(limit=20):
            await websocket.send_json(item)
        if self._latest_recognition_scores is not None:
            await websocket.send_json(self._latest_recognition_scores)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(websocket)

    def recent_events(self, limit: int = 50) -> List[dict]:
        return [user_event_to_dict(e) for e in self._history[-limit:]]


hub = EventHub()
pca_hub = PcaPlotHub()
tracking_debug_hub = TrackingDebugHub()
pca_builder: PcaPlotStateBuilder | None = None


def init_pca_builder(face_models_dir: Path | None = None) -> PcaPlotStateBuilder:
    global pca_builder
    base = face_models_dir or FACE_MODELS_DIR
    pca_builder = PcaPlotStateBuilder(
        base,
        default_pca_model_path(base),
        redraw_every=PCA_REDRAW_EVERY,
        trajectory_max_points=PCA_TRAJECTORY_MAX_POINTS,
        show_full_dictionary=PCA_SHOW_FULL_DICTIONARY,
    )
    return pca_builder


def _resolve_api_key(
    x_api_key: str | None = None,
    api_key: str | None = None,
) -> str | None:
    return x_api_key or api_key


def _verify_api_key(
    x_api_key: str | None = None,
    api_key: str | None = None,
) -> None:
    if not API_KEY:
        return
    if _resolve_api_key(x_api_key, api_key) != API_KEY:
        raise HTTPException(status_code=401, detail='Invalid API key')


def _scan_users(face_models_dir: Path) -> List[UserSummary]:
    if not face_models_dir.is_dir():
        return []
    users: List[UserSummary] = []
    for user_dir in sorted(face_models_dir.glob('user_*')):
        if not user_dir.is_dir():
            continue
        user_id = user_dir.name
        frontal = len(list(user_dir.glob('*.npy')))
        profile_dir = user_dir / 'profile'
        profile = len(list(profile_dir.glob('*.npy'))) if profile_dir.is_dir() else 0
        users.append(
            UserSummary(
                user_id=user_id,
                feature_count=frontal,
                profile_feature_count=profile,
            )
        )
    return users


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await hub.start()
    await pca_hub.start()
    await tracking_debug_hub.start()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title='Shigure API',
        description='Face recognition events for iOS / React Native clients',
        version='0.1.0',
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    @app.get('/health')
    def health() -> dict:
        return {
            'status': 'ok',
            'face_models_dir': str(FACE_MODELS_DIR),
            'face_models_exists': FACE_MODELS_DIR.is_dir(),
            'pca_show_full_dictionary': (
                pca_builder.show_full_dictionary
                if pca_builder is not None
                else PCA_SHOW_FULL_DICTIONARY
            ),
        }

    @app.get('/pca/show_full_dictionary')
    def get_show_full_dictionary(
        x_api_key: str | None = Header(default=None, alias='X-API-Key'),
    ) -> dict:
        """辞書全特徴の API 配信モードの現在値を返す."""
        _verify_api_key(x_api_key)
        if pca_builder is None:
            raise HTTPException(status_code=503, detail='PCA builder not initialized')
        return {'show_full_dictionary': pca_builder.show_full_dictionary}

    @app.put('/pca/show_full_dictionary')
    def put_show_full_dictionary(
        enabled: bool = Query(..., description='true で辞書の全特徴を PCA API に含める'),
        x_api_key: str | None = Header(default=None, alias='X-API-Key'),
    ) -> dict:
        """辞書全特徴の API 配信モードを切り替えて、最新 state を再配信する."""
        _verify_api_key(x_api_key)
        if pca_builder is None:
            raise HTTPException(status_code=503, detail='PCA builder not initialized')
        pca_builder.set_show_full_dictionary(enabled)
        pca_hub.enqueue(state_to_dict(pca_builder.build_state()))
        return {'show_full_dictionary': pca_builder.show_full_dictionary}

    @app.get('/users', response_model=List[UserSummary])
    def list_users(x_api_key: str | None = Header(default=None, alias='X-API-Key')) -> List[UserSummary]:
        _verify_api_key(x_api_key)
        return _scan_users(FACE_MODELS_DIR)

    @app.get('/events/recent')
    def recent_events(
        limit: int = 50,
        x_api_key: str | None = Header(default=None, alias='X-API-Key'),
    ) -> dict:
        _verify_api_key(x_api_key)
        return {'events': hub.recent_events(limit=limit)}

    @app.get('/contacts', response_model=List[ContactCandidate])
    def list_contacts(
        date_from: datetime = Query(alias='from', description='時間窓の開始 (ISO8601)'),
        date_to: datetime = Query(alias='to', description='時間窓の終了 (ISO8601)'),
        action: str | None = Query(default=None, description='bring_in / take_out で絞り込む'),
        x_api_key: str | None = Header(default=None, alias='X-API-Key'),
    ) -> List[ContactCandidate]:
        """指定時間窓の接触イベントを人物名・2D bbox付きで返す。

        object_search_system が「時刻+bbox」で SAM2 物体に人物を帰属させるための照会口。
        IoU 判定は呼び出し側の責務なので、ここでは候補を絞らず時間窓で返すだけにする。
        """
        _verify_api_key(x_api_key)
        if date_from > date_to:
            raise HTTPException(status_code=400, detail='from must be earlier than to')
        try:
            # shigure_core / mysql-connector を import 時点で要求しないよう遅延 import する
            from shigure_core.db.event_repository import EventRepository

            rows = EventRepository.select_contacts(date_from, date_to, action)
        except Exception as exc:
            # DB 断でも API 全体は落とさない（顔認識イベント配信は継続させる）
            raise HTTPException(status_code=503, detail=f'DB query failed: {exc}') from exc
        return [contact_row_to_model(row) for row in rows]

    @app.get('/api/users/{user_id}/features/{feature_num}/face')
    def get_feature_face(
        user_id: str,
        feature_num: int,
        x_api_key: str | None = Header(default=None, alias='X-API-Key'),
        api_key: str | None = Query(default=None),
    ) -> Response:
        """Return a JPEG thumbnail for face_models/{user_id}/{user_id}_{feature_num}.jpg."""
        _verify_api_key(x_api_key, api_key)
        image_path = find_face_image_path(FACE_MODELS_DIR, user_id, feature_num)
        if image_path is None or not image_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f'Face image not found for user_id={user_id} feature_num={feature_num}',
            )
        try:
            body = load_face_thumbnail(image_path)
        except OSError as exc:
            raise HTTPException(status_code=404, detail=f'Face image unreadable: {exc}') from exc
        return Response(content=body, media_type='image/jpeg')

    @app.websocket('/ws/events')
    async def ws_events(websocket: WebSocket) -> None:
        if API_KEY:
            key = websocket.query_params.get('api_key')
            if key != API_KEY:
                await websocket.close(code=4401, reason='Invalid API key')
                return
        await hub.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            await hub.disconnect(websocket)

    @app.websocket('/ws/pca_plot')
    async def ws_pca_plot(websocket: WebSocket) -> None:
        if API_KEY:
            key = websocket.query_params.get('api_key')
            if key != API_KEY:
                await websocket.close(code=4401, reason='Invalid API key')
                return
        await pca_hub.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            await pca_hub.disconnect(websocket)

    @app.websocket('/ws/tracking_debug')
    async def ws_tracking_debug(websocket: WebSocket) -> None:
        if API_KEY:
            key = websocket.query_params.get('api_key')
            if key != API_KEY:
                await websocket.close(code=4401, reason='Invalid API key')
                return
        await tracking_debug_hub.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            await tracking_debug_hub.disconnect(websocket)

    return app


app = create_app()
