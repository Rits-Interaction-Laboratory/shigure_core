import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from shigure_core.db.object_tracking_event_history_repository import ObjectTrackingEventHistoryRepository
from shigure_core.db.object_tracking_summary_repository import ObjectTrackingSummaryRepository
from shigure_core.nodes.api_json_util import to_jsonable as _to_jsonable

# DBをポーリングしてWebSocketにpushする間隔(秒)
_POLL_INTERVAL_SEC = 1.0


def register_object_tracking_ws_routes(app: FastAPI) -> None:
    """object_tracking関連のWebSocketエンドポイントを既存のFastAPIアプリに登録する.

    ポートを分けずに1つのAPIサーバー(bbox_visualizerの8080番)に相乗りさせるため、
    ルート登録だけを行う関数として切り出している。
    """

    @app.websocket("/ws/summary")
    async def ws_summary(websocket: WebSocket):
        """object_tracking_summary(親テーブル)の全行を一定間隔でそのまま配信する."""
        await websocket.accept()
        try:
            while True:
                rows = await asyncio.to_thread(ObjectTrackingSummaryRepository.fetch_all)
                await websocket.send_json([_to_jsonable(row) for row in rows])
                await asyncio.sleep(_POLL_INTERVAL_SEC)
        except WebSocketDisconnect:
            pass

    @app.websocket("/ws/events/{object_id}")
    async def ws_events(websocket: WebSocket, object_id: str):
        """object_id専用の子テーブル(object_tracking_event_history_<object_id>)を配信する.

        object_idはURLパスで指定する。形式が不正な場合はエラーを送ってから切断する。
        """
        await websocket.accept()
        try:
            while True:
                try:
                    rows = await asyncio.to_thread(
                        ObjectTrackingEventHistoryRepository.fetch_events, object_id)
                except ValueError as e:
                    await websocket.send_json({"error": str(e)})
                    await websocket.close(code=1008)
                    return

                await websocket.send_json([_to_jsonable(row) for row in rows])
                await asyncio.sleep(_POLL_INTERVAL_SEC)
        except WebSocketDisconnect:
            pass
