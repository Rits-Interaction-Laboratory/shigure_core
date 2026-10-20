import asyncio
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from shigure_core.db.db_util import DbConnectionError
from shigure_core.db.object_history_repository import ObjectHistoryRepository
from shigure_core.nodes.api_json_util import to_jsonable

# DBをポーリングしてWebSocketにpushする間隔(秒)
_POLL_INTERVAL_SEC = 1.0


def register_object_history_routes(app: FastAPI) -> None:
    """object_history関連のREST/WebSocketエンドポイントを既存のFastAPIアプリに登録する.

    ポートを分けずに1つのAPIサーバー(bbox_visualizerの8080番)に相乗りさせるため、
    ルート登録だけを行う関数として切り出している。
    """

    @app.get("/history")
    def get_history(object_id: Optional[str] = None, open_only: bool = False):
        """object_historyの行を取得する.

        object_idを指定すればその物体の行のみ、指定しなければ全行を返す。
        open_only=trueなら現在持ち込み中(take_out_at IS NULL)の行のみに絞る。
        """
        try:
            if object_id is not None:
                rows = ObjectHistoryRepository.fetch_by_object_id(object_id)
            else:
                rows = ObjectHistoryRepository.fetch_all(open_only=open_only)
        except DbConnectionError as e:
            raise HTTPException(status_code=503, detail=str(e))

        return [to_jsonable(row) for row in rows]

    @app.websocket("/ws/history")
    async def ws_history(websocket: WebSocket):
        """object_historyの全行を一定間隔でそのまま配信する."""
        await websocket.accept()
        try:
            while True:
                try:
                    rows = await asyncio.to_thread(ObjectHistoryRepository.fetch_all)
                except DbConnectionError:
                    rows = []

                await websocket.send_json([to_jsonable(row) for row in rows])
                await asyncio.sleep(_POLL_INTERVAL_SEC)
        except WebSocketDisconnect:
            pass
