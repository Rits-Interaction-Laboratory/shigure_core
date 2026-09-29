"""Tracking debug overlay images for WebSocket clients."""

from __future__ import annotations

import asyncio
import base64
import queue
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set

from pydantic import BaseModel


class TrackingDebugImage(BaseModel):
    type: str = 'tracking_debug_image'
    timestamp: str
    frame: int
    format: str = 'jpeg'
    image_base64: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def debug_image_msg_to_payload(msg) -> Dict[str, Any]:
    """ROS DebugImage -> JSON-serializable dict (JPEG base64)."""
    frame = 0
    if msg.header.frame_id:
        try:
            frame = int(msg.header.frame_id)
        except ValueError:
            frame = 0
    image_format = msg.format or 'jpeg'
    encoded = base64.b64encode(bytes(msg.data)).decode('ascii')
    return TrackingDebugImage(
        timestamp=_now_iso(),
        frame=frame,
        format=image_format,
        image_base64=encoded,
    ).model_dump()


class TrackingDebugHub:
    """Broadcast tracking debug images to WebSocket clients."""

    def __init__(self) -> None:
        self._clients: Set[Any] = set()
        self._stats_clients: Set[Any] = set()
        self._lock = asyncio.Lock()
        self._stats_clients_lock = asyncio.Lock()
        self._thread_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
        self._latest_payload: Optional[Dict[str, Any]] = None

    async def start(self) -> None:
        asyncio.create_task(self._broadcast_loop())
        asyncio.create_task(self._stats_loop())

    def stats(self) -> Dict[str, Any]:
        """画像送出待ちの枚数を返す."""
        return {
            'type': 'tracking_debug_stats',
            'queue_depth': self._thread_queue.qsize(),
        }

    def enqueue(self, payload: Dict[str, Any]) -> None:
        self._thread_queue.put_nowait(payload)

    async def _broadcast_loop(self) -> None:
        while True:
            payload = await asyncio.to_thread(self._thread_queue.get)
            if payload.get('type') == 'tracking_debug_image':
                self._latest_payload = payload
            async with self._lock:
                dead = []
                for ws in self._clients:
                    try:
                        await ws.send_json(payload)
                    except Exception:
                        dead.append(ws)
                for ws in dead:
                    self._clients.discard(ws)

    async def _stats_loop(self) -> None:
        """キューの深さを画像とは別の WebSocket へ送る."""
        while True:
            await asyncio.sleep(0.25)
            payload = self.stats()
            async with self._stats_clients_lock:
                clients = list(self._stats_clients)
            dead = []
            for ws in clients:
                try:
                    await ws.send_json(payload)
                except Exception:
                    dead.append(ws)
            if dead:
                async with self._stats_clients_lock:
                    for ws in dead:
                        self._stats_clients.discard(ws)

    async def connect(self, websocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._clients.add(websocket)
        if self._latest_payload is not None:
            await websocket.send_json(self._latest_payload)

    async def disconnect(self, websocket) -> None:
        async with self._lock:
            self._clients.discard(websocket)

    async def connect_stats(self, websocket) -> None:
        """キュー統計用の WebSocket を受け付ける。画像は送らない."""
        await websocket.accept()
        async with self._stats_clients_lock:
            self._stats_clients.add(websocket)
        await websocket.send_json(self.stats())

    async def disconnect_stats(self, websocket) -> None:
        """キュー統計用の WebSocket を外す."""
        async with self._stats_clients_lock:
            self._stats_clients.discard(websocket)
