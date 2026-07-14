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
        self._lock = asyncio.Lock()
        self._thread_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
        self._latest_payload: Optional[Dict[str, Any]] = None

    async def start(self) -> None:
        asyncio.create_task(self._broadcast_loop())

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

    async def connect(self, websocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._clients.add(websocket)
        if self._latest_payload is not None:
            await websocket.send_json(self._latest_payload)

    async def disconnect(self, websocket) -> None:
        async with self._lock:
            self._clients.discard(websocket)
