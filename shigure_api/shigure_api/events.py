"""Event models and ROS message → JSON mapping."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

UserEventType = Literal[
    'user_registered',
    'user_confirmed',
    'user_recognized',
    'registration_failed',
]


class UserEvent(BaseModel):
    type: UserEventType
    user_id: str
    people_id: Optional[str] = None
    score: Optional[float] = None
    cumulative_score: Optional[float] = None
    message: str
    timestamp: str = Field(default_factory=lambda: _now_iso())


class UserSummary(BaseModel):
    user_id: str
    feature_count: int
    profile_feature_count: int = 0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def event_from_dictionary_update(people_id: str, name: str) -> Optional[UserEvent]:
    if not name or name == 'none':
        return UserEvent(
            type='registration_failed',
            user_id='none',
            people_id=people_id,
            message=f'ユーザー登録に失敗しました（people_id={people_id}）',
        )
    if name.startswith('user_new'):
        return UserEvent(
            type='user_registered',
            user_id=name,
            people_id=people_id,
            message=f'新規ユーザー {name} が登録されました',
        )
    if name.startswith('user'):
        return UserEvent(
            type='user_confirmed',
            user_id=name,
            people_id=people_id,
            message=f'既存ユーザー {name} として認識・確定しました',
        )
    return UserEvent(
        type='user_confirmed',
        user_id=name,
        people_id=people_id,
        message=f'ユーザー {name} が確定しました',
    )


def event_from_face_recognition(face_id: str, score: float) -> Optional[UserEvent]:
    if face_id.endswith('@profile'):
        base_id = face_id.replace('@profile', '')
        if base_id == 'unknown':
            return None
        return UserEvent(
            type='user_recognized',
            user_id=base_id,
            score=score,
            message=f'横顔で {base_id} を認識しました（score={score:.2f}）',
        )
    if face_id == 'unknown' or not face_id.startswith('user'):
        return None
    return UserEvent(
        type='user_recognized',
        user_id=face_id,
        score=score,
        message=f'{face_id} を認識しました（score={score:.2f}）',
    )


def user_event_to_dict(event: UserEvent) -> Dict[str, Any]:
    return event.model_dump()


def cumulative_scores_to_dict(scores: Dict[str, float]) -> Dict[str, Any]:
    """全ユーザーの累積スコアマップを配信用の辞書に変換する。"""
    return {
        'type': 'cumulative_scores',
        'scores': {user_id: round(value, 4) for user_id, value in scores.items()},
        'timestamp': _now_iso(),
    }
