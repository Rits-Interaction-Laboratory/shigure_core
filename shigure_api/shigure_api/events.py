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
    message: str
    timestamp: str = Field(default_factory=lambda: _now_iso())


class UserSummary(BaseModel):
    user_id: str
    feature_count: int
    profile_feature_count: int = 0


class ContactCandidate(BaseModel):
    """接触イベント1件。object_search_system が時刻+bboxで突き合わせる候補。

    bbox は [x, y, width, height]（左上原点、shigure のカメラ解像度基準）。
    bbox 列は migration 9/10 以降にのみ入るため、それ以前の行では None になる。
    """

    shigure_event_id: str
    action: str
    created_at: str
    person_id: Optional[str] = None
    face_name: Optional[str] = None
    object_id: Optional[str] = None
    people_bbox: Optional[list[float]] = None
    object_bbox: Optional[list[float]] = None


def _bbox_or_none(x, y, w, h) -> Optional[list[float]]:
    """4値が揃っている場合のみ [x, y, w, h] を返す。1つでも NULL なら None。"""
    if x is None or y is None or w is None or h is None:
        return None
    return [float(x), float(y), float(w), float(h)]


def contact_row_to_model(row: Dict[str, Any]) -> ContactCandidate:
    """EventRepository.select_contacts の1行を API レスポンスに変換する。"""
    created_at = row.get('created_at')
    return ContactCandidate(
        shigure_event_id=str(row.get('shigure_event_id')),
        action=str(row.get('action')),
        created_at=created_at.isoformat() if created_at is not None else '',
        person_id=row.get('person_id'),
        face_name=row.get('face_name'),
        object_id=row.get('object_id'),
        people_bbox=_bbox_or_none(
            row.get('people_bbox_x'), row.get('people_bbox_y'),
            row.get('people_bbox_width'), row.get('people_bbox_height'),
        ),
        object_bbox=_bbox_or_none(
            row.get('object_bbox_x'), row.get('object_bbox_y'),
            row.get('object_bbox_width'), row.get('object_bbox_height'),
        ),
    )


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


def recognition_scores_to_dict(msg) -> Dict[str, Any]:
    """RecognitionHistory を people_id ごとの候補累積スコアへ変換する。

    出力の scores は「今映っている各 people_id → 候補face_id → その累積スコア」の
    入れ子マップになる（例: {"1": {"user_new1": 10.0, "user_new2": 1.0}}）。
    accumulate_score は people_tracking 側で認識のたびに加算された累積値。
    """
    scores: Dict[str, Dict[str, float]] = {}
    for user in msg.users:
        candidates: Dict[str, float] = {}
        for rec in user.face_info:
            if not rec.id or rec.id == 'none':
                continue
            candidates[rec.id] = round(float(rec.accumulate_score), 4)
        scores[user.people_id] = candidates
    return {
        'type': 'cumulative_scores',
        'scores': scores,
        'timestamp': _now_iso(),
    }
