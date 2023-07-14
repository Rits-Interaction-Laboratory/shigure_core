from enum import Enum

from shigure_core.enum.tracked_object_action_enum import TrackedObjectActionEnum


class ContactActionEnum(Enum):
    TAKE_OUT = 'take_out'  # 持ち出し
    BRING_IN = 'bring_in'  # 持ち込み
    OBJ_MOVE = 'obj_move'  #　移動
    TOUCH = 'touch'

    @classmethod
    def value_of(cls, value):
        for action in ContactActionEnum:
            if action.value == value:
                return action

    @classmethod
    def from_tracked_object_action_enum(cls, value: TrackedObjectActionEnum):
        if value == TrackedObjectActionEnum.BRING_IN:
            return cls.BRING_IN
        if value == TrackedObjectActionEnum.STAY:
            return cls.TOUCH
        if value == TrackedObjectActionEnum.TAKE_OUT:
            return cls.TAKE_OUT
        if value == TrackedObjectActionEnum.OBJ_MOVE:
            return cls.OBJ_MOVE
        # TODO: error処理
        return cls.BRING_IN
