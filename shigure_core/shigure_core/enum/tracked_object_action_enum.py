from enum import Enum


class TrackedObjectActionEnum(Enum):
    TAKE_OUT = 'take_out'  # 持ち出し
    BRING_IN = 'bring_in'  # 持ち込み
    OBJ_MOVE = 'obj_move'  #　移動
    STAY = 'stay'          # そのまま

    @classmethod
    def value_of(cls, value):
        for action in TrackedObjectActionEnum:
            if action.value == value:
                return action
