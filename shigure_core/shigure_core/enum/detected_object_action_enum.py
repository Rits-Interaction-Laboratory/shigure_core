from enum import Enum


class DetectedObjectActionEnum(Enum):
    TAKE_OUT = 'take_out'  # 持ち出し
    BRING_IN = 'bring_in'  # 持ち込み
