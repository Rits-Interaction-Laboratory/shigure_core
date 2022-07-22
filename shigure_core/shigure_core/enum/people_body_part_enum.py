from enum import Enum


class PeopleBodyPartEnum(Enum):
    NOSE = "nose"
    NECK = "neck"

    RIGHT_SHOULDER = "right_shoulder"
    RIGHT_ELBOW = "right_elbow"
    RIGHT_WRIST = "right_wrist"

    LEFT_SHOULDER = "left_shoulder"
    LEFT_ELBOW = "left_elbow"
    LEFT_WRIST = "left_wrist"

    MID_HIP = "mid_hip"

    RIGHT_HIP = "right_hip"
    RIGHT_KNEE = "right_knee"
    RIGHT_ANKLE = "right_ankle"

    LEFT_HIP = "left_hip"
    LEFT_KNEE = "left_knee"
    LEFT_ANKLE = "left_ankle"

    RIGHT_EYE = "right_eye"
    LEFT_EYE = "left_eye"
    RIGHT_EAR = "right_ear"
    LEFT_EAR = "left_ear"

    LEFT_BIG_TOE = "left_big_toe"
    LEFT_SMALL_TOE = "left_small_toe"
    LEFT_HEEL = "left_heel"

    RIGHT_BIG_TOE = "right_big_toe"
    RIGHT_SMALL_TOE = "right_small_toe"
    RIGHT_HEEL = "right_heel"

    @staticmethod
    def value_of():
        for body_part in PeopleBodyPartEnum:
            return body_part.value
