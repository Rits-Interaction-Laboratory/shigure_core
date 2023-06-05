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

    LEFT_WRIST_P = "left_wrist_p"

    LEFT_THUMB_CMC = "left_thumb_cmc"
    LEFT_THUMB_MCP = "left_thumb_mcp"
    LEFT_THUMB_IP = "left_thumb_ip"
    LEFT_THUMB_TIP = "left_thumb_tip"

    LEFT_INDEX_FINGER_MCP = "left_index_finger_mcp"
    LEFT_INDEX_FINGER_PIP = "left_index_finger_pip"
    LEFT_INDEX_FINGER_DIP = "left_index_finger_dip"
    LEFT_INDEX_FINGER_TIP = "left_index_finger_tip"

    LEFT_MIDDLE_FINGER_MCP = "left_middle_finger_mcp"
    LEFT_MIDDLE_FINGER_PIP = "left_middle_finger_pip"
    LEFT_MIDDLE_FINGER_DIP = "left_middle_finger_dip"
    LEFT_MIDDLE_FINGER_TIP = "left_middle_finger_tip"

    LEFT_RING_FINGER_MCP = "left_ring_finger_mcp"
    LEFT_RING_FINGER_PIP = "left_ring_finger_pip"
    LEFT_RING_FINGER_DIP = "left_ring_finger_dip"
    LEFT_RING_FINGER_TIP = "left_ring_finger_tip"

    LEFT_PINKY_MCP = "left_pinky_mcp"
    LEFT_PINKY_PIP = "left_pinky_pip"
    LEFT_PINKY_DIP = "left_pinky_dip"
    LEFT_PINKY_TIP = "left_pinky_tip"

    RIGHT_WRIST_P = "right_wrist_p"

    RIGHT_THUMB_CMC = "right_thumb_cmc"
    RIGHT_THUMB_MCP = "right_thumb_mcp"
    RIGHT_THUMB_IP = "right_thumb_ip"
    RIGHT_THUMB_TIP = "right_thumb_tip"

    RIGHT_INDEX_FINGER_MCP = "right_index_finger_mcp"
    RIGHT_INDEX_FINGER_PIP = "right_index_finger_pip"
    RIGHT_INDEX_FINGER_DIP = "right_index_finger_dip"
    RIGHT_INDEX_FINGER_TIP = "right_index_finger_tip"

    RIGHT_MIDDLE_FINGER_MCP = "right_middle_finger_mcp"
    RIGHT_MIDDLE_FINGER_PIP = "right_middle_finger_pip"
    RIGHT_MIDDLE_FINGER_DIP = "right_middle_finger_dip"
    RIGHT_MIDDLE_FINGER_TIP = "right_middle_finger_tip"

    RIGHT_RING_FINGER_MCP = "right_ring_finger_mcp"
    RIGHT_RING_FINGER_PIP = "right_ring_finger_pip"
    RIGHT_RING_FINGER_DIP = "right_ring_finger_dip"
    RIGHT_RING_FINGER_TIP = "right_ring_finger_tip"

    RIGHT_PINKY_MCP = "right_pinky_mcp"
    RIGHT_PINKY_PIP = "right_pinky_pip"
    RIGHT_PINKY_DIP = "right_pinky_dip"
    RIGHT_PINKY_TIP = "right_pinky_tip"
    
    @staticmethod
    def value_of():
        for body_part in PeopleBodyPartEnum:
            return body_part.value
