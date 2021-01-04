import numpy as np
from openpose_ros2_msgs.msg import PoseKeyPointsList, PoseKeyPoints, PoseKeyPoint

from shigure_core.nodes.people_tracking.tracking_info import TrackingInfo

_NECK_INDEX = 1


class PeopleTrackingLogic:
    """人物追跡ロジック."""

    @staticmethod
    def execute(depth_img: np.ndarray, key_points_list: PoseKeyPointsList, tracking_info: TrackingInfo,
                focal_length: float, threshold_distance: int = 1000) -> TrackingInfo:
        height, width = depth_img.shape[:2]

        current_people_list = []

        key_points: PoseKeyPoints
        for key_points in key_points_list.pose_key_points_list:
            # とりあえずNeckの座標で計算
            neck_point: PoseKeyPoint = key_points.pose_key_points[_NECK_INDEX]

            if neck_point.x == 0 and neck_point.y == 0:
                continue

            x = int(neck_point.x) if neck_point.x < width else width - 1
            y = int(neck_point.y) if neck_point.y < height else height - 1

            # 透視逆変換して保存
            current_people_list.append((neck_point.x / focal_length,
                                        neck_point.y / focal_length, depth_img[y, x]))

        return PeopleTrackingLogic.tracking(current_people_list, tracking_info, threshold_distance)

    @staticmethod
    def tracking(current_people_list: list, tracking_info: TrackingInfo, threshold_distance: int) -> TrackingInfo:
        previous_people_dict = tracking_info.get_people_dict()
        current_people_dict = {}
        if len(previous_people_dict) == 0:
            for people in current_people_list:
                current_people_dict[tracking_info.new_people_id()] = people
            tracking_info.update_people_dict(current_people_dict)
            return tracking_info

        for people_id, previous_people in previous_people_dict.items():
            previous_x, previous_y, previous_z = previous_people
            min_diff = 0
            for current_people in current_people_list:
                current_x, current_y, current_z = current_people

                diff_x = abs(previous_x - current_x)
                diff_y = abs(previous_y - current_y)
                diff_z = abs(previous_z - current_z)

                if (diff_x < threshold_distance and
                        diff_y < threshold_distance and
                        diff_z < threshold_distance):
                    if people_id not in current_people_dict.keys() or min_diff > diff_x + diff_y + diff_z:
                        current_people_dict[people_id] = (current_x, current_y, current_z)
                        min_diff = diff_x + diff_y + diff_z
                        current_people_list.remove(current_people)

        # 余った人物は新規登録
        for people in current_people_list:
            current_people_dict[tracking_info.new_people_id()] = people

        tracking_info.update_people_dict(current_people_dict)

        return tracking_info
