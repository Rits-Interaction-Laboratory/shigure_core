from operator import itemgetter

import numpy as np
from openpose_ros2_msgs.msg import PoseKeyPointsList, PoseKeyPoints, PoseKeyPoint

from shigure_core.nodes.people_tracking.tracking_info import TrackingInfo

_NECK_INDEX = 1


class PeopleTrackingLogic:
    """人物追跡ロジック."""

    @staticmethod
    def execute(depth_img: np.ndarray, key_points_list: PoseKeyPointsList, tracking_info: TrackingInfo,
                k: np.ndarray, threshold_distance: int = 1000, threshold_person: float = 0.3) -> TrackingInfo:
        height, width = depth_img.shape[:2]

        current_people_list = []

        key_points: PoseKeyPoints
        for key_points in key_points_list.pose_key_points_list:
            # 検出率スコアの平均を計算
            average_score = PeopleTrackingLogic.calculate_average_score(key_points)
            # 検出率が閾値より小さければtrackingをスキップ
            if average_score < threshold_person:
                continue

            # とりあえずNeckの座標で計算
            neck_point: PoseKeyPoint = key_points.pose_key_points[_NECK_INDEX]

            # 首が検出されていなければtrackingをスキップ
            if neck_point.x == 0 and neck_point.y == 0:
                continue

            x = int(neck_point.x) if neck_point.x < width else width - 1
            y = int(neck_point.y) if neck_point.y < height else height - 1

            s = np.asarray([[neck_point.x, neck_point.y, 1]]).T
            a_inv = np.linalg.inv(k)

            # 透視逆変換して保存
            depth = depth_img[y, x]
            m = (depth * np.matmul(a_inv, s)).T
            current_people_list.append((m[0, 0], m[0, 1], depth, key_points))

        return PeopleTrackingLogic.tracking(current_people_list, tracking_info, threshold_distance)

    @staticmethod
    def calculate_average_score(key_points: PoseKeyPoints) -> float:
        total_score = sum([key_point.score for key_point in key_points.pose_key_points])
        num_keypoints = len(key_points.pose_key_points)
        return total_score / num_keypoints if num_keypoints > 0 else 0

    @staticmethod
    def tracking(current_people_list: list, tracking_info: TrackingInfo, threshold_distance: int) -> TrackingInfo:
        previous_people_dict = tracking_info.get_people_dict()
        current_people_dict = {}
        if len(previous_people_dict) == 0:
            for people in current_people_list:
                current_people_dict[tracking_info.new_people_id()] = people
            tracking_info.update_people_dict(current_people_dict)
            return tracking_info

        linked_list = []
        for people_id, previous_people in previous_people_dict.items():
            previous_x, previous_y, previous_z, _ = previous_people
            for current_people in current_people_list:
                current_x, current_y, current_z, key_point = current_people

                diff_x = abs(previous_x - current_x)
                diff_y = abs(previous_y - current_y)
                diff_z = abs(previous_z - current_z)

                current_diff_sum = diff_x + diff_y + diff_z

                if (diff_x < threshold_distance and
                        diff_y < threshold_distance and
                        diff_z < threshold_distance):
                    linked_list.append((people_id, current_people, current_diff_sum))

        linked_list = sorted(linked_list, key=itemgetter(2))
        for people_id, people, _ in linked_list:
            if people_id not in current_people_dict.keys() and people in current_people_list:
                current_people_dict[people_id] = people
                current_people_list.remove(people)

        # 余った人物は新規登録
        for people in current_people_list:
            current_people_dict[tracking_info.new_people_id()] = people

        tracking_info.update_people_dict(current_people_dict)

        return tracking_info
