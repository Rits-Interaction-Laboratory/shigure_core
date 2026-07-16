from operator import itemgetter

import numpy as np
from openpose_ros2_msgs.msg import PoseKeyPointsList, PoseKeyPoints, PoseKeyPoint

from shigure_core.nodes.people_tracking.tracking_info import TrackingInfo

# 深度欠損(0)を補完するときに参照する近傍ウィンドウの半径(px)。
# RealSense の深度は首付近で頻繁に欠損する。欠損値(0)をそのまま透視逆変換すると
# 3次元位置が原点(0,0,0)へ飛び、前フレームとの距離が threshold_distance を超えて
# 「別人」と判定され追跡IDが振り直される（＝顔名の累積スコアがリセットされる）。
DEPTH_SAMPLE_RADIUS = 4


class PeopleTrackingLogic:
    """人物追跡ロジック."""

    @staticmethod
    def execute(depth_img: np.ndarray, key_points_list: PoseKeyPointsList, tracking_info: TrackingInfo,
                k: np.ndarray, threshold_distance: int = 1000, threshold_person: float = 0.3,
                neck_index: int = 1) -> TrackingInfo:
        height, width = depth_img.shape[:2]

        current_people_list = []

        key_points: PoseKeyPoints
        for key_points in key_points_list.pose_key_points_list:
            # 検出率スコアの平均を計算
            average_score = PeopleTrackingLogic.calculate_average_score(key_points)
            # 検出率が閾値より小さければtrackingをスキップ
            if average_score < threshold_person:
                continue

            neck_point: PoseKeyPoint = key_points.pose_key_points[neck_index]

            if neck_point.x == 0 and neck_point.y == 0:
                continue

            x = int(neck_point.x) if neck_point.x < width else width - 1
            y = int(neck_point.y) if neck_point.y < height else height - 1

            s = np.asarray([[neck_point.x, neck_point.y, 1]]).T
            a_inv = np.linalg.inv(k)

            # 透視逆変換して保存
            # 深度が欠損(0)のままだと m が (0, 0) となり位置が原点へ飛ぶため、
            # 近傍の有効値で補完してから変換する。
            depth = PeopleTrackingLogic.sample_depth(depth_img, x, y)
            if depth <= 0:
                # 近傍も全て欠損。このフレームでは位置を確定できないのでスキップする。
                continue
            m = (depth * np.matmul(a_inv, s)).T
            current_people_list.append((m[0, 0], m[0, 1], depth, key_points))

        return PeopleTrackingLogic.tracking(current_people_list, tracking_info, threshold_distance)

    @staticmethod
    def sample_depth(depth_img: np.ndarray, x: int, y: int, radius: int = DEPTH_SAMPLE_RADIUS) -> float:
        """
        (x, y) の深度を返す. 欠損(0)の場合は近傍の有効値の中央値で補完する.

        :param depth_img: 深度画像
        :param x: 参照するx座標
        :param y: 参照するy座標
        :param radius: 補完に使う近傍ウィンドウの半径(px)
        :return: 深度. 近傍も全て欠損している場合は 0.0
        """
        depth = float(depth_img[y, x])
        if depth > 0:
            return depth

        height, width = depth_img.shape[:2]
        window = depth_img[max(0, y - radius):min(height, y + radius + 1),
                           max(0, x - radius):min(width, x + radius + 1)]
        valid = window[window > 0]
        if valid.size == 0:
            return 0.0
        return float(np.median(valid))

    @staticmethod
    def calculate_average_score(key_points: PoseKeyPoints) -> float:
        total_score = sum([key_point.score for key_point in key_points.pose_key_points])
        num_keypoints = len(key_points.pose_key_points)
        return total_score / num_keypoints if num_keypoints > 0 else 0

    @staticmethod
    def tracking(current_people_list: list, tracking_info: TrackingInfo, threshold_distance: int) -> TrackingInfo:
        # マッチ候補は「前フレーム可視 ＋ 直近で見失った coasting」。
        # coasting を候補に含めることで、数フレーム見失った人物が再出現しても
        # 同じ people_id へ復帰でき、IDの振り直し（顔名累積のリセット）を抑制する。
        match_pool = tracking_info.get_match_pool()
        current_people_dict = {}
        if len(match_pool) == 0:
            for people in current_people_list:
                current_people_dict[tracking_info.new_people_id()] = people
            tracking_info.commit(current_people_dict)
            return tracking_info

        linked_list = []
        for people_id, previous_people in match_pool.items():
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
        remaining = list(current_people_list)
        for people_id, people, _ in linked_list:
            if people_id not in current_people_dict.keys() and people in remaining:
                current_people_dict[people_id] = people
                remaining.remove(people)

        # 余った人物は新規登録
        for people in remaining:
            current_people_dict[tracking_info.new_people_id()] = people

        # commit が可視(current_people_dict)を反映し、未マッチの候補を age++ して
        # max_age 以内なら coasting に残す（超えたら破棄＝完全ロスト）。
        tracking_info.commit(current_people_dict)

        return tracking_info
