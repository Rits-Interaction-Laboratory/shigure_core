import numpy as np

from openpose_ros2_msgs.msg import PoseKeyPoints as OpenPosePoseKeyPoints, PoseKeyPoint as OpenPosePoseKeyPoint
from shigure_core_msgs.msg import PoseKeyPoints as ShigurePoseKeyPoints, Point, BoundingBox, PointData


def convert_openpose_to_shigure(openpose_key_points_list: OpenPosePoseKeyPoints,
                                depth_img: np.ndarray, people_id: str, k: np.ndarray) -> ShigurePoseKeyPoints:
    """
    OpenPoseのPoseKeyPointからShigureのPoseKeyPointに変換します.
    """

    shigure_pose_key_points = ShigurePoseKeyPoints()
    shigure_pose_key_points.people_id = people_id

    x_values = []
    y_values = []

    height, width = depth_img.shape[:2]
    a_inv = np.linalg.inv(k)

    pose_key_point: OpenPosePoseKeyPoint
    for pose_key_point in openpose_key_points_list.pose_key_points:
        x = int(pose_key_point.x) if pose_key_point.x < width else width - 1
        y = int(pose_key_point.y) if pose_key_point.y < height else height - 1
        depth = float(depth_img[y, x])

        pixel_point = Point()
        pixel_point.x = pose_key_point.x
        pixel_point.y = pose_key_point.y

        projection_point = Point()
        if pose_key_point.x != 0 and pose_key_point.y != 0:
            pixel_point.z = depth

            # 透視逆変換して保存
            s = np.asarray([[pose_key_point.x, pose_key_point.y, 1]]).T
            m = (depth * np.matmul(a_inv, s)).T
            projection_point.x = m[0, 0]
            projection_point.y = m[0, 1]
            projection_point.z = depth

            x_values.append(pose_key_point.x)
            y_values.append(pose_key_point.y)
        else:
            pixel_point.z = 0.

            projection_point.x = 0.
            projection_point.y = 0.
            projection_point.z = 0.

        point_data = PointData()
        point_data.pixel_point = pixel_point
        point_data.projection_point = projection_point
        point_data.score = pose_key_point.score
        shigure_pose_key_points.point_data.append(point_data)

    bounding_box = BoundingBox()
    bounding_box.x = min(x_values)
    bounding_box.y = min(y_values)
    bounding_box.width = max(x_values) - bounding_box.x
    bounding_box.height = max(y_values) - bounding_box.y

    shigure_pose_key_points.bounding_box = bounding_box

    return shigure_pose_key_points
