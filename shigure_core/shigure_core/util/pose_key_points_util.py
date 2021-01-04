from openpose_ros2_msgs.msg import PoseKeyPoints as OpenPosePoseKeyPoints, PoseKeyPoint as OpenPosePoseKeyPoint
from shigure_core_msgs.msg import PoseKeyPoints as ShigurePoseKeyPoints, PoseKeyPoint as ShigurePoseKeyPoint, \
    BoundingBox


def convert_openpose_to_shigure(openpose_key_points_list: OpenPosePoseKeyPoints,
                                people_id: str) -> ShigurePoseKeyPoints:
    """
    OpenPoseのPoseKeyPointからShigureのPoseKeyPointに変換します.
    """

    shigure_pose_key_points = ShigurePoseKeyPoints()
    shigure_pose_key_points.people_id = people_id

    x_values = []
    y_values = []

    pose_key_point: OpenPosePoseKeyPoint
    for pose_key_point in openpose_key_points_list.pose_key_points:
        shigure_pose_key_point = ShigurePoseKeyPoint()
        shigure_pose_key_point.x = pose_key_point.x
        shigure_pose_key_point.y = pose_key_point.y
        shigure_pose_key_point.score = pose_key_point.score

        if pose_key_point.x != 0 and pose_key_point.y != 0:
            x_values.append(pose_key_point.x)
            y_values.append(pose_key_point.y)

        shigure_pose_key_points.pose_key_points.append(shigure_pose_key_point)

    bounding_box = BoundingBox()
    bounding_box.x = min(x_values)
    bounding_box.y = min(y_values)
    bounding_box.width = max(x_values) - bounding_box.x
    bounding_box.height = max(y_values) - bounding_box.y

    shigure_pose_key_points.bounding_box = bounding_box

    return shigure_pose_key_points
