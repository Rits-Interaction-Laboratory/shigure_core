import numpy as np
import cv2
import time

from openpose_ros2_msgs.msg import PoseKeyPoints as OpenPosePoseKeyPoints, PoseKeyPoint as OpenPosePoseKeyPoint
from shigure_core_msgs.msg import PoseKeyPoints as ShigurePoseKeyPoints, Point, BoundingBox, PointData
from shigure_core.enum.people_body_part_enum import PeopleBodyPartEnum

def convert_openpose_to_shigure(openpose_key_points_list: OpenPosePoseKeyPoints,
                                depth_img: np.ndarray, people_id: str, k: np.ndarray) -> ShigurePoseKeyPoints:
    """
    OpenPoseのPoseKeyPointからShigureのPoseKeyPointに変換します.
    """
    shigure_pose_key_points = ShigurePoseKeyPoints()
    shigure_pose_key_points.people_id = people_id
    x_values = []
    y_values = []
    z_values = [0] * len(openpose_key_points_list.pose_key_points)
    score_values = []
    part_values = []
    depth_array_list = []
    depth_array_group = []
    array_length = 1
    '''
    group ((0,15,16,17,18),(1,2,5),(3,4),(6,7),(8,9,12),(10,13),(11,22,23,24),(14,19,20,21))
    
    part_group = [[0,15,16,17,18],[1,2,5],[3,4],[6,7],[8,9,12],[10,13],[11,22,23,24],[14,19,20,21],
                  [25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45],
                  [46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66]]
    
    part_group = [[0,1,2,3,4],[5,6,7],[8,9,10,11],[12,13,14],[15,16,17,18],[19,20,21],[22,23,24],
                  [25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45],
                  [46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66]]
    
    part_group = [[0,15,16,17,18],[1,2,3,4,5,6,7],[8,9,10,11,12,13,14,19,20,21,22,23,24],
                  [25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45],
                  [46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66]]
    '''
    part_group = [[0,1,2,3,4],[5,6,7],[8,9,10,11],[12,13,14],[15,16,17,18],[19,20,21],[22,23,24],
                  [25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45],
                  [46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66]]
    
    height, width = depth_img.shape[:2]
    a_inv = np.linalg.inv(k)
    a = time.time()
    
    pose_key_point: OpenPosePoseKeyPoint
    for (pose_key_point, body_part) in zip(openpose_key_points_list.pose_key_points, PeopleBodyPartEnum):
        x = int(pose_key_point.x) if pose_key_point.x < width else width - 1
        y = int(pose_key_point.y) if pose_key_point.y < height else height - 1
        depth_array = depth_img[y - array_length : y + array_length + 1, x - array_length : x + array_length + 1]
        if depth_array.size != (array_length * 2 + 1) * (array_length * 2 + 1):
            depth_array = np.zeros((array_length * 2 + 1, array_length * 2 + 1))
        depth_array_list.append(depth_array)
        
        x_values.append(pose_key_point.x)
        y_values.append(pose_key_point.y)
        score_values.append(pose_key_point.score)
        part_values.append(body_part.value)
    #print("x" + str(len(x_values)) + "y" + str(len(y_values)) + "z" + str(len(z_values)) + "score" + str(len(score_values)) + "part" + str(len(part_values)))
    b = time.time()

    for group_idx in part_group:
        for part_idx in group_idx:
            try:
                depth_array_group.append(depth_array_list[part_idx])
            except:
                break
        nonzero_depth_array = np.sort(np.array(depth_array_group)[np.array(depth_array_group).nonzero()].ravel())
        #print("length: " + str(nonzero_depth_array.size))
        if len(nonzero_depth_array) == 0:
            min_depth = 0
            max_depth = 0
        else:
            min_depth = nonzero_depth_array[int(len(nonzero_depth_array) * 0.30)]
            max_depth = nonzero_depth_array[int(len(nonzero_depth_array) * 0.70)]
        #print("min: " + str(min_depth) + "max: " + str(max_depth))
        #print("list" + str(depth_array_list))
        for depth_idx in range(len(depth_array_group)):
            depth_arr = np.sort(depth_array_group[depth_idx].ravel())
            idx = 0
            while True:
            #print("idx"+ str(idx))
                depth = float(depth_arr[idx])
                idx += 1
                if all([depth != 0, min_depth < depth, depth < max_depth]):
                    break
                elif idx == np.array(depth_arr).size:
                    depth == 0
                    break
            z_values[group_idx[depth_idx]] = depth    
        depth_array_group.clear()    
    
    #print("SCORE: "+str(z_values))
    
    for (x, y, z, score, part) in zip(x_values, y_values, z_values, score_values, part_values):
        z = float(z)
        pixel_point = Point()
        pixel_point.x = x
        pixel_point.y = y
        projection_point = Point()
        if x != 0 and y != 0:
            pixel_point.z = z
            # 透視逆変換して保存
            s = np.asarray([[x, y, 1]]).T
            m = (z * np.matmul(a_inv, s)).T
            projection_point.x = m[0, 0]
            projection_point.y = m[0, 1]
            projection_point.z = z
        else:
            pixel_point.z = 0.
            projection_point.x = 0.
            projection_point.y = 0.
            projection_point.z = 0.
        point_data = PointData()
        point_data.body_part_name = part
        point_data.pixel_point = pixel_point
        point_data.projection_point = projection_point
        point_data.score = score
        shigure_pose_key_points.point_data.append(point_data)
    bounding_box = BoundingBox()
    x_values = np.array(x_values)[np.nonzero(np.array(x_values))].tolist()
    y_values = np.array(y_values)[np.nonzero(np.array(y_values))].tolist()
    bounding_box.x = min(x_values)
    bounding_box.y = min(y_values)
    bounding_box.width = max(x_values) - bounding_box.x
    bounding_box.height = max(y_values) - bounding_box.y
    shigure_pose_key_points.bounding_box = bounding_box
    c =time.time()
    #print(str(c-a))
    return shigure_pose_key_points
