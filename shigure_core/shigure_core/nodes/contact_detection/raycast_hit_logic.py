from operator import itemgetter
from typing import Tuple, List

from shigure_core_msgs.msg import TrackedObject, TrackedObjectList
from geometry_msgs.msg import Point

from shigure_core.nodes.contact_detection.cube import Cube
import pandas as pd


class RaycastHitDetectionLogic:
    """3次元座標の接触判定ロジック."""

    @classmethod
    def execute(cls, object_list: TrackedObjectList, point: Point,
                collider_distance: int = 100) -> Tuple[List[Tuple[TrackedObject, Cube]], bool]:
        """
        object_list: カメラで検知した物体リスト
        point: 判定したい3次元座標
        collider_distance: 判定の際に使う距離の閾値
        戻り値: 接触した物体リスト、接触があったかどうかのフラグ
        """
        object_cube_list = []
        result_list = []

        # オブジェクトリストをCubeに変換
        for tracked_object in object_list.tracked_object_list:
            collider: Cube = tracked_object.collider

            try:
                x, y, z = int(collider.x), int(collider.y), int(collider.z)
                width, height, depth = int(collider.width), int(collider.height), int(collider.depth)

            except Exception as e:
                print(e)
                print({'x': collider.x, 'y': collider.y, 'z': collider.z, 'width': collider.width,
                       'height': collider.height, 'depth': collider.depth})

                df = pd.DataFrame(
                    {'colliderx': [collider.x], 'collidery': [collider.y], 'colliderz': [collider.z],
                     'width': [collider.width], 'height': [collider.height], 'depth': [collider.depth]})

                x, y, z = int(df['colliderx'].fillna(0)), int(df['collidery'].fillna(0)), int(df['colliderz'].fillna(0))
                width, height, depth = int(df['width'].fillna(0)), int(df['height'].fillna(0)), int(df['depth'].fillna(0))

            object_cube_list.append((tracked_object, Cube(x, y, z, width, height, depth)))

        # 3次元座標をCubeに変換
        point_cube = cls.convert_point_to_cube(point, collider_distance)
        print({'x': point_cube.x, 'y': point_cube.y, 'z': point_cube.z, 'width': point_cube.width,
                       'height': point_cube.height, 'depth': point_cube.depth})

        # 接触判定
        is_touch = False
        for tracked_object, object_cube in object_cube_list:
            result, volume = object_cube.is_collided(point_cube)
            if result:
                result_list.append((tracked_object, object_cube))
                is_touch = True

        print("result list:", result_list)
        return result_list, is_touch

    @staticmethod
    def convert_point_to_cube(point: Point, collider_distance: int) -> Cube:
        """
        3次元座標をCube形式に変換.
        """
        x = point.x - collider_distance
        y = point.y - collider_distance
        z = point.z - collider_distance
        return Cube(x, y, z,
                    collider_distance * 2,
                    collider_distance * 2,
                    collider_distance * 2)
