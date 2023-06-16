from operator import itemgetter
from typing import Tuple, List

from shigure_core_msgs.msg import PoseKeyPoints, PointData, TrackedObject, Point, TrackedObjectList, PoseKeyPointsList

from shigure_core.enum.contact_action_enum import ContactActionEnum
from shigure_core.enum.tracked_object_action_enum import TrackedObjectActionEnum
from shigure_core.nodes.contact_detection.cube import Cube
import pandas as pd


class ContactDetectionLogic:
    """接触推定ロジック."""

    LEFT_HAND_INDEX = 4
    RIGHT_HAND_INDEX = 7

    @classmethod
    def execute(cls, object_list: TrackedObjectList, people: PoseKeyPointsList,
                hand_collider_distance: int = 300) -> Tuple[
            List[Tuple[Tuple[PoseKeyPoints, Cube, int], Tuple[TrackedObject, Cube]]], bool]:

        is_not_touch = False

        hand_cube_list = []
        person: PoseKeyPoints
        for person in people.pose_key_points_list:
            left_hand: PointData = person.point_data[cls.LEFT_HAND_INDEX]
            if int(left_hand.pixel_point.x) != 0 and int(left_hand.pixel_point.y) != 0:
                hand_cube_list.append(
                    (person,
                     ContactDetectionLogic.convert_point_to_cube(left_hand.projection_point, hand_collider_distance),
                     cls.LEFT_HAND_INDEX)
                )
            right_hand: PointData = person.point_data[cls.RIGHT_HAND_INDEX]
            if int(right_hand.pixel_point.x) != 0 and int(right_hand.pixel_point.y) != 0:
                hand_cube_list.append(
                    (person,
                     ContactDetectionLogic.convert_point_to_cube(right_hand.projection_point, hand_collider_distance),
                     cls.RIGHT_HAND_INDEX)
                )

        object_cube_list = []
        tracked_object: TrackedObject
        for tracked_object in object_list.tracked_object_list:
            action = ContactActionEnum.from_tracked_object_action_enum(
                TrackedObjectActionEnum.value_of(tracked_object.action)
            )

            is_not_touch |= action != ContactActionEnum.TOUCH

            collider: Cube = tracked_object.collider
            
            try:
                x, y, z = int(collider.x), int(collider.y), int(collider.z)
                width, height, depth = int(collider.width), int(collider.height), int(collider.depth)
            
            except Exception as e:
                print(e)
                
                print({'x': collider.x, 'y': collider.y, 'z': collider.z, 'width': collider.width, 'height': collider.height,'depth': collider.depth})
                
                df = pd.DataFrame({'colliderx':[collider.x], 'collidery':[collider.y],'colliderz':[collider.z],'width':[collider.width],'height':[collider.height],'depth':[collider.depth]})
                x, y, z = int(df['colliderx'].fillna(0)), int(df['collidery'].fillna(0)), int(df['colliderz'].fillna(0))
                
                width, height, depth = int(df['width'].fillna(0)), int(df['height'].fillna(0)), int(df['depth'].fillna(0))

            object_cube_list.append((tracked_object, Cube(x, y, z, width, height, depth)))

        linked_list = []
        object_cube: Cube
        hand_cube: Cube
        for object_item in object_cube_list:
            tracked_object_info, object_cube = object_item
            result = False
            is_bring_in_or_take_out = tracked_object_info.action == TrackedObjectActionEnum.BRING_IN.value or tracked_object_info.action == TrackedObjectActionEnum.TAKE_OUT.value
            for hand in hand_cube_list:
                _, hand_cube, _ = hand
                result, volume = object_cube.is_collided(hand_cube)
                if result | is_bring_in_or_take_out:
                    linked_list.append((hand, object_item, volume))
            if (not result) & is_bring_in_or_take_out:
                    print("Note : The system did not detect contact between the wrist and the object, but prioritized the determination of whether the object was brought in or taken away.")

        result_list = []
        linked_list = sorted(linked_list, key=itemgetter(2), reverse=True)
        for hand, object_item, _ in linked_list:
            if hand in hand_cube_list and object_item in object_cube_list:
                hand_cube_list.remove(hand)
                object_cube_list.remove(object_item)
                result_list.append((hand, object_item))

        return result_list, is_not_touch

    @staticmethod
    def convert_point_to_cube(point: Point, hand_collider_distance: int) -> Cube:
        x = point.x - hand_collider_distance
        y = point.y - hand_collider_distance
        z = point.z - hand_collider_distance
        return Cube(x, y, z,
                    x + hand_collider_distance * 2,
                    y + hand_collider_distance * 2,
                    z + hand_collider_distance * 2)
