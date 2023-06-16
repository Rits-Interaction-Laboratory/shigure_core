from collections import defaultdict
from typing import List, Dict, Tuple
import copy
import cv2
import numpy as np

from shigure_core_msgs.msg import PoseKeyPointsList
from bboxes_ex_msgs.msg import BoundingBoxes

from shigure_core.enum.detected_object_action_enum import DetectedObjectActionEnum
from shigure_core.nodes.common_model.timestamp import Timestamp
from shigure_core.nodes.common_model.union_find_tree import UnionFindTree
from shigure_core.nodes.common_model.bounding_box import BoundingBox
from shigure_core.nodes.yolox_object_detection.frame_object import FrameObject
from shigure_core.nodes.yolox_object_detection.frame_object_item import FrameObjectItem
from shigure_core.nodes.yolox_object_detection.judge_params import JudgeParams
from shigure_core.nodes.yolox_object_detection.Bbox_Object import BboxObject


class YoloxObjectDetectionLogic:
    """物体検出ロジッククラス"""
    @staticmethod
    def execute(yolox_bbox: BoundingBoxes, started_at: Timestamp, people:PoseKeyPointsList, color_img:np.ndarray, frame_object_list: List[FrameObject],
                judge_params: JudgeParams, bring_in_list:List[BboxObject],wait_item_list:[BboxObject] )-> Dict[str, List[FrameObject]]:
        """
        物体検出ロジック
        :param yolox_bbox:
        :param started_at:
        :param frame_object_list:
        :param judge_params:
        :return: 検出したObjectリスト, 更新された既知マスク
        """
    
        def is_unknown_object(class_id: str, probability: float, object_threshold=0.48) -> bool:
            """物体と思われるものの規定の物体でないものかどうか調べる関数
            Args:
                class_id (str): 物体のクラス名
                probability (float): 物体かどうかの確からしさ（max 1）
                object_threshold (float): 物体と判定するしきい値（max 1）
            Returns:
                bool: 物体と思われるものの規定の物体でないものかどうか
            """
            DEFAULT_OBJECTS = ['person','chair','laptop','tv','microwave','refrigerator','potted plant','cup','keyboard','couch','mouse','sink','book','dining table']
            is_object: bool = probability > object_threshold
            is_default_object = class_id in DEFAULT_OBJECTS
            return is_object and not(is_default_object)
        
        def is_people_object(class_id: str, probability: float, object_threshold=0.48) -> bool:
            """物体が人物であるかどうか調べる関数
            Args:
                class_id (str): 物体のクラス名
                probability (float): 物体かどうかの確からしさ（max 1）
                object_threshold (float): 物体と判定するしきい値（max 1）
            Returns:
                bool: 物体が人物であるかどうか
            """
            PEOPLE_OBJECTS = ['person']
            is_object: bool = probability > object_threshold
            is_people_object = class_id in PEOPLE_OBJECTS 
            return is_object and (is_people_object)

        def judge_take_out_object(bring_in_item, threshold=0.2) -> bool:
            """持ち去り判定を行う関数
            Args:
                bring_in_item (): 持ち去るかどうか決める対象のアイテム
                threshold (float): これ以下だと持ち去ると決めるためのしきい値
            """
            # 検知率(履歴リスト中のTrueの存在率)を計算
            found_rate = sum(bring_in_item.fhist) / len(bring_in_item.fhist)
            # 検知率が15%未満だったら
            return found_rate < threshold

        def event_of_take_out(bring_in_item) -> None:
            """持ち去りイベントを発生させる関数
            Args:
                bring_in_item (): 持ち去りたい対象のアイテム
            """
            item = FrameObjectItem(
                        DetectedObjectActionEnum.TAKE_OUT,
                        bring_in_item._bounding_box,
                        bring_in_item._size,
                        bring_in_item._mask, 
                        bring_in_item._found_at,
                        bring_in_item._class_id
                    )
        
        # ラベリング処理
        
        prev_frame_object_dict = {}
        bbox_compare_list:List[BboxObject] = []
        union_find_tree: UnionFindTree[FrameObjectItem] = UnionFindTree[FrameObjectItem]()
        frame_object_item_list = []
        result = defaultdict(list)
        
        #del_idx_reverse = []
        is_exist_start = False
        is_exist_wait = False
        is_exist_bring = False
        
        hide_judge = False

        # 検知が終了しているものは除外
        for frame_object in frame_object_list:
            if frame_object.is_finished():
                result[str(frame_object.item.detected_at)].append(frame_object)
            else:
                prev_frame_object_dict[frame_object.item] = frame_object
        frame_object_list = list(prev_frame_object_dict.values())

        yolox_bboxes = yolox_bbox.bounding_boxes #yolox-rosから受け取った物体集合から物体一つずつ取り出す
        FHIST_SIZE = 10 # 検知履歴を遡って参照する範囲
        
        # 届いた現フレームのyolox-bbox群情報を整理して新しくリストにまとめる
        bbox_item_list = []
        bbox_people_list = []
        bbox_testobject_list = []
        for i, bbox in enumerate(yolox_bboxes):
            probability = bbox.probability
            x = bbox.xmin
            y = bbox.ymin
            xmax = bbox.xmax
            ymax = bbox.ymax
            height = ymax - y
            width = xmax - x
            class_id = bbox.class_id
            
            # if (probability < 0.48) or (class_id in ['person','chair','laptop','tv','microwave','refrigerator','potted plant','cup','keyboard','couch','mouse']):
            if is_unknown_object(class_id, probability):
                brack_img = np.zeros(color_img.shape[:2])
                brack_img[y:y + height, x:x + width] = 255
                mask_img:np.ndarray = brack_img[y:y + height, x:x + width]
                
                bounding_box = BoundingBox(x, y, width, height) # BBOX(左上端座標, 幅, 高さ)
                area = width*height # BBOXの面積
                
                bbox_item = BboxObject(bounding_box, area, mask_img, started_at,class_id)
                bbox_item_list.append(bbox_item)

                test_item = [x,y,xmax,ymax]

                #人物と物体の判定用のobject_list
                bbox_testobject_list.append(test_item)

            else:
                if is_people_object(class_id, probability):
                    brack_img = np.zeros(color_img.shape[:2])
                    brack_img[y:y + height, x:x + width] = 255
                    mask_img:np.ndarray = brack_img[y:y + height, x:x + width]
                    
                    bounding_box = BoundingBox(x, y, width, height) # BBOX(左上端座標, 幅, 高さ)
                    area = width*height # BBOXの面積
                    
                    people_item = [x,y,xmax,ymax]
                    bbox_people_list.append(bounding_box)


        # if bring_in_list:
            del_idx_list = []
            # 持ち込み確定リストと現フレームリストを全照合
            for i, bring_in_item in enumerate(bring_in_list):
                for bbox_item in bbox_item_list:                                                   
                    #if bbox_item.is_exist_start: # 初期状態リストにすでに存在する現フレームアイテムは無視
                        #continue+

                    if bring_in_item.is_match(bbox_item): # その持ち込みアイテムと一致する現フレームアイテムがあったら
                        bring_in_item.fhist.append(True) # その持ち込みアイテムの検知履歴リストにTrueを追加
                        bbox_item.is_exist_bring = True # その現フレームアイテムの「持ち込み確定リストに存在する？」フラグをオン
                        break

                       
                    else: #持ち込みアイテムと一致する現フレームアイテムがないとき
                        
                        if len(people.pose_key_points_list) == 0:
                            
                            bring_in_item.fhist.append(False)
                        
                        else:


                            #人物に隠れていないかを確かめる
                            for person in people.pose_key_points_list:
                                
                                #color_imgのサイズ
                                img_height, img_width = color_img.shape[:2]

                                bounding_box = person.bounding_box
                                left = np.clip(int(bounding_box.x), 0, img_width- 1)
                                top = np.clip(int(bounding_box.y), 0, img_height - 1)
                                right = np.clip(int(bounding_box.x + bounding_box.width), 0, img_width - 1)
                                bottom = np.clip(int(bounding_box.y + bounding_box.height), 0, img_height - 1)
                                part_count: int  = len(person.point_data)
                                
                                for part in range(part_count):
                                # 対象の位置
                                    pixel_point = person.point_data[part].pixel_point
                                    x = np.clip(int(pixel_point.x), 0, img_width - 1)
                                    y = np.clip(int(pixel_point.y), 0, img_height - 1)
                                    
                                    POSE_PAIRS:List[List[int]] =  [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[8,12],[9,10],[10,11],[11,22],[11,24],[12,13],[13,14],[14,19],[14,21],[15,0],[15,17],[16,0],[16,18],[19,20],[22,11],[22,23]]
                                    # Draw Skeleton
                                    for pair in POSE_PAIRS:
                                        partA:int  = pair[0]
                                        partB:int  = pair[1]

                                        if person.point_data[partA].pixel_point.x > 0 and person.point_data[partA].pixel_point.y > 0 and  person.point_data[partB].pixel_point.x>0  and person.point_data[partB].pixel_point.x>0  :
                                            segment: tuple[float, float, float, float] =[person.point_data[partA].pixel_point.x,person.point_data[partA].pixel_point.y,person.point_data[partB].pixel_point.x,person.point_data[partB].pixel_point.y]
                                            bounding_box_src = bring_in_item._bounding_box
                                            b_item_x, b_item_y, b_item_width, b_item_height = bounding_box_src.items
                                            rectangle: tuple[float, float, float, float] = [b_item_x,b_item_y ,b_item_width,b_item_height]

                                            #骨格と持ち込み物体の重なり判定
                                            if YoloxObjectDetectionLogic.chickhide(rectangle,segment):

                                                hide_judge = True
                                                break
                                            else:

                                                hide_judge = False

                            if not hide_judge:                         
                                bring_in_item.fhist.append(False) # 持ち込み物体が持ち去られているなら検知履歴リストにFalseを追加

                    
                if len(bring_in_item.fhist) >= FHIST_SIZE: # その持ち込みアイテムの検知履歴が十分に溜まっていたら
                    if judge_take_out_object(bring_in_item):
                        print('take_out')
                        del_idx_list.append(i) # 持ち去りイベント発生(持ち込み確定リストから削除予約)
                        action = DetectedObjectActionEnum.TAKE_OUT
                        item = FrameObjectItem(
                            action,
                            bring_in_item._bounding_box,
                            bring_in_item._size,
                            bring_in_item._mask, 
                            bring_in_item._found_at,
                            bring_in_item._class_id
                        )
                        frame_object_item_list.append(item)
                        for prev_item, frame_object in prev_frame_object_dict.items():
                            is_matched, size = prev_item.is_match(item)
                            if is_matched:
                                if not union_find_tree.has_item(prev_item):
                                    union_find_tree.add(prev_item)
                                    frame_object_list.remove(frame_object)
                                if not union_find_tree.has_item(item):
                                    union_find_tree.add(item)
                                    frame_object_item_list.remove(item)
                                union_find_tree.unite(prev_item, item)
                    bring_in_item.fhist = bring_in_item.fhist[-(FHIST_SIZE-1):]  # その持ち込みアイテムの検知履歴リストを最新分のみ確保して更新
                #持ち去られたアイテムを持ち込み確定リストから削除
            if del_idx_list:
                for di in reversed(del_idx_list):
                    del bring_in_list[di]
                    
        if wait_item_list:
            del_idx_list = []
            # 待機リストと現フレームリストを全照合
            for i, wait_item in enumerate(wait_item_list):
                for bbox_item in bbox_item_list:
                    if bbox_item.is_exist_bring: # 持ち込み確定リストにすでに存在する現フレームアイテムは無視
                        continue
                    if wait_item.is_match(bbox_item): #その待機アイテムと一致する現フレームアイテムがあったら
                        wait_item.fhist.append(True) # その待機アイテムの検知履歴リストにTrueを追加
                        bbox_item.is_exist_wait = True # その現フレームアイテムの「待機リストに存在する？」フラグをオン
                        break
                else:
                    wait_item.fhist.append(False) #最後までどれとも一致しなかったらその待機アイテムの検知履歴リストにFalseを追加
                    
                if len(wait_item.fhist) >= FHIST_SIZE: # その待機アイテムの検知履歴が十分に溜まっていたら
                    found_rate = sum(wait_item.fhist) / len(wait_item.fhist) # 検知率(検知履歴リスト中のTrueの存在率)を計算
                    if (found_rate < 0.2) or (found_rate > 0.7): # 検知率が20%未満 or 70%超過だったら
                        del_idx_list.append(i) # 幻だった or 持ち込みイベント発生(待機リストから削除予約)
                    if found_rate > 0.7: # 持ち込みイベント発生の場合
                        action = DetectedObjectActionEnum.BRING_IN
                        item = FrameObjectItem(
                            action,
                            wait_item._bounding_box, 
                            wait_item._size, 
                            wait_item._mask, 
                            wait_item._found_at,
                            wait_item._class_id
                        )
                        frame_object_item_list.append(item)
                        bring_in_list.append(wait_item)
                        for prev_item, frame_object in prev_frame_object_dict.items():
                            is_matched, size = prev_item.is_match(item)
                            if is_matched:
                                if not union_find_tree.has_item(prev_item):
                                    union_find_tree.add(prev_item)
                                    frame_object_list.remove(frame_object)
                                if not union_find_tree.has_item(item):
                                    union_find_tree.add(item)
                                    frame_object_item_list.remove(item)
                                union_find_tree.unite(prev_item, item)
                    wait_item.fhist = wait_item.fhist[-(FHIST_SIZE-1):] # その待機アイテムの検知履歴リストを最新分のみ確保して更新
            # 持ち込まれた or 幻だったアイテムを待機リストから削除
            if del_idx_list:
                for di in reversed(del_idx_list):
                    del wait_item_list[di]
                    
        # 初期状態リスト・持ち込み確定リスト・待機リストいずれにも存在しない現フレームアイテムは、待機リストに追加
        for bbox_item in bbox_item_list:
            if not(bbox_item.is_exist_bring or bbox_item.is_exist_wait):
                #print(f'wait_item_append : {bbox_item._class_id}')
                wait_item_list.append(bbox_item)
                #wait = [[i._class_id, i._bounding_box._x, i._bounding_box._y, i._found_count, i._not_found_count] for i in wait_item_list]
                #_ = [print(w) for w in wait]
                
            
                    
        # リンクした範囲を1つにまとめる
        groups = union_find_tree.all_group_members().values()
        for items in groups:
            new_item: FrameObjectItem = items[0]
            mask_img = YoloxObjectDetectionLogic.update_mask_image(np.zeros(color_img.shape[:2]),new_item)
            for item in items[1:]:
                new_item, mask_img = YoloxObjectDetectionLogic.update_item(new_item, item, mask_img)
            result[str(new_item.detected_at)].append(FrameObject(new_item, judge_params.allow_empty_frame_count))
            
        # リンクしなかったframe_objectは空のフレームを挟む
        for frame_object in frame_object_list:
            frame_object.add_empty_frame()
            result[str(frame_object.item.detected_at)].append(frame_object)
        
        # リンクしなかったframe_object_itemは新たなframe_objectとして登録
        for frame_object_item in frame_object_item_list:
            frame_object = FrameObject(frame_object_item, judge_params.allow_empty_frame_count)
            result[str(frame_object_item.detected_at)].append(frame_object)
            
        return result,bring_in_list,wait_item_list,bbox_people_list,bbox_testobject_list
    
    @staticmethod
    def update_item(left: FrameObjectItem, right: FrameObjectItem, mask_img: np.ndarray) -> Tuple[FrameObjectItem, np.ndarray]:
        x = min(left.bounding_box.x, right.bounding_box.x)
        y = min(left.bounding_box.y, right.bounding_box.y)
        width = max(left.bounding_box.x + left.bounding_box.width,right.bounding_box.x + right.bounding_box.width) - x
        height = max(left.bounding_box.y + left.bounding_box.height,right.bounding_box.y + right.bounding_box.height) - y
        mask_img = YoloxObjectDetectionLogic.update_mask_image(mask_img, right)
        size = np.count_nonzero(mask_img[y:y + height, x:x + width])
        
        new_bounding_box = BoundingBox(x, y, width, height)
        
        action = left.action
        left_is_before = left.detected_at.is_before(right.detected_at)
        
        # 持ち込み時は新しい方を選択
        new_detected_at = left.detected_at if left_is_before else right.detected_at
        new_class_id = left._class_id if left_is_before else right.detected_at
        
        return FrameObjectItem(action, new_bounding_box, size, mask_img[y:y + height, x:x + width],new_detected_at,new_class_id), mask_img
    
    @staticmethod
    def update_mask_image(mask_img: np.ndarray, item: FrameObjectItem) -> np.ndarray:
        _, bounding_box, _, mask, _ ,_= item.items
        x, y, width, height = bounding_box.items
        mask_img[y:y + height, x:x + width] = np.where(mask > 0, mask, mask_img[y:y + height, x:x + width])
        return mask_img
    
    # @staticmethod
    # def intersect(a, b) :
    #     ax_mn, ay_mn, ax_mx, ay_mx = a[0], a[1], a[2], a[3]
    #     bx_mn, by_mn, bx_mx, by_mx = b[0], b[1], b[2], b[3]
    #     return (ax_mn <= bx_mx and  ax_mx >= bx_mn) and (ay_mn <=  by_mx and ay_mx >= by_mn) # and (a.minZ <= b.maxZ and a.maxZ >= b.minZ)
    
    @staticmethod
    def is_overlap(Personbox, Bring_inbox):
        """物体と思われるものの規定の物体でないものかどうか調べる関数
        Args:
            Personbox: 人物のバウンディングボックスの座標 (左上隅のx, y座標, 右下隅のx, y座標)
            Bring_inbox: Bring_in判定されているバウンディングボックスの座標 (左上隅のx, y座標, 右下隅のx, y座標)
        Returns:
            bool: 重なっている場合はTrue、そうでない場合はFalseを返します
        """
        px_mn, py_mn, px_mx, py_mx = Personbox[0], Personbox[1], Personbox[2],Personbox[3]
        bx_mn, by_mn, bx_mx, by_mx = Bring_inbox[0], Bring_inbox[1], Bring_inbox[2], Bring_inbox[3]
        # 2つのバウンディングボックスが重なっているかどうかを判定
        if (px_mn <= bx_mx and  px_mx >= bx_mn) and (py_mn <=  by_mx and py_mx >= by_mn):
            return True
        else: 
            return False
        
    @staticmethod
    def chickhide(rectangle, segment):
	# 矩形の頂点を取得
        rect_x, rect_y, rect_width, rect_height = rectangle
        rect_top_left = (rect_x, rect_y)
        rect_top_right = (rect_x + rect_width, rect_y)
        rect_bottom_left = (rect_x, rect_y + rect_height)
        rect_bottom_right = (rect_x + rect_width, rect_y + rect_height)

        # 線分の端点を取得
        seg_x1, seg_y1, seg_x2, seg_y2 = segment
        seg_start = (seg_x1, seg_y1)
        seg_end = (seg_x2, seg_y2)

        # 線分が矩形の内部にあるかチェック
        if YoloxObjectDetectionLogic.point_in_rectangle(seg_start, rectangle) or YoloxObjectDetectionLogic.point_in_rectangle(seg_end, rectangle):
            return True

        # 線分と矩形の各辺との交差判定
        if YoloxObjectDetectionLogic.line_segment_intersect(seg_start, seg_end, rect_top_left, rect_top_right):
            return True
        if YoloxObjectDetectionLogic.line_segment_intersect(seg_start, seg_end, rect_top_right, rect_bottom_right):
            return True
        if YoloxObjectDetectionLogic.line_segment_intersect(seg_start, seg_end, rect_bottom_right, rect_bottom_left):
            return True
        if YoloxObjectDetectionLogic.line_segment_intersect(seg_start, seg_end, rect_bottom_left, rect_top_left):
            return True

        return False


    @staticmethod
    def point_in_rectangle(point, rectangle):
        x, y = point
        rect_x, rect_y, rect_width, rect_height = rectangle
        return rect_x <= x <= rect_x + rect_width and rect_y <= y <= rect_y + rect_height

    @staticmethod
    def line_segment_intersect(seg1_start, seg1_end, seg2_start, seg2_end):
        # 2つの線分の方程式の係数を計算
        a1, b1, c1 = YoloxObjectDetectionLogic.line_equation(seg1_start, seg1_end)
        a2, b2, c2 = YoloxObjectDetectionLogic.line_equation(seg2_start, seg2_end)
        

        # 交差点の座標を計算
        try:
            x = (b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1)
            y = (a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1)
        except ZeroDivisionError:
            x = 0
            y = 0

        # 交差点が線分の内部にあるかチェック
        if (min(seg1_start[0], seg1_end[0]) <= x <= max(seg1_start[0], seg1_end[0]) and
                min(seg1_start[1], seg1_end[1]) <= y <= max(seg1_start[1], seg1_end[1]) and
                min(seg2_start[0], seg2_end[0]) <= x <= max(seg2_start[0], seg2_end[0]) and
                min(seg2_start[1], seg2_end[1]) <= y <= max(seg2_start[1], seg2_end[1])):
            return True

        return False
    @staticmethod
    def line_equation(start_point, end_point):
        x1, y1 = start_point
        x2, y2 = end_point
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        return a, b, c      

            
    