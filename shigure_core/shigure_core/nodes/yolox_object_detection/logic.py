from collections import defaultdict
from typing import List, Dict, Tuple
import copy
import cv2
import numpy as np

#from shigure_core_msgs.msg import BoundingBoxes,YoloxBoundingBox
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
    def execute(yolox_bbox: BoundingBoxes, started_at: Timestamp, color_img:np.ndarray, frame_object_list: List[FrameObject],
                judge_params: JudgeParams, bring_in_list:List[BboxObject],wait_item_list:[BboxObject] )-> Dict[str, List[FrameObject]]:
        """
        物体検出ロジック
        :param yolox_bbox:
        :param started_at:
        :param frame_object_list:
        :param judge_params:
        :return: 検出したObjectリスト, 更新された既知マスク
        """
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
        for i, bbox in enumerate(yolox_bboxes):
        	probability = bbox.probability
        	x = bbox.xmin
        	y = bbox.ymin
        	xmax = bbox.xmax
        	ymax = bbox.ymax
        	height = ymax - y
        	width = xmax - x
        	class_id = bbox.class_id
        	
        	if (probability < 0.48) or (class_id in ['person','chair','laptop','tv','microwave','refrigerator','potted plant','cup','keyboard','couch','mouse']):
        		pass
        	else:
        		brack_img = np.zeros(color_img.shape[:2])
        		brack_img[y:y + height, x:x + width] = 255
        		mask_img:np.ndarray = brack_img[y:y + height, x:x + width]
        		
        		bounding_box = BoundingBox(x, y, width, height) # BBOX(左上端座標, 幅, 高さ)
        		area = width*height # BBOXの面積
        		
        		bbox_item = BboxObject(bounding_box, area, mask_img, started_at,class_id)
        		bbox_item_list.append(bbox_item)
        		
        		
        if bring_in_list:
        	del_idx_list = []
        	# 持ち込み確定リストと現フレームリストを全照合
        	for i, bring_in_item in enumerate(bring_in_list):
        		for bbox_item in bbox_item_list:
        			#if bbox_item.is_exist_start: # 初期状態リストにすでに存在する現フレームアイテムは無視
        				#continue
        			if bring_in_item.is_match(bbox_item): # その持ち込みアイテムと一致する現フレームアイテムがあったら
        				bring_in_item.fhist.append(True) # その持ち込みアイテムの検知履歴リストにTrueを追加
        				bbox_item.is_exist_bring = True # その現フレームアイテムの「持ち込み確定リストに存在する？」フラグをオン
        				break
        		else:
        			bring_in_item.fhist.append(False) # 最後までどれとも一致しなかったらその持ち込みアイテムの検知履歴リストにFalseを追加
        			
        		if len(bring_in_item.fhist) >= FHIST_SIZE: # その持ち込みアイテムの検知履歴が十分に溜まっていたら
        			found_rate = sum(bring_in_item.fhist) / len(bring_in_item.fhist) # 検知率(履歴リスト中のTrueの存在率)を計算
        			if found_rate < 0.3: # 検知率が30%未満だったら
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
        			if (found_rate < 0.3) or (found_rate > 0.7): # 検知率が30%未満 or 70%超過だったら
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
        		print(f'wait_item_append : {bbox_item._class_id}')
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
        	
        return result,bring_in_list,wait_item_list
    
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
