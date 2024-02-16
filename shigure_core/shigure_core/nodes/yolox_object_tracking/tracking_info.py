import datetime


class TrackingInfo:
    """物体追跡のためのデータクラス."""

    def __init__(self):
        self._object_num = 0
        self._id_prefix = f'{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}_'

        self._object_dict = {}

    def new_object_id(self) -> str:
        """新しい物体idを取得します."""
        self._object_num += 1
        return self._get_object_id(self._object_num)

    def _get_object_id(self, object_num: int) -> str:
        """物体idを取得します."""
        return f'{self._id_prefix}{object_num}'
        
        
    def old_object_id(self) -> str:
        """持ち去り物体idを取得します."""
        self._object_num = self._object_num
        return self._get_object_id(self._object_num)
        
    def is_match(self, other):
    	bbox_x = abs(self._bounding_box._x - other._bounding_box._x)
    	bbox_y = abs(self._bounding_box._y - other._bounding_box._y)
    	bbox_width = abs(self._bounding_box._width - other._bounding_box._width)
    	bbox_height = abs(self._bounding_box._height - other._bounding_box._height)
    	if (self._class_id==other._class_id)and(bbox_x < 10) and (bbox_y < 10)and(bbox_width < 10): #& bbox_width < 30 & bbox_height < 30:
    		self._found_at = other._found_at
    		return True
    	else:
    		
    		return False


    @property
    def object_dict(self) -> dict:
        """物体idをキーとする一覧を取得します."""
        return self._object_dict

    @object_dict.setter
    def object_dict(self, objkect_dict: dict) -> None:
        """物体一覧を更新します."""
        self._object_dict = objkect_dict
