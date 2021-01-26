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

    @property
    def object_dict(self) -> dict:
        """物体idをキーとする一覧を取得します."""
        return self._object_dict

    @object_dict.setter
    def object_dict(self, objkect_dict: dict) -> None:
        """物体一覧を更新します."""
        self._object_dict = objkect_dict
