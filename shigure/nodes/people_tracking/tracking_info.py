import datetime


class TrackingInfo:
    """人物追跡のためのデータクラス."""

    def __init__(self):
        self._people_num = 0
        self._id_prefix = f'{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}_'

        self._people_dict = {}

    def new_people_id(self) -> str:
        """新しい人物idを取得します."""
        self._people_num += 1
        return self._get_people_id(self._people_num)

    def _get_people_id(self, people_num: int) -> str:
        """人物idを取得します."""
        return f'{self._id_prefix}{people_num}'

    def get_people_dict(self) -> dict:
        """人物idをキーとする一覧を取得します."""
        return self._people_dict

    def update_people_dict(self, people_dict: dict) -> None:
        """人物一覧を更新します."""
        self._people_dict = people_dict
