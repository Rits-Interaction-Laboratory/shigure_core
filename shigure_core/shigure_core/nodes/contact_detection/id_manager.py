import datetime


class IdManager:
    """イベントIDのためのデータクラス."""

    def __init__(self):
        self._event_num = 0
        self._id_prefix = f'{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}_'

    def new_event_id(self) -> str:
        """新しい物体idを取得します."""
        self._event_num += 1
        return self._get_event_id(self._event_num)

    def _get_event_id(self, event_num: int) -> str:
        """物体idを取得します."""
        return f'{self._id_prefix}{event_num}'
