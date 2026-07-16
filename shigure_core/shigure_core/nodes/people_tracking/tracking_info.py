import datetime


class TrackingInfo:
    """人物追跡のためのデータクラス.

    max_age フレームまでは「見失った人物」を coasting として保持し、再出現時に
    同じ people_id へ再マッチできるようにする（1フレームの取りこぼしで別人扱い＝
    IDが振り直され顔名の累積スコアがリセットされるのを抑制する）。

    - 可視人物 (get_people_dict): 現フレームで検出された人物。publish/描画の対象。
    - coasting: 直近で見失った人物。マッチ候補としてのみ使う（幽霊BBOXは出さない）。
    """

    def __init__(self, max_age: int = 0):
        self._people_num = 0
        self._id_prefix = f'{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}_'

        self._people_dict = {}  # 現フレームで可視の人物 {id: (x, y, z, key_points)}
        self._coasting = {}     # 直近で見失った人物 {id: (x, y, z, key_points)}
        self._age = {}          # {id: 連続で見失ったフレーム数}
        self._max_age = max(0, int(max_age))

    def set_max_age(self, max_age: int) -> None:
        """coasting で保持する最大フレーム数を更新する（0で従来動作＝保持しない）。"""
        self._max_age = max(0, int(max_age))

    def new_people_id(self) -> str:
        """新しい人物idを取得します."""
        self._people_num += 1
        return self._get_people_id(self._people_num)

    def _get_people_id(self, people_num: int) -> str:
        """人物idを取得します."""
        return f'{self._id_prefix}{people_num}'

    def get_people_dict(self) -> dict:
        """現フレームで可視の人物一覧（publish/描画の対象）を取得します."""
        return self._people_dict

    def get_match_pool(self) -> dict:
        """フレーム間マッチの候補（可視＋coasting）を取得します."""
        pool = dict(self._people_dict)
        pool.update(self._coasting)
        return pool

    def get_alive_ids(self) -> set:
        """履歴を残すべき生存ID（可視＋coasting）を取得します."""
        return set(self._people_dict.keys()) | set(self._coasting.keys())

    def commit(self, visible: dict) -> None:
        """追跡結果を反映する.

        :param visible: 現フレームで可視の人物 {id: (x, y, z, key_points)}。
            既存IDに再マッチした人物は元のIDを、新規人物は新IDをキーに持つ。
        """
        prev_pool = self.get_match_pool()
        new_coasting = {}
        new_age = {}
        for people_id, people in prev_pool.items():
            if people_id in visible:
                continue  # 今フレームで可視化 → coasting から外す（age リセット）
            age = self._age.get(people_id, 0) + 1
            if age <= self._max_age:
                new_coasting[people_id] = people  # 最後に見えた位置で保持
                new_age[people_id] = age
            # age > max_age は破棄（=完全にロストしたとみなす）

        self._people_dict = visible
        self._coasting = new_coasting
        self._age = new_age

    def update_people_dict(self, people_dict: dict) -> None:
        """人物一覧を更新します（後方互換; coasting を使わず可視のみ置き換える）."""
        self.commit(people_dict)
