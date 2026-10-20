import datetime

from shigure_core.db.db_util import get_cursor


class ObjectTrackingSummaryRepository:
    """object_idごとのbring_in/take_out時点の座標・時刻のみを1行にまとめる親テーブル用リポジトリ.

    object_tracking_event_history_<object_id> という無数に増える子テーブルに対して、
    「どのobject_idがいつ・どこでbring_inされ、いつ・どこでtake_outされたか」を
    一覧できるようにする。stay/obj_move時点の更新は行わない。
    """

    @staticmethod
    def insert_bring_in(object_id: str, class_id: str,
                        x: float, y: float, z: float,
                        width: float, height: float, depth: float,
                        bring_in_at: datetime.datetime):
        with get_cursor() as cur:
            cur.execute(
                "INSERT INTO object_tracking_summary "
                "(object_id, class_id, bring_in_at, bring_in_x, bring_in_y, bring_in_z, "
                "bring_in_width, bring_in_height, bring_in_depth) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (object_id, class_id, bring_in_at, x, y, z, width, height, depth))

    @staticmethod
    def update_take_out(object_id: str,
                        x: float, y: float, z: float,
                        width: float, height: float, depth: float,
                        take_out_at: datetime.datetime):
        with get_cursor() as cur:
            cur.execute(
                "UPDATE object_tracking_summary SET "
                "take_out_at = %s, take_out_x = %s, take_out_y = %s, take_out_z = %s, "
                "take_out_width = %s, take_out_height = %s, take_out_depth = %s "
                "WHERE object_id = %s AND take_out_at IS NULL",
                (take_out_at, x, y, z, width, height, depth, object_id))

    @staticmethod
    def fetch_all() -> list:
        """全object_idの親テーブル行をそのまま取得する."""
        with get_cursor(dictionary=True) as cur:
            cur.execute("SELECT * FROM object_tracking_summary")
            rows = cur.fetchall()
        return rows
