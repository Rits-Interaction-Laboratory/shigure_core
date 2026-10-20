import datetime

from shigure_core.db.db_util import get_cursor


class ObjectHistoryRepository:

    @staticmethod
    def has_open_row(object_id: str) -> bool:
        with get_cursor() as cur:
            cur.execute(
                "SELECT id FROM object_history WHERE object_id = %s AND take_out_at IS NULL LIMIT 1",
                (object_id,))
            row = cur.fetchone()
        return row is not None

    @staticmethod
    def insert_bring_in(object_id: str, class_id: str, x: float, y: float, z: float,
                        width: float, height: float, depth: float, now: datetime.datetime):
        with get_cursor() as cur:
            cur.execute(
                "INSERT INTO object_history"
                "(object_id, class_id, x, y, z, width, height, depth, bring_in_at, last_seen_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (object_id, class_id, x, y, z, width, height, depth, now, now))

    @staticmethod
    def update_stay(object_id: str, x: float, y: float, z: float,
                    width: float, height: float, depth: float, now: datetime.datetime):
        with get_cursor() as cur:
            cur.execute(
                "UPDATE object_history SET x = %s, y = %s, z = %s, "
                "width = %s, height = %s, depth = %s, last_seen_at = %s "
                "WHERE object_id = %s AND take_out_at IS NULL",
                (x, y, z, width, height, depth, now, object_id))

    @staticmethod
    def close_take_out(object_id: str, now: datetime.datetime):
        with get_cursor() as cur:
            cur.execute(
                "UPDATE object_history SET take_out_at = %s, last_seen_at = %s "
                "WHERE object_id = %s AND take_out_at IS NULL",
                (now, now, object_id))

    @staticmethod
    def fetch_all(open_only: bool = False) -> list:
        """object_historyの全行を取得する. open_only=Trueならtake_out_at IS NULLの行のみ."""
        query = "SELECT * FROM object_history"
        if open_only:
            query += " WHERE take_out_at IS NULL"
        query += " ORDER BY bring_in_at DESC"
        with get_cursor(dictionary=True) as cur:
            cur.execute(query)
            return cur.fetchall()

    @staticmethod
    def fetch_by_object_id(object_id: str) -> list:
        """指定object_idの行を取得する(bring_in/take_outが繰り返されている場合を考慮し配列で返す)."""
        with get_cursor(dictionary=True) as cur:
            cur.execute(
                "SELECT * FROM object_history WHERE object_id = %s ORDER BY bring_in_at DESC",
                (object_id,))
            return cur.fetchall()
