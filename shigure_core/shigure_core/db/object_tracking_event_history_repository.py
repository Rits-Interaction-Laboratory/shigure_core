import datetime
import re

import mysql.connector
from mysql.connector import errorcode

from shigure_core.db.db_util import get_cursor

# object_idは "YYYYMMDDHHMMSS_<連番>" 形式で発行される
# (shigure_core.nodes.object_tracking.tracking_info.TrackingInfo参照)。
# テーブル名にそのまま埋め込むため、この形式であることを厳密に検証してから使う。
_OBJECT_ID_PATTERN = re.compile(r'^[0-9]{14}_[0-9]+$')
_TABLE_PREFIX = 'object_tracking_event_history_'


class ObjectTrackingEventHistoryRepository:

    @staticmethod
    def _table_name(object_id: str) -> str:
        if not _OBJECT_ID_PATTERN.match(object_id):
            raise ValueError(f'unexpected object_id format: {object_id}')
        return f'{_TABLE_PREFIX}{object_id}'

    @staticmethod
    def ensure_table(object_id: str):
        """object_id専用のイベント履歴テーブルが無ければ作成する."""
        table = ObjectTrackingEventHistoryRepository._table_name(object_id)
        with get_cursor() as cur:
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS `{table}` ("
                "`id` INT NOT NULL AUTO_INCREMENT,"
                "`class_id` VARCHAR(100) NOT NULL,"
                "`action` VARCHAR(50) NOT NULL,"
                "`x` FLOAT NOT NULL,"
                "`y` FLOAT NOT NULL,"
                "`z` FLOAT NOT NULL,"
                "`width` FLOAT NOT NULL,"
                "`height` FLOAT NOT NULL,"
                "`depth` FLOAT NOT NULL,"
                "`event_at` DATETIME NOT NULL,"
                "PRIMARY KEY (`id`),"
                "INDEX (`event_at`),"
                "INDEX (`action`))")

    @staticmethod
    def insert_event(object_id: str, class_id: str, action: str,
                     x: float, y: float, z: float,
                     width: float, height: float, depth: float,
                     event_at: datetime.datetime):
        table = ObjectTrackingEventHistoryRepository._table_name(object_id)
        with get_cursor() as cur:
            cur.execute(
                f"INSERT INTO `{table}` "
                "(class_id, action, x, y, z, width, height, depth, event_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (class_id, action, x, y, z, width, height, depth, event_at))

    @staticmethod
    def fetch_events(object_id: str) -> list:
        """object_id専用の子テーブルの全行を取得する. object_idの形式が不正ならValueError.

        テーブルがまだ作成されていない(=そのobject_idのイベントが一度も記録されていない)
        場合は空リストを返す。
        """
        table = ObjectTrackingEventHistoryRepository._table_name(object_id)
        try:
            with get_cursor(dictionary=True) as cur:
                cur.execute(f"SELECT * FROM `{table}` ORDER BY event_at")
                rows = cur.fetchall()
        except mysql.connector.Error as e:
            if e.errno == errorcode.ER_NO_SUCH_TABLE:
                rows = []
            else:
                raise
        return rows
