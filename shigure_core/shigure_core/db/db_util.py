import logging
from contextlib import contextmanager

import mysql.connector

from shigure_core.db.config import config

_logger = logging.getLogger('shigure_core.db')


class DbConnectionError(RuntimeError):
    """MySQLへの接続に失敗したことを表す例外. 呼び出し元は用途に応じてハンドリングする."""


@contextmanager
def get_cursor(dictionary: bool = False):
    """コネクション/カーソルの生成とcommit・closeをまとめて面倒みるコンテキストマネージャ.

    接続自体に失敗した場合はDbConnectionErrorを送出する(呼び出し元で捕捉する)。
    with節を抜けるとき(例外時含め)必ずctx.close()する。
    """
    try:
        ctx = mysql.connector.connect(**config)
    except mysql.connector.Error as e:
        _logger.error(f'Failed to connect to MySQL: {e}')
        raise DbConnectionError(str(e)) from e

    try:
        yield ctx.cursor(dictionary=dictionary)
        ctx.commit()
    finally:
        ctx.close()
