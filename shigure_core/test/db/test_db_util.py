from unittest.mock import MagicMock, patch

import mysql.connector
import pytest

from shigure_core.db.db_util import DbConnectionError, get_cursor


@patch('shigure_core.db.db_util.mysql.connector.connect')
def test_get_cursor_raises_db_connection_error_on_connect_failure(mock_connect):
    mock_connect.side_effect = mysql.connector.Error('connection refused')

    with pytest.raises(DbConnectionError):
        with get_cursor():
            pass


@patch('shigure_core.db.db_util.mysql.connector.connect')
def test_get_cursor_commits_and_closes_on_success(mock_connect):
    fake_conn = MagicMock()
    fake_cursor = MagicMock()
    fake_conn.cursor.return_value = fake_cursor
    mock_connect.return_value = fake_conn

    with get_cursor(dictionary=True) as cur:
        assert cur is fake_cursor

    fake_conn.cursor.assert_called_once_with(dictionary=True)
    fake_conn.commit.assert_called_once()
    fake_conn.close.assert_called_once()


@patch('shigure_core.db.db_util.mysql.connector.connect')
def test_get_cursor_closes_connection_even_on_exception(mock_connect):
    fake_conn = MagicMock()
    fake_conn.cursor.return_value = MagicMock()
    mock_connect.return_value = fake_conn

    with pytest.raises(ValueError):
        with get_cursor():
            raise ValueError('boom')

    fake_conn.commit.assert_not_called()
    fake_conn.close.assert_called_once()
