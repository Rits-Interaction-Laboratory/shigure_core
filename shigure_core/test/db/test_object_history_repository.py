import datetime
from unittest.mock import MagicMock, patch

from shigure_core.db.object_history_repository import ObjectHistoryRepository


def _mock_cursor(fetchone_result=None, fetchall_result=None):
    cur = MagicMock()
    cur.fetchone.return_value = fetchone_result
    cur.fetchall.return_value = fetchall_result if fetchall_result is not None else []
    ctx = MagicMock()
    ctx.__enter__.return_value = cur
    ctx.__exit__.return_value = False
    return ctx, cur


@patch('shigure_core.db.object_history_repository.get_cursor')
def test_has_open_row_true_when_row_found(mock_get_cursor):
    ctx, cur = _mock_cursor(fetchone_result={'id': 1})
    mock_get_cursor.return_value = ctx

    result = ObjectHistoryRepository.has_open_row('20260808120000_1')

    assert result is True
    args, _ = cur.execute.call_args
    assert 'take_out_at IS NULL' in args[0]
    assert args[1] == ('20260808120000_1',)


@patch('shigure_core.db.object_history_repository.get_cursor')
def test_has_open_row_false_when_no_row(mock_get_cursor):
    ctx, _ = _mock_cursor(fetchone_result=None)
    mock_get_cursor.return_value = ctx

    result = ObjectHistoryRepository.has_open_row('20260808120000_1')

    assert result is False


@patch('shigure_core.db.object_history_repository.get_cursor')
def test_insert_bring_in_passes_parameterized_values(mock_get_cursor):
    ctx, cur = _mock_cursor()
    mock_get_cursor.return_value = ctx
    now = datetime.datetime(2026, 8, 8, 12, 0, 0)

    ObjectHistoryRepository.insert_bring_in(
        '20260808120000_1', 'cup', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, now)

    args, _ = cur.execute.call_args
    assert 'INSERT INTO object_history' in args[0]
    assert args[1] == ('20260808120000_1', 'cup', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, now, now)


@patch('shigure_core.db.object_history_repository.get_cursor')
def test_update_stay_only_updates_open_row(mock_get_cursor):
    ctx, cur = _mock_cursor()
    mock_get_cursor.return_value = ctx
    now = datetime.datetime(2026, 8, 8, 12, 0, 0)

    ObjectHistoryRepository.update_stay(
        '20260808120000_1', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, now)

    args, _ = cur.execute.call_args
    assert 'take_out_at IS NULL' in args[0]
    assert args[1] == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, now, '20260808120000_1')


@patch('shigure_core.db.object_history_repository.get_cursor')
def test_close_take_out_sets_take_out_at(mock_get_cursor):
    ctx, cur = _mock_cursor()
    mock_get_cursor.return_value = ctx
    now = datetime.datetime(2026, 8, 8, 12, 0, 0)

    ObjectHistoryRepository.close_take_out('20260808120000_1', now)

    args, _ = cur.execute.call_args
    assert 'take_out_at = %s' in args[0]
    assert args[1] == (now, now, '20260808120000_1')


@patch('shigure_core.db.object_history_repository.get_cursor')
def test_fetch_all_without_open_only_has_no_where_clause(mock_get_cursor):
    rows = [{'object_id': '20260808120000_1'}]
    ctx, cur = _mock_cursor(fetchall_result=rows)
    mock_get_cursor.return_value = ctx

    result = ObjectHistoryRepository.fetch_all()

    assert result == rows
    args, _ = cur.execute.call_args
    assert 'WHERE' not in args[0]
    assert 'ORDER BY bring_in_at DESC' in args[0]


@patch('shigure_core.db.object_history_repository.get_cursor')
def test_fetch_all_open_only_adds_where_clause(mock_get_cursor):
    ctx, cur = _mock_cursor(fetchall_result=[])
    mock_get_cursor.return_value = ctx

    ObjectHistoryRepository.fetch_all(open_only=True)

    args, _ = cur.execute.call_args
    assert 'WHERE take_out_at IS NULL' in args[0]


@patch('shigure_core.db.object_history_repository.get_cursor')
def test_fetch_by_object_id_uses_parameterized_query(mock_get_cursor):
    rows = [{'object_id': '20260808120000_1'}]
    ctx, cur = _mock_cursor(fetchall_result=rows)
    mock_get_cursor.return_value = ctx

    result = ObjectHistoryRepository.fetch_by_object_id('20260808120000_1')

    assert result == rows
    args, _ = cur.execute.call_args
    assert 'WHERE object_id = %s' in args[0]
    assert args[1] == ('20260808120000_1',)
