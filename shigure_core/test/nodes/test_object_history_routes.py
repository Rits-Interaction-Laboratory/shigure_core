import datetime
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from shigure_core.db.db_util import DbConnectionError
from shigure_core.nodes.object_history_routes import register_object_history_routes


def _build_client():
    app = FastAPI()
    register_object_history_routes(app)
    return TestClient(app)


@patch('shigure_core.nodes.object_history_routes.ObjectHistoryRepository.fetch_all')
def test_get_history_returns_all_rows_by_default(mock_fetch_all):
    now = datetime.datetime(2026, 8, 8, 12, 0, 0)
    mock_fetch_all.return_value = [
        {'object_id': '20260808120000_1', 'bring_in_at': now, 'take_out_at': None}]
    client = _build_client()

    response = client.get('/history')

    assert response.status_code == 200
    assert response.json() == [
        {'object_id': '20260808120000_1', 'bring_in_at': now.isoformat(), 'take_out_at': None}]
    mock_fetch_all.assert_called_once_with(open_only=False)


@patch('shigure_core.nodes.object_history_routes.ObjectHistoryRepository.fetch_all')
def test_get_history_open_only_passes_flag_through(mock_fetch_all):
    mock_fetch_all.return_value = []
    client = _build_client()

    response = client.get('/history', params={'open_only': 'true'})

    assert response.status_code == 200
    mock_fetch_all.assert_called_once_with(open_only=True)


@patch('shigure_core.nodes.object_history_routes.ObjectHistoryRepository.fetch_by_object_id')
@patch('shigure_core.nodes.object_history_routes.ObjectHistoryRepository.fetch_all')
def test_get_history_with_object_id_uses_fetch_by_object_id(mock_fetch_all, mock_fetch_by_id):
    mock_fetch_by_id.return_value = []
    client = _build_client()

    response = client.get('/history', params={'object_id': '20260808120000_1'})

    assert response.status_code == 200
    mock_fetch_by_id.assert_called_once_with('20260808120000_1')
    mock_fetch_all.assert_not_called()


@patch('shigure_core.nodes.object_history_routes.ObjectHistoryRepository.fetch_all')
def test_get_history_returns_503_when_db_unavailable(mock_fetch_all):
    mock_fetch_all.side_effect = DbConnectionError('connection refused')
    client = _build_client()

    response = client.get('/history')

    assert response.status_code == 503


@patch('shigure_core.nodes.object_history_routes.ObjectHistoryRepository.fetch_all')
def test_ws_history_streams_rows(mock_fetch_all):
    now = datetime.datetime(2026, 8, 8, 12, 0, 0)
    mock_fetch_all.return_value = [
        {'object_id': '20260808120000_1', 'bring_in_at': now, 'take_out_at': None}]
    client = _build_client()

    with client.websocket_connect('/ws/history') as websocket:
        data = websocket.receive_json()

    assert data == [
        {'object_id': '20260808120000_1', 'bring_in_at': now.isoformat(), 'take_out_at': None}]


@patch('shigure_core.nodes.object_history_routes.ObjectHistoryRepository.fetch_all')
def test_ws_history_sends_empty_list_when_db_unavailable(mock_fetch_all):
    mock_fetch_all.side_effect = DbConnectionError('connection refused')
    client = _build_client()

    with client.websocket_connect('/ws/history') as websocket:
        data = websocket.receive_json()

    assert data == []
