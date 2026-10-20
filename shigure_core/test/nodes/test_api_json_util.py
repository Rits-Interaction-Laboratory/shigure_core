import datetime

from shigure_core.nodes.api_json_util import to_jsonable


def test_to_jsonable_converts_datetime_to_isoformat():
    now = datetime.datetime(2026, 8, 8, 12, 0, 0)

    result = to_jsonable({'object_id': '20260808120000_1', 'bring_in_at': now})

    assert result == {'object_id': '20260808120000_1', 'bring_in_at': now.isoformat()}


def test_to_jsonable_passes_through_non_datetime_values():
    result = to_jsonable({'x': 1.5, 'class_id': 'cup', 'take_out_at': None})

    assert result == {'x': 1.5, 'class_id': 'cup', 'take_out_at': None}
