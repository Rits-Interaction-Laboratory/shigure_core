import datetime


def to_jsonable(row: dict) -> dict:
    """DBの行(dict)をJSONで送れる形に変換する(datetimeをISO文字列化するのみ)."""
    return {
        key: (value.isoformat() if isinstance(value, datetime.datetime) else value)
        for key, value in row.items()
    }
