class Timestamp:
    """タイムスタンプを管理するクラス."""

    def __init__(self, sec: int, nano_sec: int):
        self._sec: int = sec
        self._nano_sec: int = nano_sec

    def __copy__(self):
        return Timestamp(self._sec, self._nano_sec)

    def __str__(self):
        return f'{self._sec}.{self._nano_sec}'

    @property
    def timestamp(self) -> (int, int):
        return self._sec, self._nano_sec

    def is_before(self, other) -> bool:
        """
        どちらの時間が前か判定します.

        :param other:
        :return: 引数の時間より前なら true, 後ろなら false
        """
        other_sec, other_nano_sec = other.timestamp
        if self._sec != other_sec:
            return self._sec < other_sec
        return self._nano_sec < other_nano_sec
