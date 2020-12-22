class Timestamp:
    """タイムスタンプを管理するクラス."""

    sec: int
    nano_sec: int

    def __init__(self, sec: int, nano_sec: int):
        self.sec = sec
        self.nano_sec = nano_sec

    def __copy__(self):
        return Timestamp(self.sec, self.nano_sec)

    def get_timestamp(self) -> (int, int):
        return self.sec, self.nano_sec
