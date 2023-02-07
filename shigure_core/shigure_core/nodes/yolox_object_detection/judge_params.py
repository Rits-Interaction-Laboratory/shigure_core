class JudgeParams:
    """判定に利用するパラメータ."""

    def __init__(self, allow_empty_frame_count: int):
        #self._min_size = min_size
        #self._max_size = max_size
        self._allow_empty_frame_count = allow_empty_frame_count

    @property
    def min_size(self) -> int:
        return self._min_size

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def allow_empty_frame_count(self) -> int:
        return self._allow_empty_frame_count
