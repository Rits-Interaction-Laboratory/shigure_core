from queue import Queue
from typing import Tuple, TypeVar, Generic

# ジェネリック用
T = TypeVar('T')
U = TypeVar('U')


class FrameCombiner(Generic[T, U]):
    """frameを時間で合成するクラス."""

    _left_queue: Queue
    _right_queue: Queue

    _left_head: Tuple[int, int, T]
    _right_head: Tuple[int, int, U]

    def __init__(self, round_sec: int = 0, round_nano_sec: int = 0):
        """
        コンストラクタ

        :param round_sec: 許容する秒数差(sec)
        :param round_nano_sec: 許容する秒数差(nano_sec) (max 999999999)
        """
        self._left_queue = Queue()
        self._right_queue = Queue()

        self._round_sec = round_sec
        self._round_nano_sec = round_nano_sec

    def is_same_time(self, left_sec: int, left_nano_sec: int, right_sec: int, right_nano_sec: int) -> bool:
        if self._round_sec == 0 and self._round_nano_sec == 0:
            return left_sec == right_sec and left_nano_sec == right_nano_sec
        if self._round_nano_sec == 0:
            return abs(left_sec - right_sec) <= self._round_sec
        if self._round_sec == 0:
            return left_sec == right_sec and abs(left_nano_sec - right_nano_sec) <= self._round_nano_sec
        if abs(left_sec - right_sec) < self._round_sec:
            return True
        return left_sec == right_sec and abs(left_nano_sec - right_nano_sec) <= self._round_nano_sec

    def enqueue_to_left_queue(self, sec: int, nano_sec: int, obj: T) -> None:
        """
        leftキューにエンキューします.

        :param sec:
        :param nano_sec:
        :param obj:
        :return:
        """
        self._left_queue.put((sec, nano_sec, obj))

        if not hasattr(self, '_left_head'):
            self._left_head = self._left_queue.get()

    def enqueue_to_right_queue(self, sec: int, nano_sec: int, obj: U) -> None:
        """
        rightキューにエンキューします.

        :param sec:
        :param nano_sec:
        :param obj:
        :return:
        """
        self._right_queue.put((sec, nano_sec, obj))

        if not hasattr(self, '_right_head'):
            self._right_head = self._right_queue.get()

    def dequeue(self) -> Tuple[bool, int, int, T, U]:
        """
        合成した結果を時系列順にデキューします.

        :return: (result, sec, nano_sec, obj1, obj2) のタプル
        """
        if not hasattr(self, '_left_head'):
            return False, -1, -1, None, None
        if not hasattr(self, '_right_head'):
            return False, -1, -1, None, None

        left_sec, left_nano_sec, left_obj = self._left_head
        right_sec, right_nano_sec, right_obj = self._right_head

        # 同時刻の画像を取得する
        while not self.is_same_time(left_sec, left_nano_sec, right_sec, right_nano_sec):
            # leftの時刻が新しい -> rightが古いので一つ進める
            if left_sec > right_sec:
                if self._right_queue.empty():
                    return False, -1, -1, None, None
                self._right_head = self._right_queue.get()
                right_sec, right_nano_sec, right_obj = self._right_head
                continue
            if left_sec < right_sec:
                if self._left_queue.empty():
                    return False, -1, -1, None, None
                self._left_head = self._left_queue.get()
                left_sec, left_nano_sec, left_obj = self._left_head
                continue

            # leftの時刻が新しい -> rightが古いので一つ進める
            if left_nano_sec > right_nano_sec:
                if self._right_queue.empty():
                    return False, -1, -1, None, None
                self._right_head = self._right_queue.get()
                right_sec, right_nano_sec, right_obj = self._right_head
                continue
            else:
                if self._left_queue.empty():
                    return False, -1, -1, None, None
                self._left_head = self._left_queue.get()
                left_sec, left_nano_sec, left_obj = self._left_head
                continue

        if not self._left_queue.empty():
            self._left_head = self._left_queue.get()
        if not self._right_queue.empty():
            self._right_head = self._right_queue.get()

        return True, left_sec, left_nano_sec, left_obj, right_obj

    def dequeue_with_left(self) -> Tuple[bool, int, int, T, U]:
        """
        leftを基準にデキューします.

        :return:
        """
        if not hasattr(self, '_left_head'):
            return False, -1, -1, None, None
        if not hasattr(self, '_right_head'):
            return False, -1, -1, None, None

        left_sec, left_nano_sec, left_obj = self._left_head
        right_sec, right_nano_sec, right_obj = self._right_head

        while not self.is_same_time(left_sec, left_nano_sec, right_sec, right_nano_sec):
            # leftが古い場合はそのまま返す
            if left_sec < right_sec or (left_sec == right_sec and left_nano_sec < right_nano_sec):
                if not self._left_queue.empty():
                    self._left_head = self._left_queue.get()
                return True, left_sec, left_nano_sec, left_obj, None

            if not self._right_queue.empty():
                self._right_head = self._right_queue.get()
                right_sec, right_nano_sec, right_obj = self._right_head

        if not self._left_queue.empty():
            self._left_head = self._left_queue.get()
        if not self._right_queue.empty():
            self._right_head = self._right_queue.get()

        return True, left_sec, left_nano_sec, left_obj, right_obj

    def dequeue_with_right(self) -> Tuple[bool, int, int, T, U]:
        """
        rightを基準にデキューします.

        :return:
        """
        if not hasattr(self, '_left_head'):
            return False, -1, -1, None, None
        if not hasattr(self, '_right_head'):
            return False, -1, -1, None, None

        left_sec, left_nano_sec, left_obj = self._left_head
        right_sec, right_nano_sec, right_obj = self._right_head

        while not self.is_same_time(left_sec, left_nano_sec, right_sec, right_nano_sec):
            # rightが古い場合はそのまま返す
            if left_sec > right_sec or (left_sec == right_sec and left_nano_sec > right_nano_sec):
                if not self._right_queue.empty():
                    self._left_head = self._right_queue.get()
                return True, right_sec, right_nano_sec, None, right_obj

            if not self._left_queue.empty():
                self._left_head = self._left_queue.get()
                left_sec, left_nano_sec, left_obj = self._left_head

        if not self._left_queue.empty():
            self._left_head = self._left_queue.get()
        if not self._right_queue.empty():
            self._right_head = self._right_queue.get()

        return True, right_sec, right_nano_sec, left_obj, right_obj
