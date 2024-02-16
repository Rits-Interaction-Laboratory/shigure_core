from typing import List, Tuple

import numpy as np


class ProjectionMatrix:
    """透視変換行列を表すクラス."""

    def __init__(self, projection_matrix: np.ndarray):
        self._projection_matrix = projection_matrix
        self._projection_matrix_inv = np.linalg.inv(projection_matrix)

    def projection_to_view(self, u: float, v: float, depth: int) -> Tuple[float, float, float]:
        """
        2次元座標からカメラを中心とする3次元座標に変換します.

        :param u: 画像上のx座標
        :param v: 画像上のy座標
        :param depth: uv位置での深度
        :return:
        """
        s = np.asarray([[u, v, 1]]).T
        m = (depth * np.matmul(self._projection_matrix_inv, s)).T
        return m[0, 0], m[0, 1], depth
