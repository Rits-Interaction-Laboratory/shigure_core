import numpy as np


def convert_frame_to_uint8(frame_float32: np.ndarray, threshold: int = 3000) -> np.ndarray:
    """
    深度画像をcv2用にuint32のデータをuint8へ丸め込む
    デフォルトは3m

    :param frame_float32: uint32の深度画素データ
    :param threshold: 切り出す深度のしきい値
    :return: uint8に変換したframe配列データ
    """
    frame_float32 = np.where(frame_float32 > threshold, threshold, frame_float32)

    # cv2で表示するためにuint32をuint8へ変換
    # 近いほど白色に近づくように反転
    return (255 - frame_float32 * 255 / threshold).astype(np.uint8)
