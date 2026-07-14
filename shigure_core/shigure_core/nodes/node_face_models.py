import os
import glob
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
import message_filters

from shigure_core.nodes.profile_insightface import ProfileInsightFace

# 顔特徴(.npy)と顔画像(.jpg)の保存先。people_recognition / shigure_api が読む辞書と同じ場所。
# SHIGURE_FACE_MODELS_DIR 環境変数があれば最優先。無ければ固定の永続パス ~/.shigure/face_models。
# （env 未設定でも全ノード・shigure_api が同じ場所を見る／ビルドで消えない／別PCでも同挙動）
_env_face_models_dir = os.environ.get('SHIGURE_FACE_MODELS_DIR')
FACE_MODELS_DIR = (
    os.path.expanduser(_env_face_models_dir)
    if _env_face_models_dir
    else os.path.expanduser('~/.shigure/face_models')
)


class FaceCaptureNode(Node):
    def __init__(self):
        super().__init__('face_capture_node')

        # QoS Settings
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # サブスクライバー設定
        self.image_sub = message_filters.Subscriber(self, CompressedImage, '/rs/color/compressed', qos_profile=qos_profile)
        self.sync = message_filters.TimeSynchronizer([self.image_sub], 10)
        self.sync.registerCallback(self.image_callback)

        # ユーザーに名前を入力させる（辞書のフォルダ名＝表示名になるため user_ 規約に従う）
        self.name = input("ユーザー名を入力してください: ")

        dir = "user_" + self.name

        # 保存ディレクトリ設定
        self.save_directory = os.path.join(FACE_MODELS_DIR, dir)
        os.makedirs(self.save_directory, exist_ok=True)
        self.get_logger().info(f"保存先: {self.save_directory}")

        # モデルの読み込み
        self.insightface = ProfileInsightFace()
        if not self.insightface.available:
            raise RuntimeError("insightface is not installed")

        # 撮影枚数のカウント
        self.feature_count = len(glob.glob(os.path.join(self.save_directory, "*.npy")))
        self.max_images = 100

    def image_callback(self, compressed_image):
        # 最大撮影枚数に達した場合、処理を終了する
        if self.feature_count >= self.max_images:
            self.get_logger().info(f"Reached {self.max_images} images, stopping the node.")
            rclpy.shutdown()
            return

        # 画像データを取得
        color_img = self.compressed_imgmsg_to_cv2(compressed_image)

        # 画像のリサイズ
        resized_image = cv2.resize(color_img, (1280, 720))

        # 顔検出
        detected_faces = self.insightface.detect_faces(resized_image)

        # 検出された顔がある場合
        if detected_faces is not None:
            for iface_face in detected_faces:
                if not iface_face.is_frontal:
                    continue

                # 顔の四角形の位置を取得
                x, y, w, h = iface_face.bbox
                print(x, y, w, h)

                # 画像の幅と高さ
                img_height, img_width = resized_image.shape[:2]

                # 顔の座標が画像範囲外にある場合、スキップ
                if x < 20 or y < 20 or x + w > img_width - 20 or y + h > img_height - 20:
                    print("Detected face is partially outside the screen. Skipping this face.")
                    continue  # 画面外の顔を無視

                # 顔の領域をクロップして特徴を抽出
                face_crop = resized_image[y:y + h, x:x + w]

                # 顔画像と特徴ベクトルを保存
                self.save_face_data(face_crop, iface_face.embedding)

    def save_face_data(self, face_image, feature_vector):
        # 保存するファイル名を設定
        self.feature_count += 1
        img_filename = os.path.join(self.save_directory, f"{self.name}{self.feature_count}.jpg")
        npy_filename = os.path.join(self.save_directory, f"{self.name}{self.feature_count}.npy")

        # 顔画像を保存 (JPEG)
        cv2.imwrite(img_filename, face_image)

        # 特徴ベクトルを保存 (NumPy)
        np.save(npy_filename, feature_vector)

        self.get_logger().info(f"Saved face image as {img_filename} and feature vector as {npy_filename}")

    def compressed_imgmsg_to_cv2(self, compressed_img_msg):
        # ROS2のCompressedImageメッセージをOpenCVの画像形式に変換する
        np_arr = np.frombuffer(compressed_img_msg.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return image_np


def main(args=None):
    rclpy.init(args=args)
    face_capture_node = FaceCaptureNode()

    try:
        rclpy.spin(face_capture_node)
    except KeyboardInterrupt:
        pass
    finally:
        face_capture_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
