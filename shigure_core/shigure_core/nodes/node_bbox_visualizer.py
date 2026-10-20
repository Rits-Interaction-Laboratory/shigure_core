import asyncio
import threading

import cv2
import cv2.aruco as aruco
import message_filters
import numpy as np
import rclpy
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from geometry_msgs.msg import TransformStamped
from pydantic import BaseModel
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, CameraInfo
from tf2_ros import TransformBroadcaster
from bboxes_ex_msgs.msg import BoundingBoxes

from shigure_core.nodes.node_image_preview import ImagePreviewNode
from shigure_core.nodes.common_model.bounding_box import BoundingBox
from shigure_core.nodes.common_model.projection_matrix import ProjectionMatrix
from shigure_core.nodes.object_history_routes import register_object_history_routes
from shigure_core.nodes.object_tracking_ws_routes import register_object_tracking_ws_routes
from shigure_core.util import compressed_depth_util

# マーカーが見えなくなってから姿勢を無効化するまでのフレーム数
_MAX_LOST_FRAMES = 60

# /ws/detectionsをポーリングして配信する間隔(秒)
_DETECTIONS_WS_INTERVAL_SEC = 1.0


def _get_median_depth(depth_img: np.ndarray, bbox: BoundingBox) -> float:
    """BBOXの中心付近(20x20領域)の有効深度の中央値を返す。単位はmm。"""
    h, w = depth_img.shape[:2]
    x, y, width, height = bbox.items
    cx = np.clip(x + width // 2, 0, w - 1)
    cy = np.clip(y + height // 2, 0, h - 1)
    half = 10
    region = depth_img[
        max(cy - half, 0):min(cy + half, h),
        max(cx - half, 0):min(cx + half, w)
    ]
    valid = region[region > 0]
    return float(np.median(valid)) if len(valid) > 0 else 0.0


def _build_marker_to_camera(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """rvec/tvecからマーカー→カメラの4x4変換行列を作成。"""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


def _camera_to_marker(p_cam: np.ndarray, T_marker_to_cam: np.ndarray) -> np.ndarray:
    """カメラ座標系の点をマーカー座標系に変換。"""
    T_cam_to_marker = np.linalg.inv(T_marker_to_cam)
    p_h = np.array([*p_cam, 1.0])
    return (T_cam_to_marker @ p_h)[:3]


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """3x3回転行列を[x,y,z,w]クォータニオンに変換。"""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


class _DetectedObject(BaseModel):
    id: int
    class_id: str
    x: float
    y: float
    z: float


class _DetectionResponse(BaseModel):
    objects: list[_DetectedObject]


class BboxVisualizerNode(ImagePreviewNode):
    """カメラ画像にYOLOXのBBOXとARマーカー原点の3D座標を描画して配信するノード."""

    def __init__(self):
        super().__init__("bbox_visualizer_node")

        shigure_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.declare_parameter(
            'aruco_marker_id', 1,
            ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER,
                                description='原点として使うARマーカーのID'))
        self.declare_parameter(
            'aruco_marker_size', 0.2,
            ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
                                description='ARマーカーの一辺の長さ(m)'))
        self.declare_parameter(
            'show_window', True,
            ParameterDescriptor(type=ParameterType.PARAMETER_BOOL,
                                description='Falseにするとウィンドウ非表示（ヘッドレス運用時）'))
        self.declare_parameter(
            'api_host', '0.0.0.0',
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING,
                                description='検出結果配信用HTTP APIのbindホスト'))
        self.declare_parameter(
            'api_port', 8080,
            ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER,
                                description='検出結果配信用HTTP APIのポート'))

        self._marker_id: int = self.get_parameter('aruco_marker_id').value
        self._marker_size: float = self.get_parameter('aruco_marker_size').value
        self._show_window: bool = self.get_parameter('show_window').value
        self._api_host: str = self.get_parameter('api_host').value
        self._api_port: int = self.get_parameter('api_port').value

        self._aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_1000)
        self._aruco_params = aruco.DetectorParameters()
        self._aruco_detector = aruco.ArucoDetector(self._aruco_dict, self._aruco_params)

        self._T_marker_to_cam: np.ndarray | None = None
        self._lost_frame_count: int = 0
        self._marker_status_logged: str | None = None
        self._class_colors: dict = {}
        self._tf_broadcaster = TransformBroadcaster(self)

        # 検出結果配信用HTTP API（インスタンス単位で状態を保持する）
        self._latest_detections: list[_DetectedObject] = []
        self._detection_lock = threading.Lock()
        self._api_app = FastAPI()
        self._api_app.add_middleware(
            # デバッグ・ローカルLAN内向けツールのため全許可。外部公開する場合は要制限。
            CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

        @self._api_app.get("/detections", response_model=_DetectionResponse)
        def _get_detections():
            with self._detection_lock:
                return _DetectionResponse(objects=list(self._latest_detections))

        @self._api_app.websocket("/ws/detections")
        async def _ws_detections(websocket: WebSocket):
            """検出結果(ARマーカー原点座標)を一定間隔でそのまま配信する."""
            await websocket.accept()
            try:
                while True:
                    with self._detection_lock:
                        objects = list(self._latest_detections)
                    await websocket.send_json(
                        _DetectionResponse(objects=objects).model_dump())
                    await asyncio.sleep(_DETECTIONS_WS_INTERVAL_SEC)
            except WebSocketDisconnect:
                pass

        # object_tracking_summary/event_historyのWebSocket配信も同じAPIサーバー(同じポート)に相乗りさせる
        register_object_tracking_ws_routes(self._api_app)
        # object_historyのREST/WebSocket配信も同じAPIサーバー(同じポート)に相乗りさせる
        register_object_history_routes(self._api_app)

        self._uvicorn_server: uvicorn.Server | None = None
        self._api_thread = threading.Thread(target=self._run_api_server, daemon=True)

        self.result_publisher = self.create_publisher(
            CompressedImage, '/shigure/bbox_visualization', 10)

        color_sub = message_filters.Subscriber(
            self, CompressedImage, '/rs/color/compressed', qos_profile=shigure_qos)
        bbox_sub = message_filters.Subscriber(
            self, BoundingBoxes, '/bounding_boxes', qos_profile=shigure_qos)
        depth_sub = message_filters.Subscriber(
            self, CompressedImage, '/rs/aligned_depth_to_color/compressedDepth',
            qos_profile=shigure_qos)
        camera_info_sub = message_filters.Subscriber(
            self, CameraInfo, '/rs/aligned_depth_to_color/cameraInfo',
            qos_profile=shigure_qos)

        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [color_sub, bbox_sub, depth_sub, camera_info_sub],
            queue_size=30, slop=0.1
        )
        self.time_synchronizer.registerCallback(self.callback)

        if self._show_window:
            cv2.namedWindow('bbox_visualizer', cv2.WINDOW_NORMAL)

        self.get_logger().info(
            f'BboxVisualizerNode started  marker_id={self._marker_id}  '
            f'marker_size={self._marker_size}m')

        self._api_thread.start()
        self.get_logger().info(
            f"FastAPI server: http://{self._api_host}:{self._api_port}/detections")

    # ------------------------------------------------------------------
    def _run_api_server(self):
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            config = uvicorn.Config(
                self._api_app, host=self._api_host, port=self._api_port, log_level="info")
            self._uvicorn_server = uvicorn.Server(config)
            loop.run_until_complete(self._uvicorn_server.serve())
        except Exception as e:
            import traceback
            self.get_logger().error(f"[FastAPI] Failed to start: {e}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    def destroy_node(self):
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
            self._api_thread.join(timeout=5.0)

        if self._show_window:
            cv2.destroyWindow('bbox_visualizer')

        super().destroy_node()

    # ------------------------------------------------------------------
    def _broadcast_marker_tf(self, stamp, R: np.ndarray, t: np.ndarray):
        """camera -> aruco_marker のtfを配信する（他ノードがワールド座標変換に使う）。"""
        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = 'camera'
        tf_msg.child_frame_id = 'aruco_marker'
        tf_msg.transform.translation.x = float(t[0])
        tf_msg.transform.translation.y = float(t[1])
        tf_msg.transform.translation.z = float(t[2])
        qx, qy, qz, qw = _rotation_matrix_to_quaternion(R)
        tf_msg.transform.rotation.x = float(qx)
        tf_msg.transform.rotation.y = float(qy)
        tf_msg.transform.rotation.z = float(qz)
        tf_msg.transform.rotation.w = float(qw)
        self._tf_broadcaster.sendTransform(tf_msg)

    # ------------------------------------------------------------------
    def _estimate_pose(self, corners: np.ndarray,
                       camera_matrix: np.ndarray,
                       dist_coeffs: np.ndarray):
        """solvePnP でマーカー姿勢を推定する（estimatePoseSingleMarkers の代替）。
        コーナー順: 左上→右上→右下→左下"""
        s = self._marker_size / 2.0
        obj_pts = np.array([
            [-s,  s, 0],
            [ s,  s, 0],
            [ s, -s, 0],
            [-s, -s, 0],
        ], dtype=np.float32)
        _, rvec, tvec = cv2.solvePnP(
            obj_pts, corners.reshape(4, 2),
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        return rvec, tvec

    # ------------------------------------------------------------------
    def _update_marker_pose(self, color_img: np.ndarray,
                            camera_matrix: np.ndarray,
                            dist_coeffs: np.ndarray) -> tuple:
        """ArUcoマーカーを検出して変換行列を更新。(検出成功bool, 検出IDリスト)を返す。"""
        # グレースケール変換は不要（detectMarkersが内部で処理する）
        corners, ids, _ = self._aruco_detector.detectMarkers(color_img)
        if ids is None:
            return False, []

        detected_ids = ids.flatten().tolist()
        aruco.drawDetectedMarkers(color_img, corners, ids)

        # 描画ループと姿勢推定を分離：先に全マーカーの矩形を描画する
        for i, marker_id in enumerate(detected_ids):
            pts = corners[i].reshape((4, 2)).astype(int)
            x, y, w, h = cv2.boundingRect(pts)
            rect_color = (0, 255, 0) if marker_id == self._marker_id else (0, 165, 255)
            cv2.rectangle(color_img, (x, y), (x + w, y + h), rect_color, thickness=3)
            cv2.putText(color_img, f'ID:{marker_id}', (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, rect_color, thickness=2)

        # 対象IDの姿勢推定のみ実施
        if self._marker_id in detected_ids:
            idx = detected_ids.index(self._marker_id)
            try:
                rvec, tvec = self._estimate_pose(corners[idx], camera_matrix, dist_coeffs)
            except cv2.error as e:
                self.get_logger().warn(f'solvePnP failed: {e}')
                return False, detected_ids
            self._T_marker_to_cam = _build_marker_to_camera(rvec, tvec)
            return True, detected_ids

        return False, detected_ids

    # ------------------------------------------------------------------
    def callback(self, color_img_src: CompressedImage, bboxes_src: BoundingBoxes,
                 depth_src: CompressedImage, camera_info: CameraInfo):
        self.frame_count_up()

        color_img: np.ndarray = self.bridge.compressed_imgmsg_to_cv2(color_img_src)
        try:
            depth_img: np.ndarray = compressed_depth_util.convert_compressed_depth_img_to_cv2(depth_src)
        except Exception as e:
            self.get_logger().warn(f'Failed to decode depth image, skipping frame: {e}')
            return
        img_h, img_w = color_img.shape[:2]

        try:
            camera_matrix = camera_info.k.reshape((3, 3))
        except ValueError as e:
            self.get_logger().warn(f'Invalid camera_info.k, skipping frame: {e}')
            return

        # dist_coeffs が空配列（rectified済み）の場合に備えてゼロ埋め
        dist_coeffs = np.array(camera_info.d, dtype=np.float64)
        if dist_coeffs.size == 0:
            dist_coeffs = np.zeros((5,), dtype=np.float64)

        projection = ProjectionMatrix(camera_matrix)

        result_img = color_img.copy()

        # ARマーカー検出・更新
        marker_found, detected_ids = self._update_marker_pose(result_img, camera_matrix, dist_coeffs)

        # LOST フレームカウント管理
        if marker_found:
            self._lost_frame_count = 0
        else:
            self._lost_frame_count += 1
            if self._lost_frame_count >= _MAX_LOST_FRAMES:
                self._T_marker_to_cam = None

        # マーカーの軸を描画 + tf配信（他ノードがcamera<->aruco_markerを参照できるようにする）
        if self._T_marker_to_cam is not None:
            R = self._T_marker_to_cam[:3, :3]
            t = self._T_marker_to_cam[:3, 3]
            rvec, _ = cv2.Rodrigues(R)
            cv2.drawFrameAxes(result_img, camera_matrix, dist_coeffs,
                              rvec, t, self._marker_size * 0.5)
            self._broadcast_marker_tf(color_img_src.header.stamp, R, t)

        # マーカー状態を右下に表示
        font = cv2.FONT_HERSHEY_SIMPLEX
        if marker_found:
            status_text = f'Marker ID {self._marker_id}: DETECTED'
            status_color = (0, 255, 0)
        elif self._T_marker_to_cam is not None:
            lost_sec = self._lost_frame_count / 30.0
            status_text = f'Marker ID {self._marker_id}: LOST ({lost_sec:.1f}s)'
            status_color = (0, 165, 255)
        else:
            status_text = f'Marker ID {self._marker_id}: NOT FOUND'
            status_color = (0, 0, 255)

        # ヘッドレス運用でも状態変化が追えるよう、変化した時だけログに残す
        if status_text != self._marker_status_logged:
            self.get_logger().info(status_text)
            self._marker_status_logged = status_text

        ids_text = f'Detected markers: {detected_ids}' if detected_ids else 'Detected markers: none'
        (sw, _), _ = cv2.getTextSize(status_text, font, 0.6, 2)
        (iw, _), _ = cv2.getTextSize(ids_text, font, 0.55, 1)
        cv2.putText(result_img, ids_text,
                    (img_w - iw - 10, img_h - 10), font, 0.55, (200, 200, 200), thickness=1)
        cv2.putText(result_img, status_text,
                    (img_w - sw - 10, img_h - 35), font, 0.6, status_color, thickness=2)

        # BBOX描画
        _frame_detections: list[_DetectedObject] = []
        _det_id = 1
        for bbox_msg in bboxes_src.bounding_boxes:
            x1 = int(np.clip(bbox_msg.xmin, 0, img_w - 1))
            y1 = int(np.clip(bbox_msg.ymin, 0, img_h - 1))
            x2 = int(np.clip(bbox_msg.xmax, 0, img_w - 1))
            y2 = int(np.clip(bbox_msg.ymax, 0, img_h - 1))
            bbox = BoundingBox(x1, y1, x2 - x1, y2 - y1)
            x, y, width, height = bbox.items

            label = bbox_msg.class_id
            prob = bbox_msg.probability
            color = self._get_class_color(label)

            cv2.rectangle(result_img, (x, y), (x + width, y + height), color, thickness=2)

            depth_mm = _get_median_depth(depth_img, bbox)
            cx = x + width // 2
            cy = y + height // 2

            if depth_mm > 0:
                x_mm, y_mm, z_mm = projection.projection_to_view(cx, cy, depth_mm)
                p_cam = np.array([x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0])

                if self._T_marker_to_cam is not None:
                    p_marker = _camera_to_marker(p_cam, self._T_marker_to_cam)
                    coord_text = (f'X:{p_marker[0]:.2f} '
                                  f'Y:{p_marker[1]:.2f} '
                                  f'Z:{p_marker[2]:.2f}m')
                    # APIはARマーカー座標系のみを公開する(マーカー未検出時は座標系が
                    # 定まらないため、カメラ座標系にフォールバックさせず配信対象から外す)
                    _frame_detections.append(_DetectedObject(
                        id=_det_id, class_id=label,
                        x=round(float(p_marker[0]), 2),
                        y=round(float(p_marker[1]), 2),
                        z=round(float(p_marker[2]), 2)))
                    _det_id += 1
                else:
                    coord_text = (f'X:{p_cam[0]:.2f} '
                                  f'Y:{p_cam[1]:.2f} '
                                  f'Z:{p_cam[2]:.2f}m [cam]')
            else:
                coord_text = 'X:-- Y:-- Z:--'

            label_text = f'{label}: {prob:.2f}  {coord_text}'
            font_scale = 0.55
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, 1)
            label_y = max(y - 4, text_h + baseline)
            cv2.rectangle(result_img,
                          (x, label_y - text_h - baseline),
                          (x + text_w, label_y + baseline),
                          color, thickness=-1)
            cv2.putText(result_img, label_text, (x, label_y),
                        font, font_scale, (255, 255, 255), thickness=1)

        with self._detection_lock:
            self._latest_detections[:] = _frame_detections

        result_img = self.print_fps(result_img)

        result_msg = self.bridge.cv2_to_compressed_imgmsg(result_img, 'jpg')
        result_msg.header = color_img_src.header
        self.result_publisher.publish(result_msg)

        if self._show_window:
            cv2.imshow('bbox_visualizer', result_img)
            cv2.waitKey(1)

    # ------------------------------------------------------------------
    def _get_class_color(self, class_id: str) -> tuple:
        if class_id not in self._class_colors:
            rng = np.random.default_rng(abs(hash(class_id)) % (2 ** 32))
            self._class_colors[class_id] = tuple(int(c) for c in rng.integers(64, 220, size=3))
        return self._class_colors[class_id]


def main(args=None):
    rclpy.init(args=args)
    node = BboxVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
