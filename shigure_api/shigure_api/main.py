"""Entry point: ROS bridge thread + uvicorn."""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

import uvicorn

from shigure_api.config import API_HOST, API_PORT, FACE_MODELS_DIR
from shigure_api.ros_bridge import run_ros_bridge


def main() -> None:
    parser = argparse.ArgumentParser(description='Shigure FastAPI + ROS event bridge')
    parser.add_argument('--host', default=API_HOST, help='Bind host (use 0.0.0.0 for LAN)')
    parser.add_argument('--port', type=int, default=API_PORT, help='HTTP/WebSocket port')
    parser.add_argument(
        '--face-models-dir',
        default=str(FACE_MODELS_DIR),
        help='Path to face_models directory',
    )
    # ros2 launch経由では --ros-args 等が付与されるため、未知の引数は無視する
    args, _ = parser.parse_known_args()

    import shigure_api.config as config
    from shigure_api.app import hub, init_pca_builder, pca_hub, tracking_debug_hub
    from shigure_api.pca_plot import state_to_dict

    config.FACE_MODELS_DIR = Path(args.face_models_dir).expanduser()
    pca_builder = init_pca_builder(config.FACE_MODELS_DIR)

    pca_hub.enqueue(state_to_dict(pca_builder.build_state()))

    stop_event = threading.Event()

    def on_event(event) -> None:
        hub.enqueue(event)

    def on_pca_payload(payload: dict) -> None:
        pca_hub.enqueue(payload)

    def on_tracking_debug(payload: dict) -> None:
        tracking_debug_hub.enqueue(payload)

    def on_score(user_id: str, score: float) -> None:
        hub.enqueue_score(user_id, score)

    ros_thread = threading.Thread(
        target=run_ros_bridge,
        args=(on_event, stop_event),
        kwargs={
            'pca_builder': pca_builder,
            'on_pca_payload': on_pca_payload,
            'on_tracking_debug': on_tracking_debug,
            'on_score': on_score,
        },
        name='shigure_ros_bridge',
        daemon=True,
    )
    ros_thread.start()

    print(
        f'Shigure API: http://{args.host}:{args.port}  '
        f'WebSocket: ws://{args.host}:{args.port}/ws/events  '
        f'PCA plot: ws://{args.host}:{args.port}/ws/pca_plot  '
        f'Tracking debug: ws://{args.host}:{args.port}/ws/tracking_debug  '
        f'face_models={config.FACE_MODELS_DIR}'
    )

    try:
        uvicorn.run(
            'shigure_api.app:app',
            host=args.host,
            port=args.port,
            log_level='info',
        )
    finally:
        stop_event.set()
        ros_thread.join(timeout=5.0)


if __name__ == '__main__':
    main()
