# Shigure Core
![Shigure Core](https://img.shields.io/badge/shigure-core-red)

ROS2 による室内シーン変遷ロギングシステム。
カメラ映像から人物（顔認識による個人特定つき）と物体を追跡し、「**誰が・何を・どこへ**」動かしたか（持ち込み/持ち去りイベント）を MySQL に記録する。

Wiki : https://github.com/Rits-Interaction-Laboratory/shigure_core/wiki

## システム構成

```
[別PC] RealSense ─ /rs/color/compressed, /rs/aligned_depth_to_color/*
[別PC] OpenPose  ─ /openpose/pose_key_points
[別PC] YOLO11    ─ /bounding_boxes, /Segments
        │
        ▼ (本体PC)
 yolox_object_detection → object_tracking ─┐
 people_recognition（顔認識）               │
        ▼                                  ▼
 people_tracking ── /shigure/people_detection（people_id + face_name）
        │                                  │
        ▼                                  ▼
 contact_detection ── /shigure/contacted（people_id + face_name + action）
        ▼
 record_event ── MySQL（people.name に顔名、event に持込/持去を保存）
        │
 shigure_api ── Web表示（顔画像 / 追跡オーバーレイ / PCA / 認識イベント）
```

## 必要なもの

- ROS2 Humble（[公式インストール方法](https://docs.ros.org/en/humble/Installation.html)）/ Docker + Docker Compose
- Intel® RealSense™ D435 + [rs_ros2_python](https://github.com/Rits-Interaction-Laboratory/rs_ros2_python)
- [openpose_ros2](https://github.com/Rits-Interaction-Laboratory/openpose_ros2)（[docker版](https://github.com/Rits-Interaction-Laboratory/openpose_ros2_docker)）
- [people_detection_ros2](https://github.com/Rits-Interaction-Laboratory/people_detection_ros2)（[docker版](https://github.com/Rits-Interaction-Laboratory/people_detection_ros2_docker)）
- (OPTIONAL) [web_video_server](https://wiki.ros.org/web_video_server)

## ドキュメント

| ドキュメント | 内容 |
| :--- | :--- |
| [docs/setup-docker.md](docs/setup-docker.md) | **Docker でのセットアップ**（最短。全ノード + DB を一括起動） |
| [docs/setup-native.md](docs/setup-native.md) | **生 ROS2 環境でのセットアップ**（GPU 推論・開発向け） |
| [docs/usage.md](docs/usage.md) | **共通の使い方**：外部ノード起動 / 顔のユーザー登録 / 記録の開始・終了 / 動作確認 / トラブルシュート |

## クイックスタート（Docker）

```sh
git submodule update --init
cp .env.example .env      # DISPLAY と ROS_DOMAIN_ID を設定
xhost +local:docker
docker compose up --build
```

これで DB を含む全ノードが起動する。外部ノード（RealSense / OpenPose / YOLO11）の起動と、
記録開始信号の送信は [docs/usage.md](docs/usage.md) を参照。
