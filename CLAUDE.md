# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

Shigure Core は ROS2 (Humble) による室内シーン変遷ロギングシステム。カメラ画像から人物・物体を検出・追跡し、「誰が何をどこへ動かしたか」（持ち込み/持ち去り/接触イベント）を MySQL に記録する。

## コードスタイル

- コメント・docstring は日本語
- flake8 / pep257 準拠（`colcon test` で強制される）

## コマンド

ROS2 ワークスペースの `src/` 配下に置いて colcon でビルドする前提。

```sh
git submodule update --init    # 初回クローン後に必須（bbox_ex_msgs）

# ビルド（ワークスペースルートで）
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash

# 起動
ros2 launch shigure_core shigure_core_launch.py   # 全ノード
ros2 run shigure_core yolox_object_detection       # 単一ノード
ros2 run shigure_core record_event --ros-args --params-file <params.yml>
# パラメータのサンプル: shigure_core/shigure_core/nodes/params/*.yml.sample

# テスト（flake8 / pep257 のリンタテストのみ）
colcon test --packages-select shigure_core && colcon test-result --verbose

# Docker（shigure_core + MySQL + マイグレーションが起動）
cp .env.example .env   # DISPLAY と ROS_DOMAIN_ID を設定
xhost +local:docker
docker compose up --build
docker compose down -v   # DB 含め完全リセット
```

## アーキテクチャ

### パッケージ構成

| パッケージ | 種別 | 内容 |
| :--- | :--- | :--- |
| `shigure_core/` | ament_python | 全ノード実装（本体） |
| `shigure_core_msgs/` | ament_cmake | 独自メッセージ定義（.msg） |
| `bbox_ex_msgs/` | git submodule | YOLOX 検出結果メッセージ（fork 側で編集する。直接編集禁止） |

`shigure_core` はパッケージと Python モジュールが同名でネストしている点に注意:

```
shigure_core/            # ament_python パッケージ（setup.py はここ）
├── resource/db/         # マイグレーション SQL・DB 用 docker-compose
└── shigure_core/        # Python モジュール（同名ネスト）
    ├── nodes/           # node_<name>.py（Node層）+ <name>/（Logic層）
    ├── launch/          # 用途別 launch 7種
    ├── db/              # MySQL アクセス層
    └── util/            # 深度画像・座標変換ユーティリティ
```

### データフロー（パイプライン）

外部ノード（RealSense `rs_camera`・OpenPose・YOLOX・people_detection は別リポジトリ）から入力を受け、内部トピックは `/shigure/` プレフィックスで流れる:

```
YOLOX (/bounding_boxes) → yolox_object_detection → object_tracking ─┐
OpenPose (/openpose/pose_key_points) → people_tracking ─────────────┤→ contact_detection → record_event → MySQL
RealSense (depth + cameraInfo) ── 各ノードの3次元投影に使用 ──────────┘
```

`bg_subtraction` / `subtraction_analysis` / `object_detection`（背景差分系）は現在未使用の系統。稼働しているのは YOLOX 由来の `yolox_object_detection`。

### 実装規約

Node / Logic 層の分離・新ノード登録・動的パラメータ（reconfigure）対応・メッセージ変更・DB マイグレーションの規約は `.claude/rules/` に分離してある（該当パス編集時に自動適用）。

## 注意事項

- launch ファイルは各ノードを gnome-terminal（Docker 版は xterm）の別ウィンドウで起動するため GUI 環境必須。ヘッドレス環境では `ros2 run` で個別起動する
- numpy 2 系は非対応（Dockerfile で `numpy<2` に固定）
- ノードのしきい値類は Logic 層にハードコードされているものが多い。パラメータを探すときは node 層だけでなく `nodes/<name>/logic.py` も確認する
