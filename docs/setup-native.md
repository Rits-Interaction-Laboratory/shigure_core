# セットアップ（生 ROS2 環境）

ROS2 Humble がインストール済みのマシンで直接ノードを動かす手順。
GPU で顔認識を回したい場合や開発時はこちら。DB のみ Docker で立てる構成を推奨する。

## 前提

- ROS2 Humble
- 本リポジトリを ROS2 ワークスペースの `src/` 配下に clone していること
- 外部ノード（RealSense / OpenPose / YOLO11）は別途起動する → [usage.md「外部ノードの起動」](usage.md#外部ノードの起動)

## 1. 依存パッケージのインストール

```sh
git submodule update --init      # 初回のみ（bbox_ex_msgs）
pip3 install -r requirements.txt
```

GPU で InsightFace を使う場合は onnxruntime を GPU 版に差し替える：

```sh
# onnxruntime と onnxruntime-gpu を同時に入れない（CPU にフォールバックする原因）
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu
```

> **注意** : DB ドライバは必ず `mysql-connector-python`（`import mysql.connector` を提供）。
> `pip install mysql` や `mysqlclient` は別物で動かない。

## 2. 環境変数（作業する端末で毎回）

```sh
export ROS_DOMAIN_ID=10   # ← 全端末で必須。設定漏れだと他ノードと一切通信できない
# GPU を使う場合（torch 同梱の CUDA12 ライブラリをローダに通す）
export LD_LIBRARY_PATH=$(python3 -c "import os,glob;print(':'.join(sorted(glob.glob(os.path.expanduser('~/.local/lib/python3.10/site-packages/nvidia/*/lib')))))"):$LD_LIBRARY_PATH
```

- InsightFace のデバイスは自動選択。強制するなら `export SHIGURE_INSIGHTFACE_DEVICE=cpu`（または `gpu`）
- 顔辞書の場所は既定で `~/.shigure/face_models`（リポジトリ外・永続）。変えるなら `export SHIGURE_FACE_MODELS_DIR=/path/to/dir`

## 3. ビルド

> ⚠️ **merge / pull した直後、または .msg を追加・変更した後は、`shigure_core_msgs` を必ず「クリーン」再ビルドする**（`build/`・`install/` を消してからビルド）。増分ビルドだと `msg/__init__.py` が古いまま残り、`ImportError: cannot import name 'FaceRecognitionResult' ...` で people_tracking / people_recognition が**起動直後に即終了する**（実際に発生した）。

```sh
cd <ROS2 workspace>
source /opt/ros/humble/setup.bash

# merge/pull 直後・msg 変更後は shigure_core_msgs をクリーン再ビルド（この2行を先に）
rm -rf build/shigure_core_msgs install/shigure_core_msgs
colcon build --packages-select shigure_core_msgs --symlink-install

colcon build --packages-select shigure_core shigure_api
source install/setup.bash
```

## 4. DB を起動（MySQL + マイグレーション、Docker 利用）

```sh
cp .env.example .env          # 初回のみ
docker compose up -d db migrate
#   → localhost:3306 に MySQL（db=shigure, user=shigure, pass=shigure）、テーブル作成まで完了
```

## 5. 認識パイプラインの起動

```sh
ros2 launch shigure_core shigure_core_launch.py debug_mode:=true save_image:=true
```

起動するノード：yolox_object_detection / object_tracking / people_tracking / people_recognition / contact_detection /（save_image 時のみ）shigure_api

| launch 引数 | 既定 | 効果 |
| :--- | :--- | :--- |
| `debug_mode` | false | 全ノードの cv2 デバッグ窓表示＋people_recognition の自動登録ユーザーを .npy/.jpg でディスク保存 |
| `save_image` | false | 追跡デバッグ画像を `/shigure/tracking_debug_image` へ配信・保存＋Web 表示(shigure_api)を起動 |
| `enable_profile` | false | 横顔プロフィール特徴を `/profile_feature_add` に配信（横顔学習） |
| `terminal` | gnome-terminal | 各ノードを開く端末（`gnome-terminal` / `xterm` / `none`）。ヘッドレス環境は `none` |
| `record` | false | DB 保存系ノード（pose_save / record_event）も起動する（手順6の代わり） |
| `save_root_path` | (ノード既定) | record_event のイベント画像保存先 |

## 6. 記録ノードの起動（別端末・DB へ保存）

> ⚠️ **各端末で必ず `export ROS_DOMAIN_ID=10`**。設定漏れだと record_event が `/shigure/contacted` 等を**一切受信できず、DB に何も保存されない**（実際に発生した最頻の原因）。`ros2 node list --no-daemon` で他ノードが見えるかで確認できる。

```sh
# 端末A
export ROS_DOMAIN_ID=10
ros2 run shigure_core pose_save

# 端末B
export ROS_DOMAIN_ID=10
ros2 run shigure_core record_event --ros-args \
  -p save_root_path:=$HOME/shigure_events \
  -p camera_id:=1 -p frame_num:=120 -p is_recording_depth_info:=false
```

パラメータを YML で渡す場合のサンプル : [/shigure_core/shigure_core/nodes/params/](../shigure_core/shigure_core/nodes/params/)

## 次のステップ

**ノードを起動しただけでは記録は始まらない。** 記録開始信号・顔のユーザー登録・動作確認は [usage.md](usage.md) へ。
