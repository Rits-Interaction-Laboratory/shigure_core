# USAGE — 顔認識つき shigure_core 使い方ガイド

顔の事前登録 → 人物認識（顔名の付与）→ 接触（持ち込み／持ち去り）イベントの DB 記録までの手順をまとめる。
（このドキュメントは `feature/face_recognition` ブランチの統合構成が対象。既存の `README.md` は別途参照）

---

## 1. 概要 / データフロー

```
[別PC] RealSense ─ /rs/color/compressed, /rs/aligned_depth_to_color/*
[別PC] OpenPose  ─ /openpose/pose_key_points
[別PC] YOLO11    ─ /bounding_boxes, /Segments
        │
        ▼ (このPC)
 yolox_object_detection → object_tracking ─┐
 people_recognition ─ /face_recognition/results, /feature_info, /dictionary_update
        │                                   │
        ▼                                   ▼
 people_tracking ── /shigure/people_detection（people_id + face_name）
        │                                   │
        ▼                                   ▼
 contact_detection ── /shigure/contacted（people_id + face_name + action）
        │
        ▼
 record_event ── MySQL（people.name に顔名、event に持込/持去アクションを保存）
        │
 shigure_api ── Web表示（顔画像 / 追跡オーバーレイ / PCA / 認識イベント）
```

- **顔名（face_name）の付与**：people_recognition が顔を照合 → people_tracking が「頭部が顔boxに入る」対応付けでコサイン類似度を累積 → 閾値超えで `PoseKeyPoints.face_name` に格納。
- **DB へ「誰が」**：contact_detection が `Contacted.face_name = person.face_name` を設定 → record_event が `people.name` に保存（migration `8_add_name_to_people`）。

---

## 2. 前提

### 別PC（センサ側）
`realsense` / `v4l2_camera` / `openpose_node` / `yolox_ros`(YOLO11) を `ROS_DOMAIN_ID=10` で稼働。
**このPCを本体にするので、別PCでは `object_tracking` / `yolox_object_detection` は動かさない**（同名ノードの二重起動を避ける）。

### このPC（本体）— 依存パッケージ
```bash
# 顔認識
pip install insightface onnxruntime-gpu          # GPUなしは onnxruntime（-gpuと同時導入しない）
# Web API（save_image 使用時）
pip install fastapi 'uvicorn[standard]'
# record_event をネイティブ実行する場合の DB ドライバ
# ※ 必ず mysql-connector-python（import mysql.connector を提供）。
#    `pip install mysql` や `mysqlclient` は別物で動かないので注意。
pip install mysql-connector-python
```

### 環境変数（作業する端末で毎回）
```bash
export ROS_DOMAIN_ID=10
# GPU を使う場合（torch同梱のCUDA12ライブラリをローダに通す）
export LD_LIBRARY_PATH=$(python3 -c "import os,glob;print(':'.join(sorted(glob.glob(os.path.expanduser('~/.local/lib/python3.10/site-packages/nvidia/*/lib')))))"):$LD_LIBRARY_PATH
```
- InsightFace のデバイスは自動選択。強制するなら `export SHIGURE_INSIGHTFACE_DEVICE=cpu`（または `gpu`）。
- 顔辞書の場所は既定で `~/.shigure/face_models`（リポジトリ外・永続）。変えるなら `export SHIGURE_FACE_MODELS_DIR=/path/to/dir`。

---

## 3. ビルド

> ⚠️ **merge / pull した直後、または .msg を追加・変更した後は、`shigure_core_msgs` を必ず「クリーン」再ビルドする**（`build/`・`install/` を消してからビルド）。増分ビルドだと `msg/__init__.py` が古いまま残り、`ImportError: cannot import name 'FaceRecognitionResult' ...` で people_tracking / people_recognition が**起動直後に即終了する**（実際に発生した）。

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash

# merge/pull 直後・msg変更後は shigure_core_msgs をクリーン再ビルド（この2行を先に）
rm -rf build/shigure_core_msgs install/shigure_core_msgs
colcon build --packages-select shigure_core_msgs --symlink-install

colcon build --packages-select shigure_core shigure_api   # ament_python は --symlink-install 無し
source ~/ros2_ws/install/setup.bash
```

---

## 4. 使い方（推奨構成：DBはdocker、認識はネイティブ）

GPU/InsightFace はネイティブ、MySQL は docker で公開（`127.0.0.1:3306`）し、record_event はネイティブから接続する。

### 4-1. DB を起動（MySQL + マイグレーション）
```bash
cd ~/ros2_ws/shigure_core
cp .env.example .env          # DISPLAY と ROS_DOMAIN_ID を設定
docker compose up -d db migrate
#   → localhost:3306 に MySQL（db=shigure, user=shigure, pass=shigure）、テーブル作成まで完了
```

### 4-2. 顔の事前登録（人数分）— 認識パイプライン(4-3)を起動する**前**に行う
```bash
export ROS_DOMAIN_ID=10
ros2 run shigure_core face_models
#   「ユーザー名を入力してください:」→ 実名（例 ryuhei）
#   カメラ前で正面を向く → ~/.shigure/face_models/user_ryuhei/ に .npy/.jpg 保存（最大100枚）
```
確認： `ls ~/.shigure/face_models/`

> ⚠️ **people_recognition は起動時に一度だけ辞書を読み込む**。4-3 を起動した後に登録しても反映されない（「登録したのに `face_name` が空のまま」の原因）。**登録は 4-3 の前に**行う。後から登録した場合は **people_recognition を再起動**すること。

### 4-3. 認識パイプラインを起動
```bash
export ROS_DOMAIN_ID=10          # ← 全端末で必須。設定漏れだと他ノードと一切通信できない
ros2 launch shigure_core shigure_core_launch.py debug_mode:=true save_image:=true
```
起動するノード：yolox_object_detection / object_tracking / people_tracking / people_recognition / contact_detection /（save_image時のみ）shigure_api。
起動時に people_recognition が `~/.shigure/face_models` を辞書ロード → 事前登録者は即認識。

launch 引数：
| 引数 | 既定 | 効果 |
|---|---|---|
| `debug_mode` | false | 全ノードの cv2 デバッグ窓表示＋**people_recognition の自動登録ユーザーを .npy/.jpg でディスク保存** |
| `save_image` | false | 追跡デバッグ画像を `/shigure/tracking_debug_image` へ配信・保存＋Web表示(shigure_api)を起動 |
| `enable_profile` | false | 横顔プロフィール特徴を `/profile_feature_add` に配信（横顔学習） |

### 4-4. イベント記録ノードを起動（別端末・DBへ保存）

> ⚠️ **各端末で必ず `export ROS_DOMAIN_ID=10`**。設定漏れ（domain 0 のまま等）だと record_event が `/shigure/contacted` 等を**一切受信できず、DBに何も保存されない**（実際に発生した最頻の原因）。`ros2 node list --no-daemon` で他ノードが見えるかで確認できる。

```bash
# 端末A
export ROS_DOMAIN_ID=10
ros2 run shigure_core pose_save

# 端末B
export ROS_DOMAIN_ID=10
ros2 run shigure_core record_event --ros-args \
  -p save_root_path:=$HOME/shigure_events \
  -p camera_id:=1 -p frame_num:=120 -p is_recording_depth_info:=false
```
- record_event は `/shigure/contacted` を受けて、`people.name`（＝face_name）や持込/持去アクションを MySQL に保存する。

### 4-5. 記録を開始する（← この信号を送るまで保存されない）

record_event は `current_pose_id`（pose_save 由来）を含む同期購読で発火する。pose_save は **`/HL2/pose_record_signal` = `Start`** を受けて初めて `current_pose_id` を publish する設計。**ノードを起動しただけでは"記録開始"にならず**、信号を送るまでは接触が起きても（`/shigure/contacted` は流れる）DBに保存されない（実際に発生した）。

HoloLens を使わない場合は手動で信号を送る：
```bash
export ROS_DOMAIN_ID=10
# 記録開始（pose_save のタブに「記録開始」と出る／current_pose_id が流れ始める）
ros2 topic pub --once /HL2/pose_record_signal std_msgs/msg/String "{data: Start}"

# 確認：current_pose_id が流れているか（空なら記録が始まっていない）
ros2 topic echo /shigure/current_pose_id

# 記録終了 / 待機に戻す
ros2 topic pub --once /HL2/pose_record_signal std_msgs/msg/String "{data: End}"
```

---

## 5. 動作確認

### トピック
```bash
export ROS_DOMAIN_ID=10
ros2 node list --no-daemon | grep -E "people_tracking|people_recognition"   # 各1つ

# 顔ID↔骨格の結合（people_id と face_name が同一エントリに出れば成功）
ros2 topic echo /shigure/people_detection

# 接触イベント（face_name / action が乗る）
ros2 topic echo /shigure/contacted

# 認識の中間結果
ros2 topic echo /shigure/recognition_history
ros2 topic echo /face_recognition/results
```

### DB
```bash
docker compose exec db mysql -ushigure -pshigure shigure \
  -e "SELECT id, person_id, name FROM people ORDER BY id DESC LIMIT 10;"
docker compose exec db mysql -ushigure -pshigure shigure \
  -e "SELECT * FROM event ORDER BY id DESC LIMIT 10;"
```

### Web（save_image:=true のとき）
`http://<このPCのIP>:8765` … `/users`(顔画像) / `/ws/tracking_debug`(全体画像) / `/ws/pca_plot`(PCA) / `/ws/events`(認識イベント)

---

## 6. 調整パラメータ

```bash
# face_name を確定する累積スコア閾値（既定3.0、コサイン類似度の累積和。小さいほど早く名前が出る）
ros2 param set /people_tracking_node face_name_score_threshold 2.0
```
- 自動登録（未登録者を `user_newN` として辞書追加）の発火特徴数：`node_people_recognition.py` の `MIN_FEATURES_FOR_NEW_USER`（既定20）。
- 顔検出頻度を上げたい：同ファイルの `MIN_DET_SCORE`（既定0.8）を下げる。
- 自動登録ユーザーの**ディスク保存は `debug_mode:=true`（is_debug_mode=True）時のみ**。false だとメモリ辞書のみで再起動で消える。

---

## 7. トラブルシュート

| 症状 | 対処 |
|---|---|
| `ros2 node list` が実態と食い違う | `ros2 node list --no-daemon` で見る（デーモンが古いノードをキャッシュする）。端末で `export ROS_DOMAIN_ID=10` を徹底 |
| 同名ノードが2つ（object_tracking / yolox_object_detection） | 別PC側で該当ノードを停止（センサ系は残す） |
| `face_name` が空のまま | ①`/face_recognition/results` の id が実名か（unknownなら辞書未ロード/未登録）②閾値を下げる（§6）③登録は people_recognition 起動**前**に。後から登録したら people_recognition を再起動 |
| people_recognition が落ちる（別人が入った瞬間 等） | 修正済み（LabelPropagation の n_neighbors をサンプル数でクリップ）。再ビルドを確認 |
| `ImportError: cannot import name 'FaceRecognitionResult' ...` で people_tracking / people_recognition が即終了 | `shigure_core_msgs` のビルドが古い（`__init__.py` が新msgを export していない）。**クリーン再ビルド**する: `rm -rf build/shigure_core_msgs install/shigure_core_msgs && colcon build --packages-select shigure_core_msgs --symlink-install`。**msg を追加/変更した後・msgを含むブランチを merge/pull した後は必ずクリーン再ビルド** |
| GPU が使われず CPU にフォールバック | `onnxruntime` と `onnxruntime-gpu` の重複を解消（`pip uninstall onnxruntime onnxruntime-gpu; pip install onnxruntime-gpu`）＋ `LD_LIBRARY_PATH`（§2） |
| record_event がイベントを保存しない | ①**まず全端末の `ROS_DOMAIN_ID=10` を確認**（不一致だとトピック未受信で無反応）②`/HL2/pose_record_signal`=`Start` を送ったか（§4-5、`/shigure/current_pose_id` が流れているか）③`/shigure/contacted` 自体が流れているか |

---

## 8. 個人データの取り扱い（重要）

- **顔画像・特徴（face_models 配下）は個人データ。リポジトリに絶対にコミットしない**。`.gitignore` で `**/face_models/` を除外済み。実データは `~/.shigure/face_models`（リポジトリ外）に置く。
- 誤って公開リポジトリに push した場合、ブランチ削除・PRクローズだけでは消えない。GitHub Support（Private Information Removal）へ削除依頼が必要。
