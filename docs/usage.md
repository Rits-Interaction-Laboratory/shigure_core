# 使い方（Docker・生環境 共通）

セットアップ（[Docker](setup-docker.md) / [生環境](setup-native.md)）が済んでいる前提で、
外部ノードの起動 → 顔の事前登録 → 記録開始 → 動作確認 の流れをまとめる。

## 外部ノードの起動

センサ・検出系は別リポジトリ（多くは別PC）。すべて **同じ `ROS_DOMAIN_ID`** で起動すること。

RealSense（[rs_ros2_python](https://github.com/Rits-Interaction-Laboratory/rs_ros2_python)）
```sh
ros2 run rs_ros2_python rs_camera
```

OpenPose（[openpose_ros2_docker](https://github.com/Rits-Interaction-Laboratory/openpose_ros2_docker)）
```sh
docker run -it --gpus all --net host openpose_ros2_docker
bash /run.bash
```

People Detection（[people_detection_ros2_docker](https://github.com/Rits-Interaction-Laboratory/people_detection_ros2_docker)）
```sh
docker run -it --gpus all --net host people_detection_ros2_docker
bash /run.bash
```

> **注意** : 本体PC で認識パイプラインを動かす場合、別PC側では `object_tracking` / `yolox_object_detection` を動かさない（同名ノードの二重起動を避ける。センサ系は残す）。

## 顔の事前登録（人数分）

カメラが起動している状態で、face_models ノードを対話実行する。
ユーザー名を入力すると顔を最大100枚キャプチャし、特徴・顔画像を保存する。

```sh
# Docker の場合は先に: docker compose exec shigure_core bash
ros2 run shigure_core face_models
#   「ユーザー名を入力してください:」→ 実名（例 ryuhei）
#   カメラ前で正面を向く → ~/.shigure/face_models/user_ryuhei/ に .npy/.jpg 保存
```

確認： `ls ~/.shigure/face_models/`

> ⚠️ **people_recognition は起動時に一度だけ辞書を読み込む**。認識パイプラインを起動した後に登録しても反映されない（「登録したのに `face_name` が空のまま」の原因）。**登録はパイプライン起動前に**行う。後から登録した場合は people_recognition を再起動する（Docker なら `docker compose restart shigure_core`）。

## 記録の開始・終了（← この信号を送るまで DB に保存されない）

record_event は `current_pose_id`（pose_save 由来）を含む同期購読で発火する。pose_save は **`/HL2/pose_record_signal` = `Start`** を受けて初めて `current_pose_id` を publish する設計。**ノードを起動しただけでは「記録開始」にならず**、信号を送るまでは接触が起きても（`/shigure/contacted` は流れる）DB に保存されない（実際に発生した）。

HoloLens を使わない場合は手動で信号を送る：

```sh
# 記録開始（pose_save のタブに「記録開始」と出る／current_pose_id が流れ始める）
ros2 topic pub --once /HL2/pose_record_signal std_msgs/msg/String "{data: Start}"

# 確認：current_pose_id が流れているか（空なら記録が始まっていない）
ros2 topic echo /shigure/current_pose_id

# 記録終了 / 待機に戻す
ros2 topic pub --once /HL2/pose_record_signal std_msgs/msg/String "{data: End}"
```

## 動作確認

### トピック

```sh
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

```sh
docker compose exec db mysql -ushigure -pshigure shigure \
  -e "SELECT id, person_id, name FROM people ORDER BY id DESC LIMIT 10;"
docker compose exec db mysql -ushigure -pshigure shigure \
  -e "SELECT * FROM event ORDER BY id DESC LIMIT 10;"
```

ホストから直接接続する場合（PW: `shigure`）：
```sh
mysql -h 127.0.0.1 -P 3306 -u shigure -p
```

### Web（save_image:=true のとき）

`http://<本体PCのIP>:8765` … `/users`(顔画像) / `/ws/tracking_debug`(全体画像) / `/ws/pca_plot`(PCA) / `/ws/events`(認識イベント)

## 調整パラメータ

```sh
# face_name を確定する累積スコア閾値（既定3.0、コサイン類似度の累積和。小さいほど早く名前が出る）
ros2 param set /people_tracking_node face_name_score_threshold 2.0
```

- 自動登録（未登録者を `user_newN` として辞書追加）の発火特徴数：`node_people_recognition.py` の `MIN_FEATURES_FOR_NEW_USER`（既定20）
- 顔検出頻度を上げたい：同ファイルの `MIN_DET_SCORE`（既定0.8）を下げる
- 自動登録ユーザーの**ディスク保存は `debug_mode:=true`（is_debug_mode=True）時のみ**。false だとメモリ辞書のみで再起動で消える

## トラブルシュート

| 症状 | 対処 |
|---|---|
| `ros2 node list` が実態と食い違う | `ros2 node list --no-daemon` で見る（デーモンが古いノードをキャッシュする）。端末で `export ROS_DOMAIN_ID=10` を徹底 |
| 同名ノードが2つ（object_tracking / yolox_object_detection） | 別PC側で該当ノードを停止（センサ系は残す） |
| `face_name` が空のまま | ①`/face_recognition/results` の id が実名か（unknown なら辞書未ロード/未登録）②閾値を下げる（§調整パラメータ）③登録は people_recognition 起動**前**に。後から登録したら people_recognition を再起動 |
| people_recognition が落ちる（別人が入った瞬間 等） | 修正済み（LabelPropagation の n_neighbors をサンプル数でクリップ）。再ビルドを確認 |
| `ImportError: cannot import name 'FaceRecognitionResult' ...` で people_tracking / people_recognition が即終了 | `shigure_core_msgs` のビルドが古い（`__init__.py` が新 msg を export していない）。**クリーン再ビルド**する: `rm -rf build/shigure_core_msgs install/shigure_core_msgs && colcon build --packages-select shigure_core_msgs --symlink-install`。**msg を追加/変更した後・msg を含むブランチを merge/pull した後は必ずクリーン再ビルド** |
| GPU が使われず CPU にフォールバック | `onnxruntime` と `onnxruntime-gpu` の重複を解消（`pip uninstall onnxruntime onnxruntime-gpu; pip install onnxruntime-gpu`）＋ `LD_LIBRARY_PATH`（[setup-native.md](setup-native.md#2-環境変数作業する端末で毎回)） |
| record_event がイベントを保存しない | ①**まず全端末の `ROS_DOMAIN_ID` 一致を確認**（不一致だとトピック未受信で無反応）②`/HL2/pose_record_signal`=`Start` を送ったか（§記録の開始・終了、`/shigure/current_pose_id` が流れているか）③`/shigure/contacted` 自体が流れているか |

## Rviz での確認（オプション）

```sh
# Docker の場合（Dockerfile にインストール済み）
docker compose exec shigure_core bash
ros2 run rviz2 rviz2
# 生環境の場合: sudo apt install ros-humble-rviz2
```

- Fixed Frame を `marker` に指定し、Add から TF を追加すると座標系が表示される

## 個人データの取り扱い（重要）

- **顔画像・特徴（face_models 配下）は個人データ。リポジトリに絶対にコミットしない**。`.gitignore` で `**/face_models/` を除外済み。実データは `~/.shigure/face_models`（リポジトリ外）に置く
- 誤って公開リポジトリに push した場合、ブランチ削除・PR クローズだけでは消えない。GitHub Support（Private Information Removal）へ削除依頼が必要
