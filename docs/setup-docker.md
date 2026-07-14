# セットアップ（Docker）

Docker Compose で shigure_core の全ノード + MySQL + マイグレーションを一括起動する手順。
最短でシステムを動かしたい場合はこちらを推奨。

## 前提

- Docker / Docker Compose
- GUI 環境（各ノードのデバッグウィンドウを xterm で表示するため。X11）
- 外部ノード（RealSense / OpenPose / YOLO11）は別途起動する → [usage.md「外部ノードの起動」](usage.md#外部ノードの起動)

## 手順

### 1. サブモジュールの用意（初回のみ）

```sh
git submodule update --init
```

### 2. .env の作成

```sh
cp .env.example .env
```

.env を編集：

```sh
DISPLAY=:1         # echo $DISPLAY で確認した値
ROS_DOMAIN_ID=10   # 外部ノード側と同じ値に揃える（不一致だと一切通信できない）
```

### 3. コンテナからホストへの Display 出力を許可

```sh
xhost +local:docker
```

### 4. 起動

```sh
docker compose up --build   # 初回・コード変更後
docker compose up           # 2回目以降（コード変更なし）
```

> **注意** : ソースコードはイメージビルド時にコピーされる（volume マウントではない）ため、
> **コードや requirements.txt を変更したら必ず `--build` を付けて再ビルド**すること。

これで以下がすべて立ち上がる：

| サービス/ノード | 役割 |
| :--- | :--- |
| db (MySQL) + migrate | DB とマイグレーション自動適用（localhost:3306） |
| yolox_object_detection / object_tracking | 物体検出・追跡 |
| people_tracking / people_recognition | 人物追跡・顔認識 |
| contact_detection | 接触イベント判定 |
| pose_save / record_event | 骨格・イベントの DB 保存 |
| shigure_api | Web API（`SAVE_IMAGE=true` のときのみ起動、ポート 8765） |

顔辞書はホストの `~/.shigure` をマウントしているため、コンテナを作り直しても登録ユーザーは残る。

### 起動モードの変更（.env）

launch 引数は .env で変更できる（docker compose が起動時に launch へ渡す）：

| 変数 | デフォルト | 効果 |
| :--- | :--- | :--- |
| `DEBUG_MODE` | true | cv2 デバッグ窓の表示・自動登録ユーザーのディスク保存 |
| `SAVE_IMAGE` | false | 追跡デバッグ画像の配信 + Web API(shigure_api) の起動 |
| `ENABLE_PROFILE` | false | 横顔プロフィール特徴の配信・学習 |

変更後は `docker compose up` で再作成すれば反映される（イメージ再ビルドは不要）。

### 5. 停止・リセット

```sh
docker compose down      # 停止
docker compose down -v   # DB データも含めて完全リセット
```

## コンテナ内での操作

```sh
docker compose exec shigure_core bash
# 例: 顔登録ノードの対話実行
ros2 run shigure_core face_models
```

## 次のステップ

記録の開始・顔のユーザー登録・動作確認は [usage.md](usage.md) へ。
