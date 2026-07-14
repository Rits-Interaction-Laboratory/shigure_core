---
paths:
  - "shigure_core/shigure_core/nodes/**"
  - "shigure_core/setup.py"
---

# ノード実装の規約

## Node / Logic 層の分離

- **Node層** (`nodes/node_<name>.py`): ROS2 通信（購読/配信/パラメータ宣言）と Logic 呼び出しのみ。基本的に `ImagePreviewNode`（`nodes/node_image_preview.py`）を継承する
- **Logic層** (`nodes/<name>/logic.py`): アルゴリズム本体。ROS2 に依存させない。モデルクラスは同ディレクトリに置く
- cv2 描画は本来 Visualizer 層に分離したい（既存コードは Node 層に混在しているが、新規コードでは描画処理を別メソッド/クラスに分けること）

## 新ノード追加時のチェックリスト

1. `shigure_core/setup.py` の `packages` にサブパッケージを追加
2. `shigure_core/setup.py` の `console_scripts` にエントリポイントを追加
3. パラメータを使うなら `nodes/params/` にサンプル YML（`*.yml.sample`）を追加
4. 必要に応じて `shigure_core/launch/*_launch.py` に追加（複数の launch 変種があるので用途を確認して選ぶ）

`packages` と `console_scripts` の**両方**への登録を忘れると、ビルドは通るが `ros2 run` で見つからない・import エラーになる。
