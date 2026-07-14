---
paths:
  - "shigure_core_msgs/**"
  - "bbox_ex_msgs/**"
---

# メッセージ定義の規約

- `shigure_core_msgs` の .msg を追加・変更したら `CMakeLists.txt` の `rosidl_generate_interfaces` にも登録する。`--symlink-install` でも msg の再生成には再ビルドが必要
- `bbox_ex_msgs` は git submodule（Rits-Interaction-Laboratory/bbox_ex_msgs の `add-segment-msgs` ブランチ）。**このリポジトリ内で直接編集しない**。変更は fork 側リポジトリにコミットし、submodule のポインタを更新する
- msg のフィールド変更は購読側・配信側の全ノードに波及する。`ros2 interface show` やトピック名で grep して影響範囲を確認してから変更する
