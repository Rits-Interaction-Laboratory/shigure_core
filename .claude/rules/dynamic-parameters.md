---
paths:
  - "shigure_core/shigure_core/nodes/**"
---

# 動的パラメータ（Dynamic Reconfigure）の規約

ノードに新しいパラメータを追加・変更するときは、`ros2 param set` で再起動なしに変更できるようにすること（reconfigure 対応必須）。

## 実装方法（方法A：コールバック追加登録）

- `declare_parameter` には `ParameterDescriptor`（型・説明）を付ける
- 基底クラス `ImagePreviewNode` の `_on_set_parameters` を**オーバーライドしない**。サブクラスごとに専用コールバック（例: `_on_set_parameters_contact`）を定義し、`add_on_set_parameters_callback` で**追加登録**する
- コールバックは登録順に全て呼ばれるため、各クラスは自分が宣言したパラメータのみ処理し、他は素通しして `SetParametersResult(successful=True)` を返す
- Logic 層にハードコードされたしきい値を見つけたら、可能ならノード層のパラメータに昇格して Logic へ引数で渡す

## cv2 ウィンドウの消去（重要な落とし穴）

`is_debug_mode=false` にした際の `cv2.destroyAllWindows()` は、パラメータコールバックではなく**トピックコールバック（`callback_debug` など描画側）の先頭**で呼ぶこと。

理由: `cv2.destroyAllWindows()` は `cv2.waitKey()` と同じスレッドから呼ぶ必要があり、パラメータコールバックスレッドで呼んでも直後の `imshow()` でウィンドウが復活する。

```python
def callback_debug(self, ...):
    self.callback(...)           # コア処理（publish）
    if not self.is_debug_mode:
        cv2.destroyAllWindows()  # ← ここで閉じる
        return
    # デバッグ描画...
```

`callback` / `callback_debug` を分離していないノードでは、インラインの `else` 節で同様に対処する。
