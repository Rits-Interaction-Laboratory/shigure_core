---
paths:
  - "shigure_core/shigure_core/db/**"
  - "shigure_core/resource/db/**"
---

# DB まわりの規約

- スキーマ変更は golang-migrate 形式のマイグレーションで行う（`shigure_core/resource/db/migration/`）。**適用済みの既存マイグレーションは編集せず**、連番を進めて `N_<name>.up.sql` / `N_<name>.down.sql` のペアを新規追加する
- DB アクセスは `shigure_core/db/event_repository.py`（`EventRepository`）に集約する。ノードから直接 `mysql.connector` を使わない
- 既存の `EventRepository` は f-string で SQL を組み立てているが、新規クエリではプレースホルダ（`cur.execute(sql, params)`）を使うこと
- 接続情報は `db/config.py`。ローカル接続: `mysql -h 127.0.0.1 -P 3306 -u shigure -p`（PW: `shigure`）
