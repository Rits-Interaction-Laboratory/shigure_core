# ソースコード構成
ここではShigure Coreのソースコード構成について説明します。

## ディレクトリ構成
```
shigure_core
├── resource
│   └── db
│       ├── docker-compose.yml
│       └── migration
├── shigure_core
│   ├── db
│   ├── enum
│   ├── nodes
│   │   ├── xxx
│   │   ├── common_model
│   │   ├── yyy
│   │   ├── node_xxx.py
│   │   ├── node_yyy.py
│   │   └── params
│   │       ├── xxx_params.yml.sample
│   │       └── yyy_params.yml.sample
│   └── util
└── test
```

### shigure_core/resource/db
* イベントのメタデータ保存に必要なDBの設定が書かれています。
* とくに `shigure_core/resource/db/migration` はDBのテーブル定義が書かれています。

### shigure_core/shigure_core/db
* DBにアクセスするためのコード群です。

### shigure_core/shigure_core/enum
* 定数が定義されたコード群です。

### shigure_core/shigure_core/nodes/node_xxx.py
* ノードの定義がされたコード群です。
* 初期設定・ROSの通信に関わることを記述します。
* 処理ロジックは `shigure_core/shigure_core/nodes/xxx` 以下のコードに移譲します。
    * `shigure_core/shigure_core/nodes/xxx/logic.py` を呼び出します

### shigure_core/shigure_core/nodes/xxx
* そのノードについての処理ロジックが実装されたコード群です。
* ドメイン駆動設計（≒オブジェクト指向）に従ったモデリングをしています。
    * `logic.py` は処理ロジックフローを表しているため対象外です。

### shigure_core/shigure_core/nodes/common_model
* ノード間で共通して利用するモデルを実装したコード群です。

### shigure_core/shigure_core/nodes/params/xxx_params.yml.sample
* 実行時の設定値を定義したファイルです。

### shigure_core/shigure_core/util
* ユーティリティ関数が定義されたコード群です。

## 実装におけるルール
コードの品質を落とさない（可読性を維持する）ために以下のルールで実装されています。ルールを守って実装することでデバッグ容易性を高めることができます。

### ROSの設定を記述するとき
`shigure_core/shigure_core/nodes/node_xxx.py` に実装します

### 処理ロジックフローを実装するとき
`shigure_core/shigure_core/nodes/xxx/logic.py` に実装します。
したがって `shigure_core/shigure_core/nodes/xxx/` 以下に `logic.py` が必ず存在します。

### データ構造を定義するとき
`shigure_core/shigure_core/nodes/xxx/` 以下に実装します。
**データに変更を加える操作を実装するとき** はデータ構造に振る舞いをもたせる形で実装します。
**決して `node_xxx.py` や `logic.py` が状態を持つ実装をしてはいけません。**
`node_xxx.py` がデータ構造のオブジェクトを持つことは問題ありません。
