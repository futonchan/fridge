## 概要
冷蔵庫に入れる際に野菜を画像から分類する。
カメラ→AI(サーバー）→DB→アプリ

## 動かすには
venvのPython3でサーバー立てる
```
cd fridge_camera/veget_google_top30_jpg_mini_vidadd_TFLite/example
source tflite-venv/bin/activate
python3 main.py
```

終了... Ctrl+C 2回

## フォルダ
- fridge_camera ... サーバー
- FridgeChecker ... アプリ

## 学習
MobileNetV2で。

# なんか作るときに参考にするポエム
https://qiita.com/sugulu_Ogawa_ISID/items/fa0ea622979507cdad6b
https://qiita.com/sugulu_Ogawa_ISID/items/c0e8a5e6b177bfe05e99

# 参考URL
https://github.com/aws-samples/smart-cooler
https://dev.classmethod.jp/articles/smart-cooler-012/
https://aws.amazon.com/jp/blogs/news/smart-cooler/

# フードロス系URL
http://www.fao.org/3/i2697o/i2697o.pdf
果実・野菜類がアジアで郡を抜いて多い→野菜、果実の期限をチェック

http://www.fao.org/home/en/

## ブランチモデル

[GitHUb Flow](https://postd.cc/gitlab-flow/) を採用します。
- main への直接pushは禁止
  - 開発用の各topicブランチを merge することで main は変化していきます
- 開発topicブランチは main から作成
  - issue に着手するときに main から作成
  - issue を閉じる寸前に main への pull request を作る
  - pull request が merge されたら、issue は閉じて、 topicブランチは削除
  - 
## issue - branch - commit - pull request 連携

1. 開発topicブランチを作る前に issue を作る
1. 開発topicブランチ名に issue番号を含める　例: `2_aws_gpu_costs`
1. commit メッセージには issue番号を含めます　例えば `#2`
1. commit メッセージの先頭prefixをつけてそのcommit の目的を示すのがオススメ
  - [【今日からできる】コミットメッセージに 「プレフィックス」 をつけるだけで、開発効率が上がった話](https://qiita.com/numanomanu/items/45dd285b286a1f7280ed)
    - feat: 新機能
    - bug: バグ修正
    - docs: ドキュメントのみの修正
    - style: プログラム動作に影響を与えないコード修正（スペース、コードフォーマットの調整）
    - refactor: バグ修正でも新機能でもないリファクタリング
    - perf: パフォーマンス向上
    - test: テストケースの追加や修正
    - chore: ビルドプロセスの変更や、依存ライブラリ、依存ツールの増減など

  - 例：[docs: #2 音声認識のコスト・スループット試算を追記](https://github.com/metamoji/translator/pull/3/commits/8095db0d5d8c534c529be1184b28bf30997bccac)
