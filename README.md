## 概要
冷蔵庫に入れる際に野菜を画像から分類する。

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

# 参考URL
https://github.com/aws-samples/smart-cooler
https://dev.classmethod.jp/articles/smart-cooler-012/
https://aws.amazon.com/jp/blogs/news/smart-cooler/

# フードロス系URL
http://www.fao.org/3/i2697o/i2697o.pdf
果実・野菜類がアジアで郡を抜いて多い→野菜、果実の期限をチェック

http://www.fao.org/home/en/
