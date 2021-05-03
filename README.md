## 概要
画像分類転移学習EfficientNet
pip でEfficientNet入れてる、めちゃ便利

## データ
- ありとはちweights(EfficientNetの重みはpipのパッケージで保存されてる)
https://drive.google.com/file/d/1YBYy7-dJB26QVxNAZNQ61MQqMetkmhIG/view?usp=sharing

- ありとはちデータセット(data/hymenoptera_data/train/...jpgとなるように配置)
https://drive.google.com/file/d/13gFZjbF96OZSmU6W_JTMJ538oqKa7TxD/view?usp=sharing
## 学習
EfficientNetで
0～7まで

## テスト
`onlytest.py`実行結果

```
using device: cuda:0
GeForce RTX 2070
Memory Usage:
Allocated: 0.0 GB
Cached:    0.0 GB
Loaded pretrained weights for efficientnet-b0
['ants', 'bees']
-----
ants                                                                        (62.08%)
bees                                                                        (37.92%)
```