# 顕著な特徴抽出による現実世界での異常検出
## 実行環境
* Python : 3.7.11
* Numpy : 1.19.5
* Pytorch : 1.9.1  
* scikit-learn : 0.24.2
* tqdm :　4.46.0

Docker環境がある場合は，[yaegasikk/olab-attention](https://hub.docker.com/repository/docker/yaegasikk/olab-attention)からイメージを入手し，利用できます．
## データの準備
### 1. UCF-CrimeデータセットのI3D特徴量の抽出
[RTFMの実装](https://github.com/tianyu0207/RTFM)と同様の特徴量を使用しています．他のデータセットに対して実験を行う場合は[I3D_Feature_Extraction_resnet](https://github.com/GowthamGottimukkala/I3D_Feature_Extraction_resnet)を用いて特徴量抽出をおこなってください．

* **UCF-Crime train i3d Google drive**から学習データをダウンロードし，フォルダ名を`UCF-Train`に変更後，`features` に入れてください．
  * 学習済みモデルを用いて論文の結果を再現するだけであれば、この学習データのダウンロードは省略できます．
* **UCF-Crime test i3d Google drive**からテストデータをダウンロードし，フォルダ名を`UCF-Test`に変更後，`features` に入れてください．

<pre>
.
|-- dataset.py
|-- features
|    |-- UCF-Test
|    |  |-- Abuse028_x264_i3d.npy
|    |  |-- Abuse030_x264_i3d.npy
|    |  |-- Arrest001_x264_i3d.npy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
|    |-- UCF-Train
|    |  |-- Abuse001_x264_i3d.npy
|    |  |-- Abuse002_x264_i3d.npy
|    |  |-- Abuse003_x264_i3d.npy
|    |  |-- Abuse004_x264_i3d.npy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
|-- train.py
|-- list
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
</pre>

### 2. データセットの前処理とリストの作成
```
python preformer_feature.py
```
## 論文の結果の再現（テスト）
UCF-CrimeデータセットにおけるフレームレベルのAUCの性能を示す表2の結果を再現．

<img src="https://latex.codecogs.com/svg.image?d_a=64,r=3&space;" title="d_a=64,r=3 " /> のとき
```
python test.py --da 64 --r 3 --seed 9111 --test-split-size 28
```
<img src="https://latex.codecogs.com/svg.image?d_a=128,r=7&space;" title="d_a=128,r=7 " /> のとき
```
python test.py --da 128 --r 7 --seed 9111 --test-split-size 28
```
| model | AUC(%)|
|----|----|
|Sultani et al.|75.41|
|GCN-Anomaly|82.12|
|RTFM|84.30|
|Ours( <img src="https://latex.codecogs.com/svg.image?d_a=64,r=3&space;" title="d_a=64,r=3 " /> )|84.74|
|Ours( <img src="https://latex.codecogs.com/svg.image?d_a=128,r=7&space;" title="d_a=128,r=7 " /> )|84.91|

## 学習
```
python train.py --da 64 --r 3 --seed 1111
```
