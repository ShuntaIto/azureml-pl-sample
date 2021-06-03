# azureml-pl-sample

PyTorch Lightning を使用して機械学習モデルを構築し、 Azure Machine Learning の計算環境を使用して学習を行うコードサンプルです。学習からモデルデプロイまでカバーしています。

## モデル

- BERT - ライブドアニュースコーパスによるテキスト分類
- ResNet - CIFAR-10 による画像分類

### 今後のサポート予定

- GPT-2 - テキスト生成

## 使い方

0. 以下のリソースを Azure 上に用意する
   - Azure Machine Learning Workspace
   - Azure Machine Learning 上のコンピューティングインスタンス (GPU搭載)
1. conda 環境を整え、Jupyter カーネルをインストールする
```
conda env create -f=env.yml
conda activate py38-pt180
ipython kernel install --user --name=py38-pt180
```
2. \<model_name\>/train_\<model_name\>_aml.ipynb の各セルを実行する
3. deploy_\<model_name\>_aml.ipynb の各セルを実行する

コンピューティングクラスターによる分散学習を試したい場合は、GPU 搭載のインスタンスを用いたコンピューティングクラスターを作成した上で 'dist_train_\<model_name\>_aml.ipynb' を使用する.

ローカル環境で学習を実行したい場合は'model_test/\<model_name\>_model_test.ipynb' を使用する。

## 分散学習

2021/04/02 時点で, 最新の PyTorch と PyTorch Lightning の組み合わせでは分散学習を実行することができず、あえて古いバージョンを使用しています。

以下の議論を参考に適切と考えられるバージョンを指定しています。

- https://azure.github.io/azureml-examples/docs/cheatsheet/
- https://github.com/Azure/azureml-examples/tree/main/experimental/using-pytorch-lightning
- https://github.com/PyTorchLightning/pytorch-lightning/issues/4612


# azureml-pl-sample

Code samples for building ML model with PyTorch Lightning and training the model on Azure Machine Learning computing environment, covering training model to deploy model.

## Model

- BERT text classification for livedoor news corpus
- ResNet - CIFAR-10 による画像分類 image classification for CIFAR-10

### scheduled to be supported

- GPT-2 - text generation

## How to Use

0. create below resources on Azure 
   - Azure Machine Learning Workspace
   - Computing Instance with GPU on AML
1. setup conda environment and add as jupyter kernel
```
conda env create -f=env.yml
conda activate py38-pt180
ipython kernel install --user --name=py38-pt180
```
2. execute each cells in \<model_name\>/train_\<model_name\>_aml.ipynb
3. execute each cells in deploy_\<model_name\>_aml.ipynb

If you want to try distributed-training model on cluster, create Computing Cluster with GPU and use 'dist_train_\<model_name\>_aml.ipynb' notebooks.

If you want to train the model on Jupyter Notebook/JupyterLab or other jupyter-like environment, use 'model_test/\<model_name\>_model_test.ipynb'.

## distributed-training

On my trial at 2021/04/02, distributed-training with the newest PyTorch and PyTorch Lightning was failed, so I've chosen older version.

I refered below resource to choose the appropriate version for distributed-training.

- https://azure.github.io/azureml-examples/docs/cheatsheet/
- https://github.com/Azure/azureml-examples/tree/main/experimental/using-pytorch-lightning
- https://github.com/PyTorchLightning/pytorch-lightning/issues/4612

## Reference

- https://pytorch.org/docs/stable/index.html
- https://pytorch-lightning.readthedocs.io/en/latest/
- https://docs.microsoft.com/ja-jp/python/api/overview/azure/ml/?view=azure-ml-py
- https://azure.github.io/azureml-examples/docs/cheatsheet/
- https://github.com/Azure/azureml-examples/tree/main/experimental/using-pytorch-lightning
- https://github.com/PyTorchLightning/pytorch-lightning/issues/4612
- https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/ml-frameworks/keras/train-hyperparameter-tune-deploy-with-keras/train-hyperparameter-tune-deploy-with-keras.ipynb