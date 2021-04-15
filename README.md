# azureml-pl-sample

Code samples for building ML model with PyTorch Lightning and training the model on Azure Machine Learning computing environment, covering training model to deploy model.

## Model

- BERT - text classification for livedoor news corpus 
- ResNet - image classification for CIFAR-10

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
