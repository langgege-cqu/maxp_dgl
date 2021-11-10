# MAXP竞赛——DGL图数据Baseline模型

本代码库是为2021 MAXP竞赛的DGL图数据所准备的Baseline模型，供参赛选手参考学习使用DGL来构建GNN模型。

代码库包括2个部分：
---------------
1. 用于数据预处理的4个Jupyter Notebook
2. 用DGL构建的3个GNN模型(GCN,GraphSage和GAT)，以及训练模型所用的代码和辅助函数。

依赖包：
------
- dgl==0.7.1
- pytorch==1.7.0
- pandas
- numpy
- datetime

如何运行：
-------
对于4个Jupyter Notebook文件，请使用Jupyter环境运行，并注意把其中的竞赛数据文件所在的文件夹替换为你自己保存数据文件的文件夹。
并记录下你处理完成后的数据文件所在的位置，供下面模型训练使用。

**注意：** 在运行*MAXP 2021初赛数据探索和处理-2*时，内存的使用量会比较高。这个在Mac上运行没有出现问题，但是尚未在Windows和Linux环境测试。
如果在这两种环境下遇到内存问题，建议找一个内存大一些的机器处理，或者修改代码，一部分一部分的处理。

---------
整体代码结构如下：

```bash
- data: 经过jupyter预处理后的特征和原数据存放处，一级目录
- gnn: baseline 自带
- max_model/config.yaml: 超参数、路径配置
- max_model/ *_yaml.py: 根据配置训练，测试(前几行有显卡固定设置，单卡)
- max_model/ model.py: 更新se
- MAXP 2021***-1,2,3,4.ipynb: 预处理 

```
预处理

```bash
--执行jupyter1-4做完预处理
```

训练，测试(单卡 24g) 

```bash
--python train_yaml.py
--python test_yaml.py
```
