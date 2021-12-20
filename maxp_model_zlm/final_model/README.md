# 下载数据集
# 链接数据集到当前文件夹下
```shell
ln -s [xxx]/[xxx]/dataset ./final_dataset
```

# 安装依赖
```shell
conda env create -f dgl_env.yaml
```

# 预处理
```shell
bash preprocess.sh
```

# K折训练
进入`unimp`文件夹

修改`config.yaml`中的
1. `DATASET.WALK_PATH`
2. `DATASET.OUT_PATH`

根据硬件和训练需求修改`k_fold_train.sh`

然后运行
```shell
bash k_fold_train.sh
```

训练的模型位于`final_model/final_models/[xxx]/split[x]/models`

# K折预测
进入`unimp`文件夹

修改`config.yaml`中的
1. `MODEL.CHECKPOINT_BASE`
2. `MODEL.CHECKPOINT_LIST`

根据硬件和训练需求修改`k_fold_test.sh`

然后运行
```shell
bash k_fold_test.sh
```
结果文件位于输出路径中