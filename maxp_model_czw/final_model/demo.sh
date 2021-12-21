# 环境配置
cd /hetu_group/wuxiangyu/cx/language_model/DGL/dgl_maxp_model
conda activate /hetu_group/wuxiangyu/cx/envs/torch
export PATH=/hetu_group/wuxiangyu/cx/envs/torch/bin:$PATH


# bypy下载复赛数据集
cd /hetu_group/wuxiangyu/cx/language_model/DGL/dgl_maxp_model/final_dataset
bypy list
bypy --downloader aria2 download final_dataset


# 数据预处理
cd /hetu_group/wuxiangyu/cx/language_model/DGL/dgl_maxp_model/preprocess
# demo1合并边连接关系文件，合并初赛的验证集与测试集节点
python demo1.py
# demo2处理边和节点文件，创建对应节点映射文件与特征文件
python demo2.py
# demo3根据边关系创建并保存图
python demo3.py
# demo4分层抽样5折划分数据集
python demo4.py
# demo5提取并保存出入度以及一阶二阶特征,总维度2+8=10
python demo5.py


# 提取并保存deep walk特征，总维度128
cd /hetu_group/wuxiangyu/cx/language_model/DGL/dgl_maxp_model/deep_walk
bash run.sh


# 模型训练以及预测
cd /hetu_group/wuxiangyu/cx/language_model/DGL/dgl_maxp_model/unimp
CUDA_VISIBLE_DEVICES=x python train_yaml.py --cfg_file configs/configx.yaml
CUDA_VISIBLE_DEVICES=x python test_yaml.py --cfg_file ../final_models/deepwalkx/unimp_splitx/config.yaml