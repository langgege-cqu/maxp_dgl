log_name='gatv2_pseudo'
# 关闭所有之前运行的程序
ps -ef | grep "python train_yaml" | cut -c 9-15 | xargs kill -s 9
# ps -ef | grep " python train_yaml.py --k_fold 5" | cut -c 9-15 | xargs kill -s 9

cd unimp
# 训练K折
# 每张卡训练两到三折
# CUDA_VISIBLE_DEVICES=0 python train_yaml.py --k_fold 1 >'log/'$log_name'_0.log' 2>&1 &&
#     CUDA_VISIBLE_DEVICES=0 python train_yaml.py --k_fold 2 >'log/'$log_name'_1.log' 2>&1 &
# CUDA_VISIBLE_DEVICES=1 python train_yaml.py --k_fold 3 >'log/'$log_name'_2.log' 2>&1 &&
#     CUDA_VISIBLE_DEVICES=1 python train_yaml.py --k_fold 4 >'log/'$log_name'_3.log' 2>&1 &&
#         CUDA_VISIBLE_DEVICES=1 python train_yaml.py --k_fold 5 >'log/'$log_name'_4.log' 2>&1 &

# 每张卡训练一折
CUDA_VISIBLE_DEVICES=0 python train_yaml.py --k_fold 1 >'log/'$log_name'_1.log' 2>&1 &
CUDA_VISIBLE_DEVICES=1 python train_yaml.py --k_fold 2 >'log/'$log_name'_2.log' 2>&1 &
CUDA_VISIBLE_DEVICES=2 python train_yaml.py --k_fold 3 >'log/'$log_name'_3.log' 2>&1 &
CUDA_VISIBLE_DEVICES=3 python train_yaml.py --k_fold 4 >'log/'$log_name'_4.log' 2>&1 &&
    CUDA_VISIBLE_DEVICES=3 python train_yaml.py --k_fold 5 >'log/'$log_name'_5.log' 2>&1 &

# # 测试
# CUDA_VISIBLE_DEVICES=3 python train_yaml.py --k_fold 4 >log/log_4.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python train_yaml.py --k_fold 5 >log/gatv2_pseudo_5.log 2>&1 &