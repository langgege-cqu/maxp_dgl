log_name='gatv2_class'
# 关闭所有之前运行的程序
ps -ef | grep "python train_yaml" | cut -c 9-15 | xargs kill -s 9
CUDA_VISIBLE_DEVICES=0 python train_yaml.py --k_fold 0 >'log/'$log_name'_0.log' 2>&1 &&
    CUDA_VISIBLE_DEVICES=0 python train_yaml.py --k_fold 1 >'log/'$log_name'_1.log' 2>&1 &
CUDA_VISIBLE_DEVICES=1 python train_yaml.py --k_fold 2 >'log/'$log_name'_2.log' 2>&1 &&
    CUDA_VISIBLE_DEVICES=1 python train_yaml.py --k_fold 3 >'log/'$log_name'_3.log' 2>&1 &
