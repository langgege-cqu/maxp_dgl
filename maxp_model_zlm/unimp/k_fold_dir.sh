CUDA_VISIBLE_DEVICES=0 python train_yaml.py --k_fold 0 >log/gatv2_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python train_yaml.py --k_fold 1 >log/gatv2_1.log 2>&1
CUDA_VISIBLE_DEVICES=0 python train_yaml.py --k_fold 2 >log/gatv2_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python train_yaml.py --k_fold 3 >log/gatv2_3.log 2>&1 &
