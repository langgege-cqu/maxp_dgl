train_name='first_try'
file_name='train_yaml.py'
log_dir='log'
device=(0 1 5 6 7)
fold=(0 1 2 3 4)
# 关闭所有之前运行的程序
ps -ef | grep "python "$file_name | cut -c 9-15 | xargs kill -s 9

# 启动并行脚本
i=0
for k in ${fold[*]}; do
    CUDA_VISIBLE_DEVICES=${device[$i]} python $file_name --k_fold $k  >$log_dir'/'$train_name'_'$k'.log' 2>&1 &
    ((i++));
done

# fold=(0 1 2 3 4)
# l = ${#fold}
# echo "长度为：$l"
# i=0
# for k in ${fold[*]}
# do
#     echo "$len $k $i"
#     i = `expr $i + 1`
# done


# arr=(12 36 )
# length=${#arr}
# echo "长度为：$length"