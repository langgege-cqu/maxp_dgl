python deepwalk.py --data_file ../final_dataset/graph.bin --output_emb_file ../final_dataset/deep_walk1.npy --mix --gpus 0 1 2 3 4 5 6 7 --batch_size 5000 --window_size 5 --num_walks 50 --negative 1 --neg_weight 1 --walk_length 80  --lap_norm 0.01 --lr 0.1 --use_context_weight

python deepwalk.py --data_file ../final_dataset/graph.bin --output_emb_file ../final_dataset/deep_walk2.npy --mix --gpus 0 1 2 3 4 5 6 7 --batch_size 5000 --window_size 5 --num_walks 50 --negative 1 --neg_weight 1 --walk_length 100 --lap_norm 0.01 --lr 0.1 --use_context_weight
