CUDA_VISIBLE_DEVICES=0 python train_dgl.py --data_path ../data --gnn_model graphmodel \
--batch_size 1024 --gradient_accumulation_steps 1 --log_step 240 \
--n_layers 2 --fanouts 50,50 --out_path graphmodel/split01