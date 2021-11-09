CUDA_VISIBLE_DEVICES=7 python test_dgl.py --data_path dataset/split01 --gnn_model graphmodel \
--batch_size 1024 --checkpoint graphmodel/split01/dgl_model_epoch.pth --test_epochs 3 \
--out_path graphmodel/split01

CUDA_VISIBLE_DEVICES=7 python test_dgl.py --data_path dataset/split02 --gnn_model graphmodel \
--batch_size 1024 --checkpoint graphmodel/split02/dgl_model_epoch.pth --test_epochs 3 \
--out_path graphmodel/split02

CUDA_VISIBLE_DEVICES=7 python test_dgl.py --data_path dataset/split03 --gnn_model graphmodel \
--batch_size 1024 --checkpoint graphmodel/split03/dgl_model_epoch.pth --test_epochs 3 \
--out_path graphmodel/split03

CUDA_VISIBLE_DEVICES=7 python test_dgl.py --data_path dataset/split04 --gnn_model graphmodel \
--batch_size 1024 --checkpoint graphmodel/split04/dgl_model_epoch.pth --test_epochs 3 \
--out_path graphmodel/split04

CUDA_VISIBLE_DEVICES=7 python test_dgl.py --data_path dataset/split05 --gnn_model graphmodel \
--batch_size 1024 --checkpoint graphmodel/split05/dgl_model_epoch.pth --test_epochs 3 \
--out_path graphmodel/split05

CUDA_VISIBLE_DEVICES=7 python test_dgl.py --data_path dataset/split06 --gnn_model graphmodel \
--batch_size 1024 --checkpoint graphmodel/split06/dgl_model_epoch.pth --test_epochs 3 \
--out_path graphmodel/split06

CUDA_VISIBLE_DEVICES=7 python test_dgl.py --data_path dataset/split07 --gnn_model graphmodel \
--batch_size 1024 --checkpoint graphmodel/split07/dgl_model_epoch.pth --test_epochs 3 \
--out_path graphmodel/split07

CUDA_VISIBLE_DEVICES=7 python test_dgl.py --data_path dataset/split08 --gnn_model graphmodel \
--batch_size 1024 --checkpoint graphmodel/split08/dgl_model_epoch.pth --test_epochs 3 \
--out_path graphmodel/split08

CUDA_VISIBLE_DEVICES=7 python test_dgl.py --data_path dataset/split09 --gnn_model graphmodel \
--batch_size 1024 --checkpoint graphmodel/split09/dgl_model_epoch.pth --test_epochs 3 \
--out_path graphmodel/split09

CUDA_VISIBLE_DEVICES=7 python test_dgl.py --data_path dataset/split10 --gnn_model graphmodel \
--batch_size 1024 --checkpoint graphmodel/split10/dgl_model_epoch.pth --test_epochs 3 \
--out_path graphmodel/split10