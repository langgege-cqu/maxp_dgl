import os
import dgl
import pickle
import numpy as np
import torch
import random
import logging
import sys


def load_dgl_graph(base_path, k, norm_feature=False):
    """
    读取预处理的Graph，Feature和Label文件，并构建相应的数据供训练代码使用。

    :param base_path:
    :return:
    """
    graphs, _ = dgl.load_graphs(os.path.join(base_path, 'graph.bin'))
    graph = graphs[0]
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph)
    print('################ Graph info: ###############')
    print(graph)

    with open(os.path.join(base_path, 'k_fold_labels.pkl'), 'rb') as f:
        label_data = pickle.load(f)

    labels = torch.from_numpy(label_data['label'])
    tr_label_idx = label_data['tr_label_idx'][k]
    val_label_idx = label_data['val_label_idx'][k]
    test_label_idx = label_data['test_label_idx']
    print('demo:  ', tr_label_idx.shape)
    print(val_label_idx.shape)
    print('################ Label info: ################')
    print('Total labels (including not labeled): {}'.format(labels.shape[0]))
    print('               Training label number: {}'.format(tr_label_idx.shape[0]))
    print('             Validation label number: {}'.format(val_label_idx.shape[0]))
    print('                   Test label number: {}'.format(test_label_idx.shape[0]))

    # get node features
    features = np.load(os.path.join(base_path, 'features.npy'))
    node_feat = torch.from_numpy(features).float()
    print('################ Feature info: ###############')
    print('Node\'s feature shape:{}'.format(node_feat.shape))

    if norm_feature:
        node_feat = torch.nn.functional.normalize(node_feat, p=2.0, dim=-1)

        degs = graph.out_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5)
        shp = norm.shape + (1, ) * (node_feat.dim() - 1)
        norm = torch.reshape(norm, shp)
        node_feat = node_feat * norm
        print('Norm Feature Succeed')

    graph_data = (graph, labels, tr_label_idx, val_label_idx, test_label_idx, node_feat)
    return graph_data


def set_seed_logger(train_cfg):
    random.seed(train_cfg['SEED'])
    os.environ['PYTHONHASHSEED'] = str(train_cfg['SEED'])
    np.random.seed(train_cfg['SEED'])
    torch.manual_seed(train_cfg['SEED'])
    torch.cuda.manual_seed(train_cfg['SEED'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    os.makedirs(train_cfg['OUT_PATH'], exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s: %(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(train_cfg['OUT_PATH'], 'train.log')),
                            logging.StreamHandler(sys.stdout)
                        ])