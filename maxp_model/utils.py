import os
import dgl
import pickle
import numpy as np
import torch as th


def load_dgl_graph(base_path, norm_feature=False):
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

    with open(os.path.join(base_path, 'labels.pkl'), 'rb') as f:
        label_data = pickle.load(f)

    labels = th.from_numpy(label_data['label'])
    tr_label_idx = label_data['tr_label_idx']
    val_label_idx = label_data['val_label_idx']
    test_label_idx = label_data['test_label_idx']
    print('################ Label info: ################')
    print('Total labels (including not labeled): {}'.format(labels.shape[0]))
    print('               Training label number: {}'.format(tr_label_idx.shape[0]))
    print('             Validation label number: {}'.format(val_label_idx.shape[0]))
    print('                   Test label number: {}'.format(test_label_idx.shape[0]))

    # get node features
    features = np.load(os.path.join(base_path, 'features.npy'))
    node_feat = th.from_numpy(features).float()
    print('################ Feature info: ###############')
    print('Node\'s feature shape:{}'.format(node_feat.shape))
    
    
    if norm_feature:
        node_feat = th.nn.functional.normalize(node_feat, p=2.0, dim=-1)
        
        degs = graph.out_degrees().float().clamp(min=1)
        norm = th.pow(degs, -0.5)
        shp = norm.shape + (1,) * (node_feat.dim() - 1)
        norm = th.reshape(norm, shp)
        node_feat = node_feat * norm
        print('Norm Feature Succeed')
    
    graph_data = (graph, labels, tr_label_idx, val_label_idx, test_label_idx, node_feat)
    return graph_data
