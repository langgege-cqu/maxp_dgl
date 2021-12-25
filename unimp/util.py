import os
import dgl
import pickle
import numpy as np
import torch as th


def load_dgl_graph(graph_path, label_path, node_path, walk_path, edge_path, train=True):
    graph = dgl.load_graphs(graph_path)[0][0]
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    print('################ Graph info: ###############')
    print(graph)

    with open(label_path, 'rb') as f:
        label_data = pickle.load(f)
    labels = th.from_numpy(label_data['label'])
    
    if train:
        tr_label_idx = label_data['tr_label_idx']
        val_label_idx = label_data['val_label_idx']
        test_label_idx = label_data['test_label_idx']
        print('################ Label info: ################')
        print('Total labels (including not labeled): {}'.format(labels.shape[0]))
        print('               Training label number: {}'.format(tr_label_idx.shape[0]))
        print('             Validation label number: {}'.format(val_label_idx.shape[0]))
        print('                   Test label number: {}'.format(test_label_idx.shape[0]))
    else:
        tr_label_idx, val_label_idx = None, None
        test_label_idx = label_data['test_label_idx']
        print('################ Label info: ################')
        print('Total labels (including not labeled): {}'.format(labels.shape[0]))
        print('                   Test label number: {}'.format(test_label_idx.shape[0]))

    node_feat = th.from_numpy(np.load(node_path)).float()
    walk_feat = th.from_numpy(np.load(walk_path)).float()
    edge_feat = th.from_numpy(np.load(edge_path)).float()
    features = th.cat((node_feat, walk_feat, edge_feat), dim=1)
    
    print('################ Feature info: ###############')
    print('Node\'s feature shape:{}'.format(node_feat.shape))
    print('Walk\'s feature shape:{}'.format(walk_feat.shape))
    print('Edge\'s feature shape:{}'.format(edge_feat.shape))
    print('Total\'s feature shape:{}'.format(features.shape))
    
    graph_data = (graph, labels, tr_label_idx, val_label_idx, test_label_idx, features)
    return graph_data
