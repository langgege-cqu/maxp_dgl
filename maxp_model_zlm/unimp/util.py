import os
import dgl
import pickle
import numpy as np
import torch as th


def minMaxScaling(data):
    mind, maxd = data.min(), data.max()
    return mind + (data - mind) / (maxd - mind)


def load_dgl_graph(dataset_cfg):
    base_path = dataset_cfg['DATA_PATH']
    k = dataset_cfg['K_FOLD']
    to_bidirected = dataset_cfg['BIDIRECTED']

    graphs, _ = dgl.load_graphs(os.path.join(base_path, 'graph.bin'))
    edge_feat = th.cat(
        (
            minMaxScaling(graphs[0].in_degrees().unsqueeze_(1).float().add(1).log()
                          ), minMaxScaling(graphs[0].out_degrees().unsqueeze_(1).float().add(1).log())
        ),
        dim=1
    )

    graph = graphs[0]
    if to_bidirected:
        graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph)
    print('################ Graph info: ###############')
    print(graph)

    if k is None:
        with open(os.path.join(base_path, 'labels.pkl'), 'rb') as f:
            label_data = pickle.load(f)
            labels = th.from_numpy(label_data['label'])
            tr_label_idx = label_data['tr_label_idx']
            val_label_idx = label_data['val_label_idx']
            test_label_idx = label_data['test_label_idx']
    else:
        with open(os.path.join(base_path, 'k_fold_labels.pkl'), 'rb') as f:
            label_data = pickle.load(f)
            labels = th.from_numpy(label_data['label'])
            tr_label_idx = label_data['tr_label_idx'][k]
            val_label_idx = label_data['val_label_idx'][k]
            test_label_idx = label_data['test_label_idx']

    print('################ Label info: ################')
    print('Total labels (including not labeled): {}'.format(labels.shape[0]))
    print('               Training label number: {}'.format(tr_label_idx.shape[0]))
    print('             Validation label number: {}'.format(val_label_idx.shape[0]))
    print('                   Test label number: {}'.format(test_label_idx.shape[0]))

    node_feat = th.from_numpy(np.load(os.path.join(base_path, 'features.npy'))).float()
    walk_feat = th.from_numpy(np.load(os.path.join(base_path, dataset_cfg['DEEPWALK_PATH']))).float()
    neighbor_feat = th.from_numpy(np.load(os.path.join(base_path, dataset_cfg['NEIGHBOR_FEATURES_PATH']))).float()
    # features = th.cat((node_feat, walk_feat, edge_feat, neighbor_feat), dim=1)
    features = th.cat((node_feat, walk_feat, edge_feat), dim=1)

    print('################ Feature info: ###############')
    print('Node\'s feature shape:{}'.format(node_feat.shape))
    print('Walk\'s feature shape:{}'.format(walk_feat.shape))
    print('Edge\'s feature shape:{}'.format(edge_feat.shape))

    graph_data = (graph, labels, tr_label_idx, val_label_idx, test_label_idx, features)
    return graph_data
