import os
import gc
import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import dgl


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_seed(2021)


base_path = '../final_dataset'
publish_path = '.'

nodes_path = os.path.join(base_path, publish_path, 'IDandLabels.csv')


nodes_df = pd.read_csv(nodes_path, dtype={'Label':str})
print('=' * 50, '\nnodes_df\n', nodes_df)


label_dist = nodes_df.groupby(by='Label').count()
print('=' * 50, '\nlabel_dist\n', label_dist)
print('label sum', label_dist['Split_ID'].sum())

label2idx = {}
for i, l in enumerate(label_dist.index.to_list()):
    nodes_df.loc[(nodes_df.Label == l), 'label'] = i
    label2idx[l] = i
print('=' * 50, '\nlabel2idx\n', label2idx)


nodes_df.label.fillna(-1, inplace=True)
nodes_df.label = nodes_df.label.astype('int')
nodes = nodes_df[['node_idx', 'label', 'Split_ID']]
print('=' * 50, '\nnodes\n', nodes_df)


print('=' * 50)
tr_val_labels_df = nodes[(nodes.Split_ID == 0) & (nodes.label >= 0)]
test_label_df = nodes[nodes.Split_ID == 1]
print('tr_val_labels_df\n', tr_val_labels_df)
print('test_label_df\n', test_label_df)

split = 5
tr_labels_idx = [np.array([0]) for _ in range(split)]
val_labels_idx = [np.array([0]) for _ in range(split)]

for label in range(len(label2idx)):
    label_idx = tr_val_labels_df[tr_val_labels_df.label == label].node_idx.to_numpy()
    # np.random.shuffle(label_idx)
    segs = np.linspace(0, label_idx.shape[0], num=split + 1, endpoint=True, dtype=int)
    
    for i in range(split): 
        if i > 0:
            tr_labels_idx[i] = np.append(tr_labels_idx[i], label_idx[: segs[i]])

        if i < split - 1:
            tr_labels_idx[i] = np.append(tr_labels_idx[i], label_idx[segs[i + 1]:])
            
        val_labels_idx[i] = np.append(val_labels_idx[i], label_idx[segs[i]: segs[i + 1]])

tr_labels_idx = [tr_labels_idx[i][1:] for i in range(split)]
val_labels_idx = [val_labels_idx[i][1:] for i in range(split)]
test_labels_idx = test_label_df.node_idx.to_numpy()
labels = nodes.label.to_numpy()


print('=' * 50)
for i in range(split):
    label_path = os.path.join(base_path, publish_path, 'label_split{}.pkl'.format(i + 1))
    
    with open(label_path, 'wb') as f:
        pickle.dump({'tr_label_idx': tr_labels_idx[i], 
                     'val_label_idx': val_labels_idx[i], 
                     'test_label_idx': test_labels_idx,
                     'label': labels}, f)
    print('Process split {} | tr_labels_idx {} | val_labels_idx {} | test_labels_idx {} | labels {}'.format(
        i, tr_labels_idx[i].shape, val_labels_idx[i].shape, test_labels_idx.shape, labels.shape))
