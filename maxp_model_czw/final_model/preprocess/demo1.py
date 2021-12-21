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

link_p1_path = os.path.join(base_path, publish_path, 'link_phase1.csv')
link_p2_path = os.path.join(base_path, publish_path, 'link_phase2.csv')
train_nodes_path = os.path.join(base_path, publish_path, 'train_nodes.csv')
val_nodes_path = os.path.join(base_path, publish_path, 'validation_nodes.csv')
test_nodes_path = os.path.join(base_path, publish_path, 'test_nodes.csv')


print('=' * 50)
edge_df1 = pd.read_csv(link_p1_path)
print('\nedge_df1:\n', edge_df1)
edge_df2 = pd.read_csv(link_p2_path)
print('\nedge_df2:\n', edge_df2)
edge_df = pd.concat((edge_df1, edge_df2), ignore_index=True)
print('\nedge_df:\n', edge_df)
edge_df.to_csv(os.path.join(base_path, publish_path, 'link_phase.csv'), index=False)


print('=' * 50)
node_df1 = pd.read_csv(train_nodes_path)
print('\nnode_df1:\n', node_df1)
node_df2 = pd.read_csv(val_nodes_path)
print('\nnode_df2:\n', node_df2)
node_df = pd.concat((node_df1, node_df2), ignore_index=True)
print('\nnode_df:\n', node_df)
node_df.to_csv(os.path.join(base_path, publish_path, 'train_val_nodes.csv'), index=False)
