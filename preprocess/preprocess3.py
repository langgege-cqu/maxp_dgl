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

link_p1_path = os.path.join(base_path, publish_path, 'link_phase_drop.csv')
nodes_path = os.path.join(base_path, publish_path, 'IDandLabels.csv')


nodes_df = pd.read_csv(nodes_path, dtype={'Label':str})
print('=' * 50, '\nnodes_df\n', nodes_df)


edges_df = pd.read_csv(link_p1_path)
print('=' * 50, '\nedges_df\n', edges_df)


edges = edges_df.merge(nodes_df, on='paper_id', how='left')
edges = edges.merge(nodes_df, left_on='reference_paper_id', right_on='paper_id', how='left')
edges.rename(columns={'paper_id_x': 'paper_id', 'node_idx_x': 'src_nid', 'node_idx_y': 'dst_nid'}, inplace=True)
edges = edges[['src_nid', 'dst_nid', 'paper_id', 'reference_paper_id']]
print('=' * 50, '\nedges\n', edges)


src_nid = edges.src_nid.to_numpy()
dst_nid = edges.dst_nid.to_numpy()
graph = dgl.graph((src_nid, dst_nid))
print('=' * 50, '\ngraph\n', graph)

graph_path = os.path.join(base_path, publish_path, 'graph.bin')
dgl.data.utils.save_graphs(graph_path, [graph])
