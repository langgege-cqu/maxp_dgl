import os
import gc
import time
import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

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

graph_path = os.path.join(base_path, publish_path, 'graph.bin')

graph = dgl.load_graphs(graph_path)[0][0]
print('graph', graph)


def MinMaxScaling(data):
    mind, maxd = data.min(), data.max()
    return mind + (data - mind) / (maxd - mind)


edge_feat = torch.cat((
    MinMaxScaling(graph.in_degrees().unsqueeze_(1).float().add(1).log()), 
    MinMaxScaling(graph.out_degrees().unsqueeze_(1).float().add(1).log())
), dim=1)
print('edge_feat', edge_feat.shape)

workers = 20
segs = np.linspace(0, graph.num_nodes(), num=workers + 1, endpoint=True, dtype=int)

def feat_map(i):
    node_list = list(range(segs[i], segs[i + 1]))
    
    tensor_list = []
    for k in node_list:
        tensor = torch.FloatTensor([
            edge_feat[graph.predecessors(k), 0].mean().item(), 
            edge_feat[graph.predecessors(k), 0].std().item(), 
            edge_feat[graph.predecessors(k), 1].mean().item(), 
            edge_feat[graph.predecessors(k), 1].std().item(), 
            edge_feat[graph.successors(k), 0].mean().item(), 
            edge_feat[graph.successors(k), 0].std().item(),
            edge_feat[graph.successors(k), 1].mean().item(),
            edge_feat[graph.successors(k), 1].std().item(),
        ])
        tensor_list.append(tensor)
    tensor_list = torch.stack(tensor_list, dim=0)
    return tensor_list
    

localtime = time.asctime(time.localtime(time.time()))
print('Start Extract', localtime)

with Pool(workers) as p:
    rets = p.map(feat_map, range(workers))
    
localtime = time.asctime(time.localtime(time.time()))
print('End Extract', localtime)


features_neigh = torch.cat(rets, dim=0)
print('features_neigh', features_neigh.shape)


features_neigh = torch.cat((edge_feat, features_neigh), dim=1).numpy()
features_neigh[np.isnan(features_neigh)] = 0.0
print('total feature', features_neigh.shape)

features_neigh_path = os.path.join(base_path, publish_path, 'features_neigh.npy')
with open(features_neigh_path, 'wb') as f:
    np.save(f, features_neigh)
    
