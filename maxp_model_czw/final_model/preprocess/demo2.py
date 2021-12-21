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

link_p1_path = os.path.join(base_path, publish_path, 'link_phase.csv')
train_nodes_path = os.path.join(base_path, publish_path, 'train_val_nodes.csv')
val_nodes_path = os.path.join(base_path, publish_path, 'test_nodes.csv')


edge_df = pd.read_csv(link_p1_path)
print('=' * 50, '\nedge_df:\n', edge_df)


nodes = pd.concat([edge_df['paper_id'], edge_df['reference_paper_id']])
nodes = pd.DataFrame(nodes.drop_duplicates())
nodes.rename(columns={0:'paper_id'}, inplace=True)
print('=' * 50, '\nnodes\n', nodes)


def process_node(line):
    nid, feat_json, label = line.strip().split('\"')
    
    feat_list = [float(feat[1:-1]) for feat in feat_json[1:-1].split(', ')]
    
    if len(feat_list) != 300:
        print('此行数据有问题 {}'.format(line))
    
    return nid[:-1], feat_list, label[1:]


nid_list = []
label_list = []
tr_val_list = []

train_nodes_num = 0
with open(train_nodes_path, 'r') as f:
    for line in f:
        train_nodes_num += 1
        
with open(train_nodes_path, 'r') as f:
    pbar = tqdm(f, total=train_nodes_num)
    for i, line in enumerate(pbar):
        pbar.set_description('Process train nodes | line : {}'.format(i))
        if i > 0:
            nid, _, label = process_node(line)
            nid_list.append(nid)
            label_list.append(label)
            tr_val_list.append(0)

val_nodes_num = 0
with open(val_nodes_path, 'r') as f:
    for line in f:
        val_nodes_num += 1          
            
with open(val_nodes_path, 'r') as f:
    pbar = tqdm(f, total=val_nodes_num)
    for i, line in enumerate(pbar):
        pbar.set_description('Process validation nodes | line : {}'.format(i))
        if i > 0:
            nid, _, label = process_node(line)
            nid_list.append(nid)
            label_list.append(label)
            tr_val_list.append(1)
            
nid_arr = np.array(nid_list)
label_arr = np.array(label_list)
tr_val_arr = np.array(tr_val_list)
    

nid_label_df = pd.DataFrame({'paper_id':nid_arr, 'Label': label_arr, 'Split_ID':tr_val_arr})
nid_label_df.reset_index(inplace=True)
nid_label_df.rename(columns={'index':'node_idx'}, inplace=True)
print('=' * 50, '\nnid_label_df\n', nid_label_df)


print('=' * 50)
ids = nid_label_df.paper_id.drop_duplicates()
if len(nid_label_df) == len(ids):
    print('ID在Train和Validation没有重复')
else:
    print('ID在Train和Validation重复重复数目: {}'.format(len(nid_label_df) - len(ids)))
    
    
inboth = nid_label_df.merge(nodes, on='paper_id', how='inner')
print('=' * 50, '\ninboth\n', inboth)

edge_node = nodes.merge(nid_label_df, on='paper_id', how='left')
print('=' * 50, '\nedge_node\n', edge_node)
print('共有{}边列表的节点在给出的节点列表里没有对应，缺乏特征'.format(edge_node[edge_node.node_idx.isna()].shape[0]))


diff_nodes = edge_node[edge_node.node_idx.isna()]
num_miss = len(diff_nodes)
diff_nodes.ID = diff_nodes.paper_id
diff_nodes.Split_ID = 1
diff_nodes.node_idx = 0
diff_nodes.reset_index(inplace=True)
diff_nodes.drop(['index'], axis=1, inplace=True)
diff_nodes.node_idx = diff_nodes.node_idx + diff_nodes.index + len(nid_label_df)
diff_nodes = diff_nodes[['node_idx', 'paper_id', 'Label', 'Split_ID']]
print('=' * 50, '\ndiff_nodes\n', diff_nodes)


nid_label_df = pd.concat([nid_label_df, diff_nodes])
nid_label_df.to_csv(os.path.join(base_path, publish_path, 'IDandLabels.csv'), index=False)
print('Save nid_label_df in file: IDandLabels.csv')

del edge_df, nodes, nid_label_df, inboth, edge_node, diff_nodes
gc.collect()



feature_list = []

with open(train_nodes_path, 'r') as f:
    pbar = tqdm(f, total=train_nodes_num)
    for i, line in enumerate(pbar):
        pbar.set_description('Process train nodes | line : {}'.format(i))
        if i > 0:
            _, features, _ = process_node(line)
            feature_list.append(features)

with open(val_nodes_path, 'r') as f:
    pbar = tqdm(f, total=val_nodes_num)
    for i, line in enumerate(pbar):
        pbar.set_description('Process validation nodes | line : {}'.format(i))
        if i > 0:
            _, features, _ = process_node(line)
            feature_list.append(features)
            
feat_arr = np.array(feature_list)
del feature_list
gc.collect()


diff_node_feat_arr = np.tile(np.mean(feat_arr, axis=0), (num_miss, 1))
feat_arr = np.concatenate((feat_arr, diff_node_feat_arr), axis=0)
print('feat_arr shape', feat_arr.shape)

with open(os.path.join(base_path, publish_path, 'features.npy'), 'wb') as f:
    np.save(f, feat_arr)
print('Save feat_arr in file: features.npy')