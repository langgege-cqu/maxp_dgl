import os
import sys
import yaml
import random
import argparse
import numpy as np
import torch as th
import pandas as pd
from tqdm import tqdm
from torch import optim
import torch.nn as thnn
import torch.nn.functional as F
import inspect
import dgl
from dgl.dataloading.neighbor import MultiLayerNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader

from unicmp import UniCMP
from util import load_dgl_graph

def print_hparams(hp):
    attributes = inspect.getmembers(hp, lambda a: not (inspect.isroutine(a)))
    return dict([a for a in attributes if not (a[0].startswith('__') and a[0].endswith('__'))])


def set_seed_logger(dataset_cfg):
    random.seed(dataset_cfg['SEED'])
    os.environ['PYTHONHASHSEED'] = str(dataset_cfg['SEED'])
    np.random.seed(dataset_cfg['SEED'])
    th.manual_seed(dataset_cfg['SEED'])
    th.cuda.manual_seed(dataset_cfg['SEED'])
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True

    
def init_model(model_cfg, k_fold, device):
    if model_cfg['GNN_MODEL'] == 'unicmp':
        model = UniCMP(input_size=model_cfg['INPUT_SIZE'], num_class=model_cfg['NUM_CLASS'],
                       num_layers=model_cfg['NUM_LAYERS'], num_heads=model_cfg['NUM_HEADS'],
                       hidden_size=model_cfg['HIDDEN_SIZE'], label_drop=model_cfg['LABEL_DROP'],
                       feat_drop=model_cfg['FEAT_DORP'], graph_drop=model_cfg['GRAPH_DORP'],
                       attn_drop=model_cfg['ATTN_DROP'], drop=model_cfg['DROP'], 
                       use_sage=model_cfg['USE_SAGE'], use_conv=model_cfg['USE_CONV'], 
                       use_attn=model_cfg['USE_ATTN'], use_resnet=model_cfg['USE_RESNET'], 
                       use_densenet=model_cfg['USE_DESNET'])
    else:
        raise NotImplementedError('Not support algorithm: {}'.format(model_cfg['GNN_MODEL']))
        
    state = th.load(model_cfg['CHECKPOINTS'][k_fold], map_location='cpu')
    model.load_state_dict(state)
    print('Load checkpoint in', model_cfg['CHECKPOINTS'])


    model = model.to(device)
    return model


def get_dataloader(dataset_cfg, graph_data):
    graph, labels, train_nid, val_nid, test_nid, node_feat = graph_data
    sampler = MultiLayerNeighborSampler(dataset_cfg['FANOUTS'])
    test_dataloader = NodeDataLoader(graph, test_nid, sampler, batch_size=dataset_cfg['BATCH_SIZE'],
                                     shuffle=False, drop_last=False, num_workers=0)

    return test_dataloader, node_feat, labels


def load_subtensor(node_feats, labels, seeds, input_nodes, n_classes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    input_feats = node_feats[input_nodes].to(device)
    
    masklabels = labels.clone()
    masklabels[seeds] = -1
    input_labels = masklabels[input_nodes]

    input_labels[input_labels < 0] = n_classes
    input_labels = input_labels.to(device)
    
    batch_labels = labels[seeds].to(device)
    return input_feats, input_labels, batch_labels


def test_epoch(model, test_dataloader, node_feats, labels, n_classes, device):
    th.cuda.empty_cache()
    model.eval()
    
    result = []
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    th.cuda.empty_cache()
    with th.no_grad():
        for step, (input_nodes, seeds, blocks) in pbar:
            input_feats, input_labels, batch_labels = load_subtensor(node_feats, labels, seeds, input_nodes, 
                                                                     n_classes, device)
            blocks = [block.to(device) for block in blocks]
            batch_logits = model(blocks, input_feats, input_labels)
            batch_logits = F.softmax(batch_logits, dim=-1)
            result.extend(batch_logits.detach().cpu().numpy().tolist())
            
    # result = np.concatenate(result, axis=0)
    return result


def id2name(x):
    return chr(x + 65)


def test(model_cfg, dataset_cfg, device, graph_data):
    set_seed_logger(dataset_cfg)
    graph, labels, train_nid, val_nid, test_nid, node_feats = graph_data
        
    test_dataloader, node_feats, labels = get_dataloader(dataset_cfg, graph_data)
    k_fold_result = []
    for k_fold in range(len(model_cfg['CHECKPOINTS'])):
        model = init_model(model_cfg, k_fold, device)
        result = test_epoch(model, test_dataloader, node_feats, labels, model_cfg['NUM_CLASS'], device)
        k_fold_result.append(result)
    k_fold_result = np.array(k_fold_result)
    k_fold_result = np.sum(k_fold_result, axis=0)
    result = np.argmax(k_fold_result, axis=1)
    nodes_path = os.path.join(dataset_cfg['DATA_PATH'], 'IDandLabels.csv')
    nodes_df = pd.read_csv(nodes_path, dtype={'Label':str})
        
    df = pd.DataFrame({'node_idx': test_nid, 'label': result})
    for row in df.itertuples():
        node_idx = getattr(row, 'node_idx')
        label = getattr(row, 'label')
        labels[node_idx] = label
    df['label'] = df['label'].apply(id2name)
    mged = pd.merge(df, nodes_df[['node_idx', 'paper_id']], on='node_idx', how='left')
    result_csv = dataset_cfg['K_FOLD_PATH']
    pd.DataFrame({'id': mged['paper_id'], 'label': mged['label']}).to_csv(result_csv, index=False)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Node Classification')
    parser.add_argument('--cfg_file', type=str, help="Path of config files.")
    args = parser.parse_args()
    yaml_path = args.cfg_file
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    model_cfg = config['MODEL']
    dataset_cfg = config['DATASET']

    device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
    dataset_cfg['BATCH_SIZE'] = int(dataset_cfg['BATCH_SIZE'] / dataset_cfg['GRADIENT_ACCUMULATION_STEPS'])
    graph_data = load_dgl_graph(dataset_cfg['DATA_PATH'], dataset_cfg['K_FOLD'])
        
    test(model_cfg, dataset_cfg, device, graph_data)
    