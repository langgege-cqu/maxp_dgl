import os
import random
import inspect
import argparse
import numpy as np
import torch as th
import pandas as pd
from tqdm import tqdm
import torch.nn as thnn
import torch.nn.functional as F
import yaml
import dgl
from dgl.dataloading.neighbor import MultiLayerNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader
from unimp import GNNModel
from models import GraphSageModel, GraphConvModel, GraphAttnModel, GraphModel
from utils import load_dgl_graph


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
th.cuda.set_device(0)


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
    
    os.makedirs(dataset_cfg['OUT_PATH'], exist_ok=True)

    
def init_model(model_cfg, in_feat, device):
    if model_cfg['GNN_MODEL'] == 'graphsage':
        model = GraphModel(1, model_cfg['IN_FEAT'], model_cfg['HIDDEN_DIM'], model_cfg['N_LAYERS'], model_cfg['MLP_DIM'],
                           num_attention_heads=model_cfg['NUM_ATTENTION_HEADS'], n_classes=model_cfg['N_CLASS'],
                           activation=F.relu, input_drop=model_cfg['INPUT_DROP'], drop_rate=model_cfg['DROP_RATE'],
                           attn_drop=model_cfg['ATTN_DROP'])
    elif model_cfg['GNN_MODEL'] == 'graphconv':
        model = GraphModel(2, model_cfg['IN_FEAT'], model_cfg['HIDDEN_DIM'], model_cfg['N_LAYERS'], model_cfg['MLP_DIM'],
                           num_attention_heads=model_cfg['NUM_ATTENTION_HEADS'], n_classes=model_cfg['N_CLASS'],
                           activation=F.relu, input_drop=model_cfg['INPUT_DROP'], drop_rate=model_cfg['DROP_RATE'],
                           attn_drop=model_cfg['ATTN_DROP'])
    elif model_cfg['GNN_MODEL'] == 'graphattn':
        model = GraphModel(3, model_cfg['IN_FEAT'], model_cfg['HIDDEN_DIM'], model_cfg['N_LAYERS'], model_cfg['MLP_DIM'],
                           num_attention_heads=model_cfg['NUM_ATTENTION_HEADS'], n_classes=model_cfg['N_CLASS'],
                           activation=F.relu, input_drop=model_cfg['INPUT_DROP'], drop_rate=model_cfg['DROP_RATE'],
                           attn_drop=model_cfg['ATTN_DROP'])
    elif model_cfg['GNN_MODEL'] == 'graphmodel':
        model = GraphModel(0, model_cfg['IN_FEAT'], model_cfg['HIDDEN_DIM'], model_cfg['N_LAYERS'], model_cfg['MLP_DIM'],
                           num_attention_heads=model_cfg['NUM_ATTENTION_HEADS'], n_classes=model_cfg['N_CLASS'],
                           activation=F.relu, input_drop=model_cfg['INPUT_DROP'], drop_rate=model_cfg['DROP_RATE'],
                           attn_drop=model_cfg['ATTN_DROP'])
    elif model_cfg['GNN_MODEL'] == 'unimp':
        model = GNNModel(input_size=model_cfg['IN_FEAT'], num_class=model_cfg['N_CLASS'])
    else:
        raise NotImplementedError('So far, only support three algorithms: GraphSage, GraphConv, and GraphAttn')
 

    checkpoint = th.load(model_cfg['CHECKPOINT'])  # map_location='cpu'
    model.load_state_dict(checkpoint)
 
    model = model.to(device)
    return model


def get_dataloader(dataset_cfg, graph_data):
    graph, labels, train_nid, val_nid, test_nid, node_feat = graph_data
    sampler = MultiLayerNeighborSampler(dataset_cfg['FANOUTS'])
    test_dataloader = NodeDataLoader(graph, test_nid, sampler, batch_size=dataset_cfg['BATCH_SIZE'],
                                     shuffle=False, drop_last=False, num_workers=0)

    return test_dataloader, node_feat, labels


def load_subtensor(node_feats, labels, seeds, input_nodes, n_classes, device, training=False):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    input_feats = node_feats[input_nodes].to(device)
    
    masklabels = labels.clone()
    masklabels[seeds] = -1
    input_labels = masklabels[input_nodes]
        
    if training:
        rd_m = th.rand(input_labels.shape[0]) 
        rd_y = th.randint(0, n_classes, size=input_labels.shape)
        input_labels[rd_m < 0.12] = -1
        input_labels[rd_m > 0.97] = rd_y[rd_m > 0.97]

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
                                                                     n_classes, device, training=False)
            blocks = [block.to(device) for block in blocks]
            batch_logits = model(blocks, input_feats, input_labels)
            batch_results = th.argmax(batch_logits, dim=1)
            result.extend(batch_results.detach().cpu().numpy().tolist())

    return result
 
    
def id2name(x):
    return chr(x + 65)


def test(model_cfg, dataset_cfg, device, graph_data):
    set_seed_logger(dataset_cfg)
    graph, labels, train_nid, val_nid, test_nid, node_feats = graph_data
        
    test_dataloader, node_feats, labels = get_dataloader(dataset_cfg, graph_data)

    in_feat = node_feats.shape[1]
    model = init_model(model_cfg, in_feat, device)
    print('  Model = %s', str(model))
 
    nodes_path = os.path.join(dataset_cfg['DATA_PATH'], 'IDandLabels.csv')
    nodes_df = pd.read_csv(nodes_path, dtype={'Label':str})
    
    
    result = test_epoch(model, test_dataloader, node_feats, labels, model_cfg['N_CLASS'], device)
    df = pd.DataFrame({'node_idx': test_nid, 'label': result})
    for row in df.itertuples():
        node_idx = getattr(row, 'node_idx')
        label = getattr(row, 'label')
        labels[node_idx] = label
    df['label'] = df['label'].apply(id2name)
    mged = pd.merge(df, nodes_df[['node_idx', 'paper_id']], on='node_idx', how='left')
    result_csv = dataset_cfg['TEST_RESULT']
    pd.DataFrame({'id': mged['paper_id'], 'label': mged['label']}).to_csv(result_csv, index=False)
        

if __name__ == '__main__':

    yaml_path = 'config.yaml'
    f = open(yaml_path, 'r', encoding='utf-8')
    config = f.read()
    config = yaml.load(config)
    model_cfg = config['MODEL']
    dataset_cfg = config['DATASET']

    device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
    graph_data = load_dgl_graph(dataset_cfg['DATA_PATH'], norm_feature=dataset_cfg['NORM_FEATURE'])
        
    test(model_cfg, dataset_cfg, device, graph_data)
