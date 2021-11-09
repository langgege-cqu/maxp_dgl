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

import dgl
from dgl.dataloading.neighbor import MultiLayerNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader

from models import GraphSageModel, GraphConvModel, GraphAttnModel, GraphModel
from utils import load_dgl_graph


def print_hparams(hp):
    attributes = inspect.getmembers(hp, lambda a: not (inspect.isroutine(a)))
    return dict([a for a in attributes if not (a[0].startswith('__') and a[0].endswith('__'))])


def get_args(description='Graph Node Classification'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_path', type=str, help="Path of saved processed data files.")
    parser.add_argument('--gnn_model', type=str, choices=['graphsage', 'graphconv', 'graphattn', 'graphmodel'],
                        required=True, default='graphsage')
    parser.add_argument('--norm_feature', action='store_true')
    
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--mlp_dim', type=int, default=256)
    parser.add_argument('--num_attention_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_classes', type=int, default=23)
    parser.add_argument("--fanouts", type=str, help="fanout numbers", default='50,50')
    parser.add_argument("--input_drop", type=float, default=0.3)
    parser.add_argument("--drop_rate", type=float, default=0.5)
    parser.add_argument("--attn_drop", type=float, default=0.2)

    parser.add_argument('--batch_size', type=int, required=True, default=1024)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--out_path', type=str, required=True, help="Absolute path for saving model parameters")
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--test_epochs', type=int, default=3)
    args = parser.parse_args()

    args.fanouts = [int(i) for i in args.fanouts.split(',')]

    return args


def set_seed_logger(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed(args.seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
    
    os.makedirs(args.out_path, exist_ok=True)

    
def init_model(args, in_feat, device):
    if args.gnn_model == 'graphsage':
        model = GraphModel(1, in_feat, args.hidden_dim, args.n_layers, mlp_dim=args.mlp_dim,
                           num_attention_heads=args.num_attention_heads, n_classes=args.n_classes,
                           activation=F.relu, input_drop=args.input_drop, drop_rate=args.drop_rate,
                           attn_drop=args.attn_drop)
    elif args.gnn_model == 'graphconv':
        model = GraphModel(2, in_feat, args.hidden_dim, args.n_layers, mlp_dim=args.mlp_dim,
                           num_attention_heads=args.num_attention_heads, n_classes=args.n_classes,
                           activation=F.relu, input_drop=args.input_drop, drop_rate=args.drop_rate,
                           attn_drop=args.attn_drop)
    elif args.gnn_model == 'graphattn':
        model = GraphModel(3, in_feat, args.hidden_dim, args.n_layers, mlp_dim=args.mlp_dim,
                           num_attention_heads=args.num_attention_heads, n_classes=args.n_classes,
                           activation=F.relu, input_drop=args.input_drop, drop_rate=args.drop_rate,
                           attn_drop=args.attn_drop)
    elif args.gnn_model == 'graphmodel':
        model = GraphModel(0, in_feat, args.hidden_dim, args.n_layers, mlp_dim=args.mlp_dim,
                           num_attention_heads=args.num_attention_heads, n_classes=args.n_classes,
                           activation=F.relu, input_drop=args.input_drop, drop_rate=args.drop_rate,
                           attn_drop=args.attn_drop)
 
    else:
        raise NotImplementedError('So far, only support three algorithms: GraphSage, GraphConv, and GraphAttn')
 
    checkpoint = th.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint)
 
    model = model.to(device)
    return model


def get_dataloader(args, graph_data):
    graph, labels, train_nid, val_nid, test_nid, node_feat = graph_data
    sampler = MultiLayerNeighborSampler(args.fanouts)
    test_dataloader = NodeDataLoader(graph, test_nid, sampler, batch_size=args.batch_size,
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


def test(args, device, graph_data):
    set_seed_logger(args)
    graph, labels, train_nid, val_nid, test_nid, node_feats = graph_data
        
    test_dataloader, node_feats, labels = get_dataloader(args, graph_data)

    in_feat = node_feats.shape[1]
    model = init_model(args, in_feat, device)
    print('  Model = %s', str(model))
    print('  Args = %s', str(print_hparams(args)))
 
    nodes_path = os.path.join(args.data_path, 'IDandLabels.csv')
    nodes_df = pd.read_csv(nodes_path, dtype={'Label':str})
    
    for epoch in range(args.test_epochs):
        result = test_epoch(model, test_dataloader, node_feats, labels, args.n_classes, device)
        df = pd.DataFrame({'node_idx': test_nid, 'label': result})
        
        for row in df.itertuples():
            node_idx = getattr(row, 'node_idx')
            label = getattr(row, 'label')
            labels[node_idx] = label
        
        df['label'] = df['label'].apply(id2name)
        mged = pd.merge(df, nodes_df[['node_idx', 'paper_id']], on='node_idx', how='left')
        result_csv = os.path.join(args.out_path, '{}_{}.csv'.format(args.gnn_model, epoch + 1))
        pd.DataFrame({'id': mged['paper_id'], 'label': mged['label']}).to_csv(result_csv, index=False)
        

if __name__ == '__main__':
    args = get_args()
    device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
    graph_data = load_dgl_graph(args.data_path, norm_feature=args.norm_feature)
        
    test(args, device, graph_data)
