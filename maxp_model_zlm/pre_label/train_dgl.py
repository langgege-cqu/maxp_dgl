import os
import sys
import time
import random
import inspect
import logging
import argparse
import numpy as np
import torch as th
import torch.nn as thnn
import torch.nn.functional as F

import dgl
from dgl.dataloading.neighbor import MultiLayerNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader

from models import GraphSageModel, GraphConvModel, GraphAttnModel, GraphModel
from utils import load_dgl_graph
from optimization import OptimAdam


def print_hparams(hp):
    attributes = inspect.getmembers(hp, lambda a: not (inspect.isroutine(a)))
    return dict([a for a in attributes if not (a[0].startswith('__') and a[0].endswith('__'))])


def get_args(description='Graph Node Classification'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_path', type=str, help="Path of saved processed data files.", default='data/')
    parser.add_argument('--gnn_model', type=str, choices=['graphsage', 'graphconv', 'graphattn', 'graphmodel'],
                        required=True, default='graphsage')
    parser.add_argument('--norm_feature', action='store_true')
    
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--mlp_dim', type=int, default=256)
    parser.add_argument('--num_attention_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--n_classes', type=int, default=23)
    parser.add_argument("--fanouts", type=str, help="fanout numbers", default='50,50,50')
    parser.add_argument("--input_drop", type=float, default=0.3)
    parser.add_argument("--drop_rate", type=float, default=0.5)
    parser.add_argument("--attn_drop", type=float, default=0.2)

    parser.add_argument('--batch_size', type=int, required=True, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0008, help='initial learning rate')
    parser.add_argument('--coef_lr', type=float, default=0.8)
    parser.add_argument('--weight_decay', type=float, default=0.005, help='weight decay')
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--log_step', type=int, default=200)
    parser.add_argument('--out_path', type=str, required=True, help="Absolute path for saving model parameters")
    
    parser.add_argument('--checkpoint', type=str, default='')
    args = parser.parse_args()

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)
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
    logging.basicConfig(level=logging.INFO,
                        format=
                        '%(asctime)s - %(levelname)s: %(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(args.out_path, 'train.log')),
                            logging.StreamHandler(sys.stdout)])


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
        '''
        checkpoint = th.load('graphmodel/split06/dgl_model_epoch.pth', map_location='cpu')
        checkpoint1 = th.load('graphmodel/split06_sage/dgl_model_epoch.pth', map_location='cpu')
        checkpoint2 = th.load('graphmodel/split06_conv/dgl_model_epoch.pth', map_location='cpu')
        checkpoint3 = th.load('graphmodel/split06_attn/dgl_model_epoch.pth', map_location='cpu')

        state = model.state_dict()
        for k in state.keys():
            if 'graphsage' in k:
                state[k] = checkpoint1[k]
            elif 'graphconv' in k:
                state[k] = checkpoint2[k]
            elif 'graphattn' in k:
                state[k] = checkpoint3[k]
            else:
                state[k] = checkpoint[k]
                
        model.load_state_dict(state)
        '''
    else:
        raise NotImplementedError('So far, only support three algorithms: GraphSage, GraphConv, and GraphAttn')
    '''
    if args.gnn_model in ['graphsage', 'graphconv', 'graphattn']:
        checkpoint = th.load(args.checkpoint, map_location='cpu')
        state = model.state_dict()
        for k in state.keys():
            state[k] = checkpoint[k]
        model.load_state_dict(state)
        for p in model.label_embed.parameters():
            p.requires_grad = False
        for p in model.input_drop.parameters():
            p.requires_grad = False
        for p in model.feature_mlp.parameters():
            p.requires_grad = False
    '''
    model = model.to(device)
    return model


def prep_optimizer(args, model, num_train_optimization_steps):
    param_optimizer = list(filter(lambda p: p[1].requires_grad, model.named_parameters()))
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    no_decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    def check_dgl_module(name):
        dgl_name = ['graphconv.', 'graphsage.', 'graphattn.']
        for m in dgl_name:
            if m in name:
                return True

        return False

    no_decay_graph_param_tp = [(n, p) for n, p in no_decay_param_tp if check_dgl_module(n)]
    no_decay_nograph_param_tp = [(n, p) for n, p in no_decay_param_tp if not check_dgl_module(n)]

    decay_graph_param_tp = [(n, p) for n, p in decay_param_tp if check_dgl_module(n)]
    decay_nograph_param_tp = [(n, p) for n, p in decay_param_tp if not check_dgl_module(n)]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in no_decay_graph_param_tp], 'weight_decay': args.weight_decay, 'lr': args.lr * args.coef_lr},
        {'params': [p for n, p in no_decay_nograph_param_tp], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in decay_graph_param_tp], 'weight_decay': 0.0, 'lr': args.lr * args.coef_lr},
        {'params': [p for n, p in decay_nograph_param_tp], 'weight_decay': 0.0}
    ]

    optimizer = OptimAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                          schedule='warmup_linear', t_total=num_train_optimization_steps,
                          weight_decay=args.weight_decay, max_grad_norm=1.0)

    return optimizer


def get_dataloader(args, graph_data):
    graph, labels, train_nid, val_nid, test_nid, node_feat = graph_data
    sampler = MultiLayerNeighborSampler(args.fanouts) # 50

    train_dataloader = NodeDataLoader(graph, train_nid, sampler, batch_size=args.batch_size,
                                      shuffle=True, drop_last=False, num_workers=args.num_workers)
    val_dataloader = NodeDataLoader(graph, val_nid, sampler, batch_size=args.batch_size,
                                    shuffle=True, drop_last=False,  num_workers=args.num_workers)

    return train_dataloader, val_dataloader, node_feat, labels


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


def train_epoch(epoch, model, train_dataloader, node_feat, labels, optimizer, criterion, n_classes, device, global_step, log_step):
    th.cuda.empty_cache()
    model.train()

    start_time = time.time()
    train_loss_list, train_acc_list = [], []
    for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
        # forward
        input_feats, input_labels, batch_labels = load_subtensor(node_feat, labels, seeds, input_nodes,
                                                                 n_classes, device, training=True)
        blocks = [block.to(device) for block in blocks]
        # metric and loss
        train_batch_logits = model(blocks, input_feats, input_labels)
        train_loss = criterion(train_batch_logits, batch_labels)
        train_loss_list.append(train_loss.cpu().detach().numpy())
        tr_batch_pred = th.sum((th.argmax(train_batch_logits, dim=1) == batch_labels).type(th.int)) / \
                        th.tensor(batch_labels.shape[0])
        train_acc_list.append(float(tr_batch_pred.detach().mean()))

        if args.gradient_accumulation_steps > 1:
            train_loss = train_loss / args.gradient_accumulation_steps

        train_loss.backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            thnn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            global_step = global_step + 1
            if global_step % log_step == 0:
                logging.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Acc: %f, Time/step: %f", 
                             epoch + 1, args.epochs, step + 1, len(train_dataloader),
                             "-".join([str('%.6f' % itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                             float(train_loss), float(tr_batch_pred.detach().mean()),
                             (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

    total_loss = sum(train_loss_list) / len(train_loss_list)
    total_acc = sum(train_acc_list) / len(train_acc_list)
    return total_loss, total_acc, global_step


def val_epoch(model, val_dataloader, node_feat, labels, criterion, n_classes, device):
    th.cuda.empty_cache()
    model.eval()

    val_loss_list, val_acc_list = [], []
    with th.no_grad():
        for step, (input_nodes, seeds, blocks) in enumerate(val_dataloader):
            input_feats, input_labels, batch_labels = load_subtensor(node_feat, labels, seeds, input_nodes,
                                                                     n_classes, device, training=False)
            blocks = [block.to(device) for block in blocks]
 
            val_batch_logits = model(blocks, input_feats, input_labels)
   
            val_loss = criterion(val_batch_logits, batch_labels)
            val_loss_list.append(val_loss.detach().cpu().numpy())
            val_batch_pred = th.sum((th.argmax(val_batch_logits, dim=1) == batch_labels).type(th.int)) /\
                             th.tensor(batch_labels.shape[0])
            val_acc_list.append(float(val_batch_pred.detach().mean()))

    total_loss = sum(val_loss_list) / len(val_loss_list)
    total_acc = sum(val_acc_list) / len(val_acc_list)
    return total_loss, total_acc


def train(args, device, graph_data):
    set_seed_logger(args)
    train_dataloader, val_dataloader, node_feat, labels = get_dataloader(args, graph_data)
    num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                    / args.gradient_accumulation_steps) * args.epochs

    in_feat = node_feat.shape[1]
    model = init_model(args, in_feat, device)
    logging.info('  Model = %s', str(model))
    logging.info('  Args = %s', str(print_hparams(args)))
    logging.info('  Training parameters = %d', sum(p.numel() for p in model.parameters() if p.requires_grad))
    logging.info('  Model parameters = %d', sum(p.numel() for p in model.parameters()))
    logging.info('  Num of train examples = %d', len(train_dataloader))
    logging.info('  Num of val examples = %d', len(val_dataloader))
    logging.info('  Num steps = %d', num_train_optimization_steps * args.gradient_accumulation_steps)

    optimizer = prep_optimizer(args, model, num_train_optimization_steps)
    criterion = thnn.CrossEntropyLoss().to(device)

    output_folder = os.path.join(args.out_path, 'models')
    os.makedirs(output_folder, exist_ok=True)
    
    global_step, best_records = 0, [-1, 0.0]
    for epoch in range(args.epochs):
        tr_loss, tr_acc, global_step = train_epoch(epoch, model, train_dataloader, node_feat, labels, optimizer, criterion,
                                                   args.n_classes, device, global_step, args.log_step)
        logging.info("Train Epoch %d/%s Finished | Train Loss: %f | Train Acc: %f ", epoch + 1, args.epochs, tr_loss, tr_acc)

        val_loss, val_acc = val_epoch(model, val_dataloader, node_feat, labels, criterion, args.n_classes, device)
        logging.info("Val Epoch %d/%s Finished | Val Loss: %f | Val Acc: %f ", epoch + 1, args.epochs, val_loss, val_acc)

        if val_acc > best_records[1]:
            best_records = [epoch + 1, val_acc]
            
        model_path = os.path.join(output_folder, 'dgl_model_epoch{:02d}'.format(epoch + 1) + '.pth')
        th.save(model.state_dict(), model_path)
    
    logging.info("Best Epoch %d | Val Acc: %f ", best_records[0], best_records[1])

if __name__ == '__main__':
    args = get_args()
    device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
    graph_data = load_dgl_graph(args.data_path, norm_feature=args.norm_feature)
    train(args, device, graph_data)
