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
import yaml
import dgl
from dgl.dataloading.neighbor import MultiLayerNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader
from torch import optim
from models import GraphSageModel, GraphConvModel, GraphAttnModel, GraphModel
from unimp import GNNModel
from utils import load_dgl_graph
from optimization import OptimAdam


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
    logging.basicConfig(level=logging.INFO,
                        format=
                        '%(asctime)s - %(levelname)s: %(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(dataset_cfg['OUT_PATH'], 'train.log')),
                            logging.StreamHandler(sys.stdout)])


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
    model = model.to(device)
    return model


def prep_optimizer(dataset_cfg, model, num_train_optimization_steps):
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
        {'params': [p for n, p in no_decay_graph_param_tp], 'weight_decay': dataset_cfg['WEIGHT_DECAY'], 'lr': dataset_cfg['LEARNING_RATE'] * dataset_cfg['COEF_LR']},
        {'params': [p for n, p in no_decay_nograph_param_tp], 'weight_decay': dataset_cfg['WEIGHT_DECAY']},
        {'params': [p for n, p in decay_graph_param_tp], 'weight_decay': 0.0, 'lr': dataset_cfg['LEARNING_RATE'] * dataset_cfg['COEF_LR']},
        {'params': [p for n, p in decay_nograph_param_tp], 'weight_decay': 0.0}
    ]

    optimizer = OptimAdam(optimizer_grouped_parameters, lr=dataset_cfg['LEARNING_RATE'], warmup=dataset_cfg['WARMUP_PROPORTION'],
                          schedule='warmup_linear', t_total=num_train_optimization_steps,
                          weight_decay=dataset_cfg['WEIGHT_DECAY'], max_grad_norm=1.0)

    return optimizer


def get_dataloader(dataset_cfg, graph_data):
    graph, labels, train_nid, val_nid, test_nid, node_feat = graph_data
    sampler = MultiLayerNeighborSampler(dataset_cfg['FANOUTS']) # 50

    train_dataloader = NodeDataLoader(graph, train_nid, sampler, batch_size=dataset_cfg['BATCH_SIZE'],
                                      shuffle=True, drop_last=False, num_workers=dataset_cfg['NUM_WORKERS'])
    val_dataloader = NodeDataLoader(graph, val_nid, sampler, batch_size=dataset_cfg['BATCH_SIZE'],
                                    shuffle=True, drop_last=False,  num_workers=dataset_cfg['NUM_WORKERS'])

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

        if dataset_cfg['GRADIENT_ACCUMULATION_STEPS'] > 1:
            train_loss = train_loss / dataset_cfg['GRADIENT_ACCUMULATION_STEPS']

        train_loss.backward()

        if (step + 1) % dataset_cfg['GRADIENT_ACCUMULATION_STEPS'] == 0:
            thnn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            global_step = global_step + 1
            #Lr: %s, "-".join([str('%.6f' % itm) for itm in sorted(list(set(optimizer.get_lr())))]),
            if global_step % log_step == 0:
                logging.info("Epoch: %d/%s, Step: %d/%d,  Loss: %f, Acc: %f, Time/step: %f", 
                             epoch + 1, dataset_cfg['EPOCHS'], step + 1, len(train_dataloader),
                             float(train_loss), float(tr_batch_pred.detach().mean()),
                             (time.time() - start_time) / (log_step * dataset_cfg['GRADIENT_ACCUMULATION_STEPS']))
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


def train(model_cfg, dataset_cfg, device, graph_data):
    set_seed_logger(dataset_cfg)
    train_dataloader, val_dataloader, node_feat, labels = get_dataloader(dataset_cfg, graph_data)
    num_train_optimization_steps = (int(len(train_dataloader) + dataset_cfg['GRADIENT_ACCUMULATION_STEPS'] - 1)
                                    / dataset_cfg['GRADIENT_ACCUMULATION_STEPS']) * dataset_cfg['EPOCHS']

    in_feat = node_feat.shape[1]
    model = init_model(model_cfg, in_feat, device)
    logging.info('  Model = %s', str(model))
    logging.info('  Training parameters = %d', sum(p.numel() for p in model.parameters() if p.requires_grad))
    logging.info('  Model parameters = %d', sum(p.numel() for p in model.parameters()))
    logging.info('  Num of train examples = %d', len(train_dataloader))
    logging.info('  Num of val examples = %d', len(val_dataloader))
    logging.info('  Num steps = %d', num_train_optimization_steps * dataset_cfg['GRADIENT_ACCUMULATION_STEPS'])

    # optimizer = prep_optimizer(dataset_cfg, model, num_train_optimization_steps)
    optimizer = optim.Adam(model.parameters(), lr=dataset_cfg['LEARNING_RATE'], weight_decay=dataset_cfg['WEIGHT_DECAY'])
    criterion = thnn.CrossEntropyLoss().to(device)

    output_folder = dataset_cfg['OUT_PATH']
    os.makedirs(output_folder, exist_ok=True)
    
    global_step, best_records = 0, [-1, 0.0]
    for epoch in range(dataset_cfg['EPOCHS']):
        tr_loss, tr_acc, global_step = train_epoch(epoch, model, train_dataloader, node_feat, labels, optimizer, criterion,
                                                   model_cfg['N_CLASS'], device, global_step, dataset_cfg['LOG_STEP'])
        logging.info("Train Epoch %d/%s Finished | Train Loss: %f | Train Acc: %f ", epoch + 1, dataset_cfg['EPOCHS'], tr_loss, tr_acc)

        val_loss, val_acc = val_epoch(model, val_dataloader, node_feat, labels, criterion, model_cfg['N_CLASS'], device)
        logging.info("Val Epoch %d/%s Finished | Val Loss: %f | Val Acc: %f ", epoch + 1, dataset_cfg['EPOCHS'], val_loss, val_acc)

        if val_acc > best_records[1]:
            best_records = [epoch + 1, val_acc]
            
        model_path = os.path.join(output_folder, 'se_fc_drop0.1_decay0.2_dgl_model_epoch{:02d}'.format(epoch + 1) + '_val_{:.4f}'.format(val_acc)+'.pth')
        th.save(model.state_dict(), model_path)
    
    logging.info("Best Epoch %d | Val Acc: %f ", best_records[0], best_records[1])


if __name__ == '__main__':
    yaml_path = 'config.yaml'
    f = open(yaml_path, 'r', encoding='utf-8')
    config = f.read()
    config = yaml.load(config)
    model_cfg = config['MODEL']
    dataset_cfg = config['DATASET']

    device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
    graph_data = load_dgl_graph(dataset_cfg['DATA_PATH'], norm_feature=dataset_cfg['NORM_FEATURE'])
    train(model_cfg, dataset_cfg, device, graph_data)
