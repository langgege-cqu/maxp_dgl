import os
import sys
import math
import time
import yaml
import random
import logging
import argparse
import numpy as np
import torch as th
import torch.nn as thnn

import dgl
from dgl.dataloading.neighbor import MultiLayerNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader
from dataloader import InOutDataLaoder

from dunicmp import DouUniCMP
from util import load_dgl_graph
from optimization import OptimAdam
from loss import LabelSmoothingLoss, AsymmetricLoss


def set_seed_logger(dataset_cfg):
    random.seed(dataset_cfg['SEED'])
    os.environ['PYTHONHASHSEED'] = str(dataset_cfg['SEED'])
    np.random.seed(dataset_cfg['SEED'])
    th.manual_seed(dataset_cfg['SEED'])
    th.cuda.manual_seed(dataset_cfg['SEED'])
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(dataset_cfg['OUT_PATH'], 'train.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )


def init_model(model_cfg, device):
    if model_cfg['GNN_MODEL'] == 'unicmp':
        model = DouUniCMP(
            input_size=model_cfg['INPUT_SIZE'],
            num_class=model_cfg['NUM_CLASS'],
            num_layers=model_cfg['NUM_LAYERS'],
            num_heads=model_cfg['NUM_HEADS'],
            hidden_size=model_cfg['HIDDEN_SIZE'],
            label_drop=model_cfg['LABEL_DROP'],
            feat_drop=model_cfg['FEAT_DORP'],
            graph_drop=model_cfg['GRAPH_DORP'],
            attn_drop=model_cfg['ATTN_DROP'],
            drop=model_cfg['DROP'],
            use_sage=model_cfg['USE_SAGE'],
            use_conv=model_cfg['USE_CONV'],
            use_attn=model_cfg['USE_ATTN'],
            use_resnet=model_cfg['USE_RESNET'],
            use_densenet=model_cfg['USE_DESNET']
        )
    else:
        raise NotImplementedError('Not support algorithm: {}'.format(model_cfg['GNN_MODEL']))

    if os.path.isfile(model_cfg['CHECKPOINT']):
        state = th.load(model_cfg['CHECKPOINT'], map_location='cpu')
        model.load_state_dict(state)
        logging.info('Load checkpoint in: %s', args.checkpoint)

    model = model.to(device)
    return model


def prep_optimizer(optimizer_cfg, model, num_train_optimization_steps):
    param_optimizer = list(filter(lambda p: p[1].requires_grad, model.named_parameters()))
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    no_decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    def check_dgl_module(name):
        dgl_name = ['graphconv', 'graphsage', 'graphattn']
        for m in dgl_name:
            if m in name:
                return True
        return False

    no_decay_graph_param_tp = [(n, p) for n, p in no_decay_param_tp if check_dgl_module(n)]
    no_decay_nograph_param_tp = [(n, p) for n, p in no_decay_param_tp if not check_dgl_module(n)]

    decay_graph_param_tp = [(n, p) for n, p in decay_param_tp if check_dgl_module(n)]
    decay_nograph_param_tp = [(n, p) for n, p in decay_param_tp if not check_dgl_module(n)]

    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in no_decay_graph_param_tp],
            'weight_decay': optimizer_cfg['WEIGHT_DECAY'],
            'lr': optimizer_cfg['LEARNING_RATE'] * optimizer_cfg['COEF_LR']
        }, {
            'params': [p for n, p in no_decay_nograph_param_tp],
            'weight_decay': optimizer_cfg['WEIGHT_DECAY']
        }, {
            'params': [p for n, p in decay_graph_param_tp],
            'weight_decay': 0.0,
            'lr': optimizer_cfg['LEARNING_RATE'] * optimizer_cfg['COEF_LR']
        }, {
            'params': [p for n, p in decay_nograph_param_tp],
            'weight_decay': 0.0
        }
    ]

    optimizer = OptimAdam(
        optimizer_grouped_parameters,
        lr=optimizer_cfg['LEARNING_RATE'],
        warmup=optimizer_cfg['WARMUP_PROPORTION'],
        schedule=optimizer_cfg['SCHEDULE'],
        t_total=num_train_optimization_steps,
        weight_decay=optimizer_cfg['WEIGHT_DECAY'],
        max_grad_norm=1.0
    )

    return optimizer


def prep_criterion(criterion_cfg, device):
    if criterion_cfg['LOSS_TYPE'] == 'ASL':
        criterion = AsymmetricLoss(
            criterion_cfg['NUM_CLASS'], gamma_neg=criterion_cfg['GAMMA_NEG'], gamma_pos=criterion_cfg['GAMMA_POS']
        )
    elif criterion_cfg['LOSS_TYPE'] == 'SCL':
        criterion = LabelSmoothingLoss(criterion_cfg['NUM_CLASS'], smoothing=criterion_cfg['SMOOTHING'])
    elif criterion_cfg['LOSS_TYPE'] == 'CL':
        criterion = thnn.CrossEntropyLoss()
    else:
        raise NotImplementedError('Not support algorithm: {}'.format(criterion_cfg['GNN_MODEL']))

    criterion = criterion.to(device)

    return criterion


def get_dataloader(dataset_cfg, graph, nid, drop=False):
    src, dst = graph.all_edges()

    if drop:
        mask = th.zeros_like(src).bernoulli_(dataset_cfg['EDGE_DROP']) == 0
        src = src[mask]
        dst = dst[mask]

    sample_graph = dgl.graph((src, dst), num_nodes=graph.number_of_nodes())
    sample_graph = dgl.add_self_loop(sample_graph)

    dataloader = InOutDataLaoder(
        g=sample_graph,
        nid=nid,
        in_fanout=dataset_cfg['IN_FANOUTS'],
        out_fanout=dataset_cfg['OUT_FANOUTS'],
        batch_size=dataset_cfg['BATCH_SIZE'],
        shuffle=True
    )
    return dataloader


def load_subtensor(dataset_cfg, node_feats, labels, seeds, input_nodes, device, mask=False):
    input_feats = node_feats[input_nodes].to(device)

    masklabels = labels.clone()
    masklabels[seeds] = -1
    input_labels = masklabels[input_nodes]

    if mask:
        rd_m = th.rand(input_labels.shape[0])
        rd_y = th.randint(0, dataset_cfg['NUM_CLASS'], size=input_labels.shape)
        input_labels[rd_m < dataset_cfg['MASK_LABEL']] = -1
        input_labels[rd_m > 1 - dataset_cfg['REPLACE_LABEL']] = rd_y[rd_m > 1 - dataset_cfg['REPLACE_LABEL']]

    input_labels[input_labels < 0] = dataset_cfg['NUM_CLASS']
    input_labels = input_labels.to(device)

    batch_labels = labels[seeds].to(device)
    return input_feats, input_labels, batch_labels


def train_epoch(
    epoch, model, train_dataloader, dataset_cfg, node_feats, labels, optimizer, criterion, device, global_step
):
    th.cuda.empty_cache()
    model.train()

    start_time = time.time()
    train_loss_list, train_acc_list = [], []
    for step, (in_input_nodes, in_blocks, out_input_nodes, out_blocks, seeds) in enumerate(train_dataloader):
        in_input_feats, in_input_labels, batch_labels = load_subtensor(
            dataset_cfg, node_feats, labels, seeds, in_input_nodes, device, mask=True
        )

        out_input_feats, out_input_labels, batch_labels = load_subtensor(
            dataset_cfg, node_feats, labels, seeds, out_input_nodes, device, mask=True
        )

        in_blocks = [block.to(device) for block in in_blocks]

        out_blocks = [block.to(device) for block in out_blocks]

        train_batch_logits = model(
            in_blocks, in_input_feats, in_input_labels, out_blocks, out_input_feats, out_input_labels
        )
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
            if global_step % dataset_cfg['LOG_STEP'] == 0:
                logging.info(
                    '  Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Acc: %f, Time/step: %f', epoch + 1,
                    dataset_cfg['EPOCHS'], step + 1, len(train_dataloader),
                    '-'.join([str('%.6f' % itm) for itm in sorted(list(set(optimizer.get_lr())))]), float(train_loss),
                    float(tr_batch_pred.detach().mean()),
                    (time.time() - start_time) / (dataset_cfg['LOG_STEP'] * dataset_cfg['GRADIENT_ACCUMULATION_STEPS'])
                )
                start_time = time.time()

    total_loss = sum(train_loss_list) / len(train_loss_list)
    total_acc = sum(train_acc_list) / len(train_acc_list)
    return total_loss, total_acc, global_step


def val_epoch(model, val_dataloader, dataset_cfg, node_feats, labels, criterion, device):
    th.cuda.empty_cache()
    model.eval()

    val_loss_list, val_acc_list = [], []
    with th.no_grad():
        for step, (in_input_nodes, in_blocks, out_input_nodes, out_blocks, seeds) in enumerate(val_dataloader):
            in_input_feats, in_input_labels, batch_labels = load_subtensor(
                dataset_cfg, node_feats, labels, seeds, in_input_nodes, device, mask=False
            )

            out_input_feats, out_input_labels, batch_labels = load_subtensor(
                dataset_cfg, node_feats, labels, seeds, out_input_nodes, device, mask=False
            )

            in_blocks = [block.to(device) for block in in_blocks]

            out_blocks = [block.to(device) for block in out_blocks]

            val_batch_logits = model(
                in_blocks, in_input_feats, in_input_labels, out_blocks, out_input_feats, out_input_labels
            )
            val_loss = criterion(val_batch_logits, batch_labels)
            val_loss_list.append(val_loss.detach().cpu().numpy())
            val_batch_pred = th.sum((th.argmax(val_batch_logits, dim=1) == batch_labels).type(th.int)) / \
                             th.tensor(batch_labels.shape[0])
            val_acc_list.append(float(val_batch_pred.detach().mean()))

    total_loss = sum(val_loss_list) / (len(val_loss_list))
    total_acc = sum(val_acc_list) / (len(val_acc_list))
    return total_loss, total_acc


def train(model_cfg, dataset_cfg, optimizer_cfg, criterion_cfg, device, graph_data):
    set_seed_logger(dataset_cfg)
    output_folder = os.path.join(dataset_cfg['OUT_PATH'], 'models')
    os.makedirs(output_folder, exist_ok=True)

    graph, labels, train_nid, val_nid, test_nid, node_feats = graph_data
    graph = dgl.remove_self_loop(graph)

    val_dataloader = get_dataloader(dataset_cfg, graph, val_nid, drop=False)

    train_num = math.ceil(len(train_nid) / dataset_cfg['BATCH_SIZE'])
    num_train_optimization_steps = (
        int(train_num + dataset_cfg['GRADIENT_ACCUMULATION_STEPS'] - 1) / dataset_cfg['GRADIENT_ACCUMULATION_STEPS']
    ) * dataset_cfg['EPOCHS']

    model = init_model(model_cfg, device)
    logging.info('Model = %s', str(model))
    logging.info('Model config = %s', str(dict(model_cfg)))
    logging.info('Dataset config = %s', str(dict(dataset_cfg)))
    logging.info('Optimizer config = %s', str(dict(optimizer_cfg)))
    logging.info('Criterion config = %s', str(dict(criterion_cfg)))
    logging.info('Training parameters = %d', sum(p.numel() for p in model.parameters() if p.requires_grad))
    logging.info('Model parameters = %d', sum(p.numel() for p in model.parameters()))
    logging.info('Num steps = %d', num_train_optimization_steps * dataset_cfg['GRADIENT_ACCUMULATION_STEPS'])

    optimizer = prep_optimizer(optimizer_cfg, model, num_train_optimization_steps)
    criterion = prep_criterion(criterion_cfg, device)

    global_step = 0
    best_record = {'epoch': -1, 'train loss': -1, 'train acc': 0.0, 'val loss': -1, 'val acc': 0.0}

    for epoch in range(dataset_cfg['EPOCHS']):
        train_dataloader = get_dataloader(dataset_cfg, graph, train_nid, drop=True)

        tr_loss, tr_acc, global_step = train_epoch(
            epoch, model, train_dataloader, dataset_cfg, node_feats, labels, optimizer, criterion, device, global_step
        )
        logging.info(
            'Train Epoch %d/%s Finished | Train Loss: %f | Train Acc: %f', epoch + 1, dataset_cfg['EPOCHS'], tr_loss,
            tr_acc
        )

        val_loss, val_acc = val_epoch(model, val_dataloader, dataset_cfg, node_feats, labels, criterion, device)
        logging.info(
            'Val Epoch %d/%s Finished | Val Loss: %f | Val Acc: %f', epoch + 1, dataset_cfg['EPOCHS'], val_loss, val_acc
        )

        model_path = os.path.join(
            output_folder,
            '{}_epoch{:02d}_val{:04d}.pth'.format(dataset_cfg['MODEL_PREFIX'], epoch + 1, int(val_acc * 10000))
        )

        if val_acc > best_record['val acc']:
            best_record = {
                'epoch': epoch + 1,
                'train loss': tr_loss,
                'train acc': tr_acc,
                'val loss': val_loss,
                'val acc': val_acc
            }
            th.save(model.state_dict(), model_path)

    logging.info(
        'Best Epoch %d | Train Loss: %f | Train Acc: %f | Val Loss: %f | Val Acc: %f ', best_record['epoch'],
        best_record['train loss'], best_record['train acc'], best_record['val loss'], best_record['val acc']
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Node Classification')
    parser.add_argument('--cfg_file', type=str, default='config.yaml', help="Path of config files.")
    args = parser.parse_args()
    yaml_path = args.cfg_file

    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    model_cfg = config['MODEL']
    dataset_cfg = config['DATASET']
    optimizer_cfg = config['OPTIMIZER']
    criterion_cfg = config['CRITERION']

    output_folder = dataset_cfg['OUT_PATH']
    os.makedirs(output_folder, exist_ok=True)
    os.system('cp {} {}/config.yaml'.format(yaml_path, output_folder))

    device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
    dataset_cfg['BATCH_SIZE'] = int(dataset_cfg['BATCH_SIZE'] / dataset_cfg['GRADIENT_ACCUMULATION_STEPS'])
    k_fold = dataset_cfg['K_FOLD']
    graph_data = load_dgl_graph(dataset_cfg['DATA_PATH'], k=k_fold, to_bidirected=False)

    train(model_cfg, dataset_cfg, optimizer_cfg, criterion_cfg, device, graph_data)
