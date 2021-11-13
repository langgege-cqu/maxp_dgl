import os
import sys
import time
import yaml
import random
import logging
import argparse
import numpy as np
import torch as th
from torch import optim
import torch.nn as thnn
import torch.nn.functional as F

import dgl
from dgl.dataloading.neighbor import MultiLayerNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader

from unicmp import UniCMP
from utils import load_dgl_graph
from optimization import OptimAdam
from loss import LabelSmoothingLoss


def set_seed_logger(dataset_cfg):
    random.seed(dataset_cfg['SEED'])
    os.environ['PYTHONHASHSEED'] = str(dataset_cfg['SEED'])
    np.random.seed(dataset_cfg['SEED'])
    th.manual_seed(dataset_cfg['SEED'])
    th.cuda.manual_seed(dataset_cfg['SEED'])
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True

    logging.basicConfig(level=logging.INFO,
                        format=
                        '%(asctime)s - %(levelname)s: %(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(dataset_cfg['OUT_PATH'], 'train.log')),
                            logging.StreamHandler(sys.stdout)])


def init_model(model_cfg, device):
    if model_cfg['GNN_MODEL'] == 'unicmp':
        model = UniCMP(input_size=model_cfg['INPUT_SIZE'], num_class=model_cfg['NUM_CLASS'], num_layers=model_cfg['NUM_LAYERS'],
                       num_heads=model_cfg['NUM_HEADS'], hidden_size=model_cfg['HIDDEN_SIZE'], label_drop=model_cfg['LABEL_DROP'],
                       feat_drop=model_cfg['FEAT_DORP'], attn_drop=model_cfg['ATTN_DROP'], drop=model_cfg['DROP'],
                       use_sage=model_cfg['USE_SAGE'], use_conv=model_cfg['USE_CONV'], use_attn=model_cfg['USE_ATTN'],
                       use_resnet=model_cfg['USE_RESNET'], use_densenet=model_cfg['USE_DESNET'])
    else:
        raise NotImplementedError('Not support three algorithms: {}'.format(model_cfg['GNN_MODEL']))
        
    if os.path.isfile(model_cfg['CHECKPOINT']):
        state = torch.load(model_cfg['CHECKPOINT'], map_location='cpu')
        model.load_state_dict(state)
        logging.info('Load checkpoint in: %s', args.checkpoint)

    model = model.to(device)
    return model


def prep_optimizer(dataset_cfg, model, num_train_optimization_steps):
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
    sampler = MultiLayerNeighborSampler(dataset_cfg['FANOUTS'])

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
            if global_step % log_step == 0:
                '''
                current_lr = [param_group['lr'] for param_group in optimizer.param_groups][0]
                logging.info('  Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Acc: %f, Time/step: %f',
                             epoch + 1, dataset_cfg['EPOCHS'], step + 1, len(train_dataloader),
                             current_lr, float(train_loss), float(tr_batch_pred.detach().mean()),
                             (time.time() - start_time) / (log_step * dataset_cfg['GRADIENT_ACCUMULATION_STEPS']))
                '''
                logging.info('  Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Acc: %f, Time/step: %f',
                             epoch + 1, dataset_cfg['EPOCHS'], step + 1, len(train_dataloader),
                             '-'.join([str('%.6f' % itm) for itm in sorted(list(set(optimizer.get_lr())))]),
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
    output_folder = os.path.join(dataset_cfg['OUT_PATH'], 'models')
    os.makedirs(output_folder, exist_ok=True)

    train_dataloader, val_dataloader, node_feat, labels = get_dataloader(dataset_cfg, graph_data)
    num_train_optimization_steps = (int(len(train_dataloader) + dataset_cfg['GRADIENT_ACCUMULATION_STEPS'] - 1)
                                    / dataset_cfg['GRADIENT_ACCUMULATION_STEPS']) * dataset_cfg['EPOCHS']

    model = init_model(model_cfg, device)
    logging.info('Model = %s', str(model))
    logging.info('Model config = %s', str(dict(model_cfg)))
    logging.info('Dataset config = %s', str(dict(dataset_cfg)))
    logging.info('Training parameters = %d', sum(p.numel() for p in model.parameters() if p.requires_grad))
    logging.info('Model parameters = %d', sum(p.numel() for p in model.parameters()))
    logging.info('Num of train examples = %d', len(train_dataloader))
    logging.info('Num of val examples = %d', len(val_dataloader))
    logging.info('Num steps = %d', num_train_optimization_steps * dataset_cfg['GRADIENT_ACCUMULATION_STEPS'])

    optimizer = prep_optimizer(dataset_cfg, model, num_train_optimization_steps)
    # optimizer = optim.Adam(model.parameters(), lr=dataset_cfg['LEARNING_RATE'], weight_decay=dataset_cfg['WEIGHT_DECAY'])
    
    if model_cfg['SMOOTHING'] > 0:
        criterion = LabelSmoothingLoss(model_cfg['NUM_CLASS'], smoothing=model_cfg['SMOOTHING'])
    else:
        criterion = thnn.CrossEntropyLoss()
    criterion = criterion.to(device)
    
    global_step, best_records = 0, [-1, 0.0]
    for epoch in range(dataset_cfg['EPOCHS']):
        tr_loss, tr_acc, global_step = train_epoch(epoch, model, train_dataloader, node_feat, labels, optimizer, criterion,
                                                   model_cfg['NUM_CLASS'], device, global_step, dataset_cfg['LOG_STEP'])
        logging.info('Train Epoch %d/%s Finished | Train Loss: %f | Train Acc: %f', epoch + 1, dataset_cfg['EPOCHS'], tr_loss, tr_acc)

        val_loss, val_acc = val_epoch(model, val_dataloader, node_feat, labels, criterion, model_cfg['NUM_CLASS'], device)
        logging.info('Val Epoch %d/%s Finished | Val Loss: %f | Val Acc: %f', epoch + 1, dataset_cfg['EPOCHS'], val_loss, val_acc)

        if val_acc > best_records[1]:
            best_records = [epoch + 1, val_acc]
            
        model_path = os.path.join(output_folder, '{}_epoch{:02d}_val{:04d}.pth'.format(dataset_cfg['MODEL_PREFIX'], epoch + 1, int(val_acc * 10000)))
        th.save(model.state_dict(), model_path)
    
    logging.info('Best Epoch %d | Val Acc: %f ', best_records[0], best_records[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Node Classification')
    parser.add_argument('--cfg_file', type=str, default='config.yaml', help="Path of config files.")
    args = parser.parse_args()
    yaml_path = args.cfg_file
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    model_cfg = config['MODEL']
    dataset_cfg = config['DATASET']
    
    output_folder = dataset_cfg['OUT_PATH']
    os.makedirs(output_folder, exist_ok=True)
    os.system('cp {} {}/config.yaml'.format(yaml_path, output_folder))
 
    device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
    dataset_cfg['BATCH_SIZE'] = int(dataset_cfg['BATCH_SIZE'] / dataset_cfg['GRADIENT_ACCUMULATION_STEPS'])
    graph_data = load_dgl_graph(dataset_cfg['DATA_PATH'], norm_feature=dataset_cfg['NORM_FEATURE'])
    
    train(model_cfg, dataset_cfg, device, graph_data)
