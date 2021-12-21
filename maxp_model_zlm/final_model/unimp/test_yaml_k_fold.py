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

import dgl
from dgl.dataloading.neighbor import MultiLayerNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader

from unimp import UniMP
from util import load_dgl_graph


def set_seed_logger(dataset_cfg):
    random.seed(dataset_cfg['SEED'])
    os.environ['PYTHONHASHSEED'] = str(dataset_cfg['SEED'])
    np.random.seed(dataset_cfg['SEED'])
    th.manual_seed(dataset_cfg['SEED'])
    th.cuda.manual_seed(dataset_cfg['SEED'])
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True


def init_model(model_cfg, device):
    if model_cfg['GNN_MODEL'] == 'unimp':
        model = UniMP(
            input_size=model_cfg['INPUT_SIZE'],
            feat_size=model_cfg['FEAT_SIZE'],
            num_class=model_cfg['NUM_CLASS'],
            num_layers=model_cfg['NUM_LAYERS'],
            num_heads=model_cfg['NUM_HEADS'],
            hidden_size=model_cfg['HIDDEN_SIZE'],
            attn_heads=model_cfg['ATTN_HEADS'],
            label_drop=model_cfg['LABEL_DROP'],
            feat_drop=model_cfg['FEAT_DORP'],
            graph_drop=model_cfg['GRAPH_DORP'],
            attn_drop=model_cfg['ATTN_DROP'],
            drop=model_cfg['DROP'],
            use_sage=model_cfg['USE_SAGE'],
            use_conv=model_cfg['USE_CONV'],
            use_attn=model_cfg['USE_ATTN'],
            use_resnet=model_cfg['USE_RESNET'],
            use_densenet=model_cfg['USE_DESNET'],
            gatv2=model_cfg['GATV2'],
            se_mul=model_cfg['USE_SEMUL']
        )
    else:
        raise NotImplementedError('Not support algorithm: {}'.format(model_cfg['GNN_MODEL']))

    state = th.load(model_cfg['CHECKPOINT'], map_location='cpu')
    model.load_state_dict(state)
    print('Load checkpoint in', model_cfg['CHECKPOINT'])

    model = model.to(device)
    return model


def get_dataloader(dataset_cfg, graph_data):
    graph, labels, train_nid, val_nid, test_nid, node_feat = graph_data
    graph = dgl.add_self_loop(graph)
    sampler = MultiLayerNeighborSampler(dataset_cfg['FANOUTS'])
    test_dataloader = NodeDataLoader(
        graph, test_nid, sampler, batch_size=dataset_cfg['BATCH_SIZE'], shuffle=False, drop_last=False, num_workers=0
    )

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
            input_feats, input_labels, batch_labels = load_subtensor(
                node_feats, labels, seeds, input_nodes, n_classes, device
            )
            blocks = [block.to(device) for block in blocks]
            batch_logits = model(blocks, input_feats, input_labels)
            batch_logits = F.softmax(batch_logits, dim=-1)
            result.extend(batch_logits.detach().cpu().numpy())
    return result


def id2name(x):
    return chr(x + 65)


def test(model_cfg, dataset_cfg, device, graph_data):
    set_seed_logger(dataset_cfg)

    graph, labels, train_nid, val_nid, test_nid, node_feats = graph_data
    test_dataloader, node_feats, labels = get_dataloader(dataset_cfg, graph_data)

    k_fold_result = []

    for index, checkpoint_path in enumerate(model_cfg['CHECKPOINT_LIST']):
        checkpoint_path = os.path.join(model_cfg['CHECKPOINT_BASE'], checkpoint_path)
        print('Test checkpoint', checkpoint_path)

        model_cfg['CHECKPOINT'] = checkpoint_path
        model = init_model(model_cfg, device)
        print('Model config', str(dict(model_cfg)))
        print('Dataset config', str(dict(model_cfg)))

        result = test_epoch(model, test_dataloader, node_feats, labels, model_cfg['NUM_CLASS'], device)
        result = np.array(result)
        result_npy_path = os.path.join(
            dataset_cfg['OUT_PATH'], '{}_fold{}.npy'.format(dataset_cfg['TEST_PREFIX'], index)
        )
        np.save(result_npy_path, result)
        k_fold_result.append(result)

    # 保存每一折的结果
    k_fold_result = np.array(k_fold_result)
    result_npy_path = os.path.join(
        dataset_cfg['OUT_PATH'], '{}_{}_fold.npy'.format(dataset_cfg['TEST_PREFIX'], len(model_cfg['CHECKPOINT_LIST']))
    )
    np.save(result_npy_path, k_fold_result)

    # 保存id label
    # k_fold_result = np.mean(k_fold_result, axis=0)
    # result = np.argmax(k_fold_result, axis=-1)
    # df = pd.DataFrame({'node_idx': test_nid, 'label': result})

    # nodes_path = os.path.join('../final_dataset', 'IDandLabels.csv')
    # nodes_df = pd.read_csv(nodes_path, dtype={'Label': str})
    # df['label'] = df['label'].apply(id2name)
    # mged = pd.merge(df, nodes_df[['node_idx', 'paper_id']], on='node_idx', how='left')

    # result_csv = os.path.join(dataset_cfg['OUT_PATH'], '{}.csv'.format(dataset_cfg['TEST_PREFIX']))
    # pd.DataFrame({'id': mged['paper_id'], 'label': mged['label']}).to_csv(result_csv, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Node Classification')
    parser.add_argument('--cfg_file', type=str, default='config.yaml', help="Path of config files.")
    args = parser.parse_args()
    yaml_path = args.cfg_file

    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    model_cfg = config['MODEL']
    dataset_cfg = config['DATASET']
    dataset_cfg['K_FOLD'] = 1

    device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
    dataset_cfg['BATCH_SIZE'] = int(dataset_cfg['BATCH_SIZE'] / dataset_cfg['GRADIENT_ACCUMULATION_STEPS'])
    graph_data = load_dgl_graph(dataset_cfg)

    test(model_cfg, dataset_cfg, device, graph_data)
