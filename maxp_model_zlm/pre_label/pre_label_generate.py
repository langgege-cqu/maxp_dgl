# %%
# 引入库
import dgl
import dgl.nn as dglnn
import logging
from dgl.dataloading.neighbor import MultiLayerNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader
from numpy.lib.function_base import select
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
import datetime
import numpy as np

import pandas as pd
import os
import gc
from shutil import copyfile

from yaml import load, Loader
from utils import load_dgl_graph, set_seed_logger

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)


class PreLable(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_class):
        super(PreLable, self).__init__()

        self.f = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_class),
        )

    def forward(self, x):
        x = self.f(x)
        return x


class PreLableDataSet(Dataset):
    def __init__(self, features) -> None:

        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


# %%
def get_dataloader(dataset_cfg, graph_data):
    graph, labels, train_nid, val_nid, test_nid, node_feat = graph_data

    dataset = PreLableDataSet(node_feat)

    tdataloader = DataLoader(dataset,
                             batch_size=dataset_cfg['BATCH_SIZE'],
                             shuffle=False,
                             drop_last=False,
                             num_workers=dataset_cfg['NUM_WORKERS'])

    del graph, labels, train_nid, val_nid, test_nid
    gc.collect()

    return tdataloader


def test_epoch(model, test_dataloader, device):
    torch.cuda.empty_cache()
    model.eval()

    result = []
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    torch.cuda.empty_cache()
    with torch.no_grad():
        for step, batch_data in pbar:
            features = batch_data
            features = features.to(device)
            batch_logits = model(features)
            batch_results = torch.softmax(batch_logits, dim=-1)
            result.extend(batch_results.detach().cpu().numpy().tolist())
    return result


# %%
# train stage one
def test(model_cfg, dataset_cfg, train_cfg, device, graph_data, k_fold):
    # 获得数据
    dataloader = get_dataloader(dataset_cfg, graph_data)

    model = PreLable(in_dim=300, hidden_dim=128, n_class=23)
    checkpoint = torch.load(model_cfg['CHECKPOINT'])
    model.load_state_dict(checkpoint)

    model = model.to(device)

    logging.info('  Model = %s', str(model))
    logging.info('  Test parameters = %d',
                 sum(p.numel() for p in model.parameters() if p.requires_grad))
    logging.info('  Model parameters = %d', sum(p.numel() for p in model.parameters()))
    logging.info('  Num of test examples = %d', len(dataloader))

    result = test_epoch(model, dataloader, device)
    np.save(os.path.join(dataset_cfg['DATA_PATH'], 'pre_label_probs.npy'), result)


# %%
if __name__ == '__main__':
    config_path = 'config.yaml'
    f = open(config_path, 'r', encoding='utf-8')
    config = f.read()
    config = load(config, Loader=Loader)
    dataset_cfg = config['DATASET']
    model_cfg = config['MODEL']
    train_cfg = config['TRAIN']

    set_seed_logger(train_cfg)

    # 加载图
    k_fold = 2
    graph_data = load_dgl_graph(dataset_cfg['DATA_PATH'],
                                k_fold,
                                norm_feature=dataset_cfg['NORM_FEATURE'])

    # 训练
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    test(model_cfg, dataset_cfg, train_cfg, device, graph_data, k_fold)
