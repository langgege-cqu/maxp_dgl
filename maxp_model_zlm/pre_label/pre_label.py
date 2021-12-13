# %%
# 引入库
import dgl
import dgl.nn as dglnn
import logging
from dgl.dataloading.neighbor import MultiLayerNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader
from numpy.lib.function_base import select

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
import datetime

import time
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
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Sigmoid(),
            nn.Linear(hidden_dim // 2, n_class),
        )

    def forward(self, x):
        x = self.f(x)
        return x


class PreLableDataSet(Dataset):
    def __init__(self, ids, features, labels) -> None:
        # 只选取有标签的
        ids_labels = labels[ids]
        select_ids = ids[ids_labels != -1]

        self.ids = select_ids
        self.features = features[select_ids]
        self.labels = labels[select_ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.ids[index], self.features[index], self.labels[index]


# %%
def get_dataloader(dataset_cfg, graph_data):
    graph, labels, train_nid, val_nid, test_nid, node_feat = graph_data

    train_dataset = PreLableDataSet(train_nid, node_feat, labels)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=dataset_cfg['BATCH_SIZE'],
                                  shuffle=True,
                                  drop_last=False,
                                  num_workers=dataset_cfg['NUM_WORKERS'])

    val_dataset = PreLableDataSet(val_nid, node_feat, labels)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=dataset_cfg['BATCH_SIZE'],
                                shuffle=False,
                                drop_last=False,
                                num_workers=dataset_cfg['NUM_WORKERS'])

    # test_dataset = PreLableDataSet(test_nid, node_feat, labels)

    # test_dataloader = DataLoader(test_dataset,
    #                              batch_size=dataset_cfg['BATCH_SIZE'],
    #                              shuffle=False,
    #                              drop_last=False,
    #                              num_workers=dataset_cfg['NUM_WORKERS'])

    del graph, labels, train_nid, val_nid, test_nid, node_feat
    gc.collect()

    return train_dataloader, val_dataloader


# %%
def train_epoch(epoch, train_cfg, model, train_dataloader, optimizer, criterion, global_step,
                log_step, device):
    torch.cuda.empty_cache()
    model.train()
    train_loss_list, train_acc_list = [], []

    start_time = time.time()
    for step, batch_data in enumerate(train_dataloader):

        ids, features, labels = batch_data
        features, labels = features.to(device), labels.to(device)
        predict = model(features)

        train_batch_pred = torch.sum((torch.argmax(predict, dim=1) == labels).type(
            torch.int)) / torch.tensor(labels.shape[0])
        train_acc_list.append(float(train_batch_pred.detach().mean()))

        loss = criterion(predict, labels)

        train_loss_list.append(loss.cpu().detach().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step = global_step + 1

        if global_step % log_step == 0:
            logging.info("Epoch: %d/%s, Step: %d/%d, Loss: %f, Acc: %f, Time/step: %f", epoch + 1,
                         train_cfg['EPOCHS'], step + 1, len(train_dataloader),
                         float(train_loss_list[-1]),
                         train_batch_pred.detach().mean(), (time.time() - start_time))

            writer.add_scalar('Train/loss', loss.cpu().detach().numpy(), global_step)
            writer.add_scalar('Train/acc', train_batch_pred.detach().mean(), global_step)
            start_time = time.time()

    total_loss = sum(train_loss_list) / len(train_loss_list)
    total_acc = sum(train_acc_list) / len(train_acc_list)
    return total_loss, total_acc, global_step


# %%
def val_epoch(model, val_dataloader, criterion, device):
    torch.cuda.empty_cache()
    model.eval()

    val_loss_list, val_acc_list = [], []

    with torch.no_grad():
        for _, batch_data in enumerate(val_dataloader):

            ids, features, labels = batch_data
            features, labels = features.to(device), labels.to(device)
            predict = model(features)

            val_batch_pred = torch.sum((torch.argmax(predict, dim=1) == labels).type(
                torch.int)) / torch.tensor(labels.shape[0])
            val_acc_list.append(float(val_batch_pred.detach().mean()))

            loss = criterion(predict, labels)
            val_loss_list.append(loss.cpu().detach().numpy())

    total_loss = sum(val_loss_list) / len(val_loss_list)
    total_acc = sum(val_acc_list) / len(val_acc_list)

    return total_loss, total_acc


# %%
# train stage one
def train(model_cfg, dataset_cfg, train_cfg, device, graph_data, k_fold):

    output_folder = os.path.join(train_cfg['OUT_PATH'], train_cfg['NAME'],
                                 'fold{}'.format(str(k_fold)))
    os.makedirs(output_folder, exist_ok=True)

    # 复制配置文件
    file_name = config_path.split('/')[-1]
    copyfile(config_path, os.path.join(output_folder, file_name))

    # 获得数据
    train_dataloader, val_dataloader = get_dataloader(dataset_cfg, graph_data)

    model = PreLable(in_dim=300, hidden_dim=128, n_class=23)

    logging.info('  Model = %s', str(model))
    logging.info('  Training parameters = %d',
                 sum(p.numel() for p in model.parameters() if p.requires_grad))
    logging.info('  Model parameters = %d', sum(p.numel() for p in model.parameters()))
    logging.info('  Num of train examples = %d', len(train_dataloader))
    logging.info('  Num of val examples = %d', len(val_dataloader))

    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    global_step, best_records = 0, [-1, 100, 0]
    for epoch in range(train_cfg['EPOCHS']):
        tr_loss, tr_acc, global_step = train_epoch(epoch, train_cfg, model, train_dataloader,
                                                   optimizer, criterion, global_step,
                                                   train_cfg['LOG_STEP'], device)
        logging.info("Train Epoch %d/%s Finished | Train Loss: %f | Train Acc: %f", epoch + 1,
                     train_cfg['EPOCHS'], tr_loss, tr_acc)
        writer.add_scalar('Train/epoch_loss', tr_loss, epoch)
        writer.add_scalar('Train/epoch_acc', tr_acc, epoch)

        val_loss, val_acc = val_epoch(model, val_dataloader, criterion, device)

        logging.info("Val Epoch %d/%s Finished | Val Loss: %f | Val Acc: %f", epoch + 1,
                     train_cfg['EPOCHS'], val_loss, val_acc)
        writer.add_scalar('Val/epoch_loss', val_loss, epoch)
        writer.add_scalar('Val/epoch_acc', val_acc, epoch)

        if val_loss < best_records[1] or val_acc > best_records[2]:
            best_records = [epoch + 1, val_loss, val_acc]

            model_path = os.path.join(
                output_folder, 'epoch{:02d}_val_loss_{:.4f}_val_acc{:.4f}.pth'.format(
                    epoch + 1, val_loss, val_acc))
            torch.save(model.state_dict(), model_path)


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

    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.datetime.now())
    log_dir = 'logs/pretrain/{}_{}'.format(train_cfg['NAME'], TIMESTAMP)
    writer = SummaryWriter(log_dir=log_dir)

    # 加载图
    k_fold = 2
    graph_data = load_dgl_graph(dataset_cfg['DATA_PATH'],
                                k_fold,
                                norm_feature=dataset_cfg['NORM_FEATURE'])

    # 训练
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train(model_cfg, dataset_cfg, train_cfg, device, graph_data, k_fold)
