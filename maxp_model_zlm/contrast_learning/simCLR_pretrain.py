# %%
# 引入库
import dgl
import dgl.nn as dglnn
import logging
from dgl.dataloading.neighbor import MultiLayerNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader
from dgl.dataloading.pytorch.dataloader import _NodeCollator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import datetime

import time
import os
from shutil import copyfile

from yaml import load, Loader
from utils import load_dgl_graph, set_seed_logger

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)


# %%
# 定义Encoder模型
class GraphAttnEncoder(nn.Module):
    def __init__(self, in_feats, hidden_dim, n_layers, heads, activation, feat_drop, attn_drop,
                 label_drop, do_label_embed, num_class):
        super(GraphAttnEncoder, self).__init__()
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.heads = heads
        self.feat_dropout = feat_drop
        self.attn_dropout = attn_drop
        self.activation = activation
        self.do_label_embed = do_label_embed

        self.layers = nn.ModuleList()

        self.label_embed = nn.Embedding(num_class + 1, in_feats, padding_idx=num_class)
        self.label_dropout = nn.Dropout(label_drop)
        self.label_embed_layer = nn.Sequential(nn.Linear(in_feats * 2, in_feats * 4),
                                               nn.LayerNorm(in_feats * 4), nn.ReLU(),
                                               nn.Linear(in_feats * 4, in_feats))

        # build multiple layers
        self.layers.append(
            dglnn.GATConv(in_feats=self.in_feats,
                          out_feats=self.hidden_dim,
                          num_heads=self.heads[0],
                          feat_drop=self.feat_dropout,
                          attn_drop=self.attn_dropout,
                          activation=self.activation))

        for l in range(1, self.n_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layers.append(
                dglnn.GATConv(in_feats=self.hidden_dim * self.heads[l - 1],
                              out_feats=self.hidden_dim,
                              num_heads=self.heads[l],
                              feat_drop=self.feat_dropout,
                              attn_drop=self.attn_dropout,
                              activation=self.activation))

    def forward(self, blocks, features, input_labels):
        h = features
        if self.do_label_embed:
            input_labels_embed = self.label_embed(input_labels)
            input_labels_embed = self.label_dropout(input_labels_embed)
            # h = input_labels + h
            h = self.label_embed_layer(torch.cat((h, input_labels_embed), dim=-1))

        for l in range(self.n_layers):
            h = self.layers[l](blocks[l], h).flatten(1)
        return h


class GraphConvEncoder(nn.Module):
    def __init__(self, in_feats, out_feats, n_layers, activation, feat_drop, label_drop,
                 do_label_embed, num_class):
        super(GraphConvEncoder, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.n_layers = n_layers
        self.activation = activation
        self.do_label_embed = do_label_embed

        self.layers = nn.ModuleList()

        self.label_embed = nn.Embedding(num_class + 1, in_feats, padding_idx=num_class)
        self.label_dropout = nn.Dropout(label_drop)
        self.feat_dropout = nn.Dropout(feat_drop)
        self.label_embed_layer = nn.Sequential(nn.Linear(in_feats * 2, in_feats * 4),
                                               nn.LayerNorm(in_feats * 4), nn.ReLU(),
                                               nn.Linear(in_feats * 4, in_feats))

        # build multiple layers
        self.layers.append(
            dglnn.GraphConv(in_feats=self.in_feats,
                            out_feats=self.out_feats,
                            weight=True,
                            bias=True,
                            activation=self.activation))

        for l in range(1, self.n_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layers.append(
                dglnn.GraphConv(in_feats=self.out_feats,
                                out_feats=self.out_feats,
                                weight=True,
                                bias=True,
                                activation=self.activation))

    def forward(self, blocks, features, input_labels):
        h = features
        if self.do_label_embed:
            input_labels_embed = self.label_embed(input_labels)
            input_labels_embed = self.label_dropout(input_labels_embed)
            # h = input_labels + h
            h = self.label_embed_layer(torch.cat((h, input_labels_embed), dim=-1))

        for l in range(self.n_layers):
            h = self.feat_dropout(h)
            h = self.layers[l](blocks[l], h)
        return h


class SimCLRStage1(nn.Module):
    def __init__(self, in_feats, hidden_dim, n_layers, heads, activation, feat_drop, attn_drop,
                 label_drop, do_label_embed, num_class):
        super(SimCLRStage1, self).__init__()
        self.encoder = GraphAttnEncoder(in_feats, hidden_dim, n_layers, heads, activation,
                                        feat_drop, attn_drop, label_drop, do_label_embed, num_class)

        heads = heads[-1]
        self.g = nn.Sequential(
            nn.Linear(hidden_dim * heads, hidden_dim * heads // 2, bias=False),
            nn.BatchNorm1d(hidden_dim * heads // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * heads // 2, hidden_dim * heads // 4, bias=True))

        self.f = nn.Sequential(nn.Linear(hidden_dim * heads // 4, num_class, bias=True))

    def forward(self, blocks, features, input_labels):
        feature = self.encoder(blocks, features, input_labels)
        feature = self.g(feature)
        out = self.f(feature)

        feature = F.normalize(feature, dim=-1)
        return feature, out


class SimCLRStage2(nn.Module):
    def __init__(self, in_feats, out_feats, n_layers, activation, feat_drop, label_drop,
                 do_label_embed, num_class):
        super(SimCLRStage2, self).__init__()
        self.encoder = GraphConvEncoder(in_feats, out_feats, n_layers, activation, feat_drop,
                                        label_drop, do_label_embed, num_class)

        self.g = nn.Sequential(nn.Linear(out_feats, out_feats // 2, bias=False),
                               nn.BatchNorm1d(out_feats // 2), nn.ReLU(inplace=True),
                               nn.Linear(out_feats // 2, out_feats // 4, bias=True))

        self.f = nn.Sequential(nn.Linear(out_feats // 4, num_class, bias=True))

    def forward(self, blocks, features, input_labels):
        feature = self.encoder(blocks, features, input_labels)
        feature = self.g(feature)
        out = self.f(feature)

        feature = F.normalize(feature, dim=-1)
        return feature, out


class SimSimSiam(nn.Module):
    def __init__(self, in_feats, hidden_dim, n_layers, heads, activation, feat_drop, attn_drop,
                 label_drop, do_label_embed, num_class):
        super(SimCLRStage1, self).__init__()
        self.encoder = GraphAttnEncoder(in_feats, hidden_dim, n_layers, heads, activation,
                                        feat_drop, attn_drop, label_drop, do_label_embed, num_class)

        heads = heads[-1]
        self.f = nn.Sequential(
            self.encoder, nn.Linear(hidden_dim * heads, hidden_dim * heads // 2, bias=False),
            nn.BatchNorm1d(hidden_dim * heads // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * heads // 2, hidden_dim * heads // 2, bias=False),
            nn.BatchNorm1d(hidden_dim * heads // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * heads // 2, hidden_dim * heads, bias=False),
            nn.BatchNorm1d(hidden_dim * heads, affine=False))

        self.h = nn.Sequential(nn.Linear(hidden_dim * heads, hidden_dim * heads // 2, bias=False),
                               nn.BatchNorm1d(hidden_dim * heads // 2), nn.ReLU(inplace=True),
                               nn.Linear(hidden_dim * heads // 2, hidden_dim * heads))

    def forward(self, blocks, features, input_labels):
        z = self.f(blocks, features, input_labels)
        p = self.h(z)
        return z, p


# %%
# loss
class SimLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(SimLoss, self).__init__()
        self.temperature = temperature

    def forward(self, out_1, out_2):
        # 分母 ：X.X.T，再去掉对角线值，分析结果一行，可以看成它与除了这行外的其他行都进行了点积运算（包括out_1和out_2）,
        # 而每一行为一个batch的一个取值，即一个输入图像的特征表示，
        # 因此，X.X.T，再去掉对角线值表示，每个输入图像的特征与其所有输出特征（包括out_1和out_2）的点积，用点积来衡量相似性
        # 加上exp操作，该操作实际计算了分母
        B, _ = out_1.shape
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.mm(out, out.t().contiguous())
        sim_matrix = torch.exp(sim_matrix) / self.temperature
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * B, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * B, -1)

        # 分子： *为对应位置相乘，也是点积
        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss


# %%
def get_dataloader(dataset_cfg, graph_data):
    graph, labels, train_nid, val_nid, test_nid, node_feat = graph_data
    sampler = MultiLayerNeighborSampler(dataset_cfg['FANOUTS'])  # 50
    sampler2 = MultiLayerNeighborSampler(dataset_cfg['FANOUTS'] / 2)  # 50

    train_dataloader = NodeDataLoader(graph,
                                      train_nid,
                                      sampler,
                                      batch_size=dataset_cfg['BATCH_SIZE'],
                                      shuffle=True,
                                      drop_last=False,
                                      num_workers=dataset_cfg['NUM_WORKERS'])
    train_dataloader.collator2 = _NodeCollator(g=graph, nids=train_nid, block_sampler=sampler2)
    val_dataloader = NodeDataLoader(graph,
                                    val_nid,
                                    sampler,
                                    batch_size=dataset_cfg['BATCH_SIZE'],
                                    shuffle=True,
                                    drop_last=False,
                                    num_workers=dataset_cfg['NUM_WORKERS'])

    return train_dataloader, val_dataloader, node_feat, labels


# %%
def load_subtensor(node_feats, labels, seeds, input_nodes, n_classes, device, training=False):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    input_feats = node_feats[input_nodes].to(device)

    masklabels = labels.clone()
    masklabels[seeds] = -1
    input_labels = masklabels[input_nodes]

    if training:
        rd_m = torch.rand(input_labels.shape[0])
        rd_y = torch.randint(0, n_classes, size=input_labels.shape)
        input_labels[rd_m < 0.15] = -1
        input_labels[rd_m > 0.97] = rd_y[rd_m > 0.97]

    input_labels[input_labels < 0] = n_classes
    input_labels = input_labels.to(device)

    batch_labels = labels[seeds].to(device)
    return input_feats, input_labels, batch_labels


# %%
def train_epoch(epoch, train_cfg, model, train_dataloader, node_feat, labels, optimizer, criterion1,
                criterion2, n_classes, device, global_step, log_step):
    torch.cuda.empty_cache()
    model.train()
    train_loss_list, train_acc_list = [], []

    start_time = time.time()
    for step, batch_data in enumerate(train_dataloader):
        # 获得数据对
        input_nodes, output_nodes, blocks = batch_data
        input_feats, input_labels, batch_labels = load_subtensor(node_feat,
                                                                 labels,
                                                                 output_nodes,
                                                                 input_nodes,
                                                                 n_classes,
                                                                 device,
                                                                 training=True)
        blocks = [block.to(device) for block in blocks]

        _batch_data = train_dataloader.collator.collate(output_nodes)
        _input_nodes, _output_nodes, _blocks = _batch_data
        _input_feats, _input_labels, _batch_labels = load_subtensor(node_feat,
                                                                    labels,
                                                                    _output_nodes,
                                                                    _input_nodes,
                                                                    n_classes,
                                                                    device,
                                                                    training=True)

        _blocks = [_block.to(device) for _block in _blocks]

        feature_l, pre_l = model(blocks, input_feats, input_labels)
        feature_r, pre_r = model(_blocks, _input_feats, _input_labels)

        predict = (pre_l + pre_r) / 2
        tr_batch_pred = torch.sum((torch.argmax(predict, dim=1) == batch_labels).type(
            torch.int)) / torch.tensor(batch_labels.shape[0])
        train_acc_list.append(float(tr_batch_pred.detach().mean()))

        loss1 = criterion1(feature_l, feature_r)
        loss2 = criterion2(predict, batch_labels)
        loss = w_1 * loss1 + w_2 * loss2

        train_loss_list.append(loss.cpu().detach().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step = global_step + 1

        if global_step % log_step == 0:
            logging.info("Epoch: %d/%s, Step: %d/%d, Loss: %f, Acc: %f, Time/step: %f", epoch + 1,
                         train_cfg['EPOCHS'], step + 1, len(train_dataloader),
                         float(train_loss_list[-1]), float(tr_batch_pred.detach().mean()),
                         (time.time() - start_time))

            writer.add_scalar('Train/loss1', loss1.cpu().detach().numpy(), global_step)
            writer.add_scalar('Train/loss2', loss2.cpu().detach().numpy(), global_step)
            writer.add_scalar('Train/loss', loss.cpu().detach().numpy(), global_step)
            writer.add_scalar('Train/acc', tr_batch_pred.detach().mean(), global_step)

            start_time = time.time()

    total_loss = sum(train_loss_list) / len(train_loss_list)
    total_acc = sum(train_acc_list) / len(train_acc_list)
    return total_loss, total_acc, global_step


# %%
def val_epoch(model, val_dataloader, node_feat, labels, criterion1, criterion2, n_classes, device):
    torch.cuda.empty_cache()
    model.eval()

    val_loss_list, val_acc_list = [], []

    with torch.no_grad():
        for _, batch_data in enumerate(val_dataloader):
            # 获得数据对
            input_nodes, output_nodes, blocks = batch_data
            input_feats, input_labels, batch_labels = load_subtensor(node_feat,
                                                                     labels,
                                                                     output_nodes,
                                                                     input_nodes,
                                                                     n_classes,
                                                                     device,
                                                                     training=False)
            blocks = [block.to(device) for block in blocks]

            _batch_data = val_dataloader.collator.collate(output_nodes)
            _input_nodes, _output_nodes, _blocks = _batch_data
            _input_feats, _input_labels, _ = load_subtensor(node_feat,
                                                            labels,
                                                            _output_nodes,
                                                            _input_nodes,
                                                            n_classes,
                                                            device,
                                                            training=False)

            _blocks = [_block.to(device) for _block in _blocks]

            feature_l, pre_l = model(blocks, input_feats, input_labels)
            feature_r, pre_r = model(_blocks, _input_feats, _input_labels)

            predict = (pre_l + pre_r) / 2
            val_batch_pred = torch.sum((torch.argmax(predict, dim=1) == batch_labels).type(torch.int)) /\
                             torch.tensor(batch_labels.shape[0])
            val_acc_list.append(float(val_batch_pred.detach().mean()))

            loss1 = criterion1(feature_l, feature_r)
            loss2 = criterion2(predict, batch_labels)
            loss = w_1 * loss1 + w_2 * loss2

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
    train_dataloader, val_dataloader, node_feat, labels = get_dataloader(dataset_cfg, graph_data)

    gcn_cfg = model_cfg['GCN']
    model = SimCLRStage2(in_feats=gcn_cfg['IN_FEATS'],
                         out_feats=gcn_cfg['OUT_FEATS'],
                         n_layers=gcn_cfg['N_LAYERS'],
                         activation=F.relu,
                         feat_drop=gcn_cfg['FEAT_DROP'],
                         label_drop=gcn_cfg['LABEL_DROP'],
                         num_class=gcn_cfg['N_CLASS'],
                         do_label_embed=gcn_cfg['LABEL_EMB'])

    # gat_cfg = model_cfg['GAT']
    # model = SimCLRStage1(in_feats=gat_cfg['IN_FEATS'],
    #                      hidden_dim=gat_cfg['HIDDEN_DIM'],
    #                      n_layers=gat_cfg['N_LAYERS'],
    #                      activation=F.relu,
    #                      heads=gat_cfg['HEADS'],
    #                      feat_drop=gat_cfg['FEAT_DROP'],
    #                      attn_drop=gat_cfg['ATTN_DROP'],
    #                      label_drop=gat_cfg['LABEL_DROP'],
    #                      num_class=gat_cfg['N_CLASS'],
    #                      do_label_embed=gat_cfg['LABEL_EMB'])

    logging.info('  Model = %s', str(model))
    logging.info('  Training parameters = %d',
                 sum(p.numel() for p in model.parameters() if p.requires_grad))
    logging.info('  Model parameters = %d', sum(p.numel() for p in model.parameters()))
    logging.info('  Num of train examples = %d', len(train_dataloader))
    logging.info('  Num of val examples = %d', len(val_dataloader))

    model = model.to(device)

    loss1 = SimLoss().to(device)
    loss2 = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    global_step, best_records = 0, [-1, 100, 0]
    for epoch in range(train_cfg['EPOCHS']):
        tr_loss, tr_acc, global_step = train_epoch(epoch, train_cfg, model, train_dataloader,
                                                   node_feat, labels, optimizer, loss1, loss2,
                                                   dataset_cfg['N_CLASS'], device, global_step,
                                                   train_cfg['LOG_STEP'])
        logging.info("Train Epoch %d/%s Finished | Train Loss: %f | Train Acc: %f", epoch + 1,
                     train_cfg['EPOCHS'], tr_loss, tr_acc)
        writer.add_scalar('Train/epoch_loss', tr_loss, epoch)
        writer.add_scalar('Train/epoch_acc', tr_acc, epoch)

        val_loss, val_acc = val_epoch(model, val_dataloader, node_feat, labels, loss1, loss2,
                                      dataset_cfg['N_CLASS'], device)

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

    w_1, w_2 = 1, 0

    # 加载图
    k_fold = 2
    graph_data = load_dgl_graph(dataset_cfg['DATA_PATH'],
                                k_fold,
                                norm_feature=dataset_cfg['NORM_FEATURE'])

    # 训练
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train(model_cfg, dataset_cfg, train_cfg, device, graph_data, k_fold)
