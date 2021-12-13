# %%
# 引入库
import dgl
import dgl.nn as dglnn
import logging
from dgl.dataloading.neighbor import MultiLayerNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import datetime

import time
import os

from yaml import load, Loader
from utils import load_dgl_graph, set_seed_logger

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)


# %%
# 定义Encoder模型
class GraphAttnEncoder(nn.Module):
    def __init__(self, in_feats, hidden_dim, n_layers, heads, activation, feat_drop, attn_drop,
                 label_drop, num_class):
        super(GraphAttnEncoder, self).__init__()
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.heads = heads
        self.feat_dropout = feat_drop
        self.attn_dropout = attn_drop
        self.activation = activation

        self.layers = nn.ModuleList()

        self.label_embed = nn.Embedding(num_class + 1, in_feats, padding_idx=num_class)
        self.label_dropout = nn.Dropout(label_drop)

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
        input_labels = self.label_embed(input_labels)
        input_labels = self.label_dropout(input_labels)
        h = input_labels + features

        for l in range(self.n_layers):
            h = self.layers[l](blocks[l], h).flatten(1)
        return h


class SimCLRStage1(nn.Module):
    def __init__(self, in_feats, hidden_dim, n_layers, heads, activation, feat_drop, attn_drop,
                 label_drop, num_class):
        super(SimCLRStage1, self).__init__()
        self.encoder = GraphAttnEncoder(in_feats, hidden_dim, n_layers, heads, activation,
                                        feat_drop, attn_drop, label_drop, num_class)

        heads = heads[-1]
        self.g = nn.Sequential(
            nn.Linear(hidden_dim * heads, hidden_dim * heads // 2, bias=False),
            nn.BatchNorm1d(hidden_dim * heads // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * heads // 2, hidden_dim * heads // 4, bias=True))

    def forward(self, blocks, features, input_labels):
        feature = self.encoder(blocks, features, input_labels)
        out = self.g(feature)
        out = F.normalize(out, dim=-1)
        return out


class SimCLRStage2(torch.nn.Module):
    def __init__(self, in_feats, hidden_dim, n_layers, heads, activation, feat_drop, attn_drop,
                 label_drop, num_class):
        super(SimCLRStage2, self).__init__()
        # encoder
        self.encoder = GraphAttnEncoder(in_feats, hidden_dim, n_layers, heads, activation,
                                        feat_drop, attn_drop, label_drop, num_class)
        # classifier
        heads = heads[-1]

        self.g = nn.Sequential(
            nn.Linear(hidden_dim * heads, hidden_dim * heads // 2, bias=False),
            nn.BatchNorm1d(hidden_dim * heads // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * heads // 2, hidden_dim * heads // 4, bias=True))

        self.fc = nn.Sequential(nn.Linear(hidden_dim * heads // 4, num_class, bias=True))

        # for param in self.f.parameters():
        #     param.requires_grad = False

    def forward(self, blocks, features, input_labels):
        feature = self.encoder(blocks, features, input_labels)
        feature = self.g(feature)
        out = self.fc(feature)
        return out


# %%
# loss
class SimLoss(nn.Module):
    def __init__(self):
        super(SimLoss, self).__init__()

    def forward(self, out_1, out_2, temperature=0.5):
        # 分母 ：X.X.T，再去掉对角线值，分析结果一行，可以看成它与除了这行外的其他行都进行了点积运算（包括out_1和out_2）,
        # 而每一行为一个batch的一个取值，即一个输入图像的特征表示，
        # 因此，X.X.T，再去掉对角线值表示，每个输入图像的特征与其所有输出特征（包括out_1和out_2）的点积，用点积来衡量相似性
        # 加上exp操作，该操作实际计算了分母
        B, _ = out_1.shape
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.mm(out, out.t().contiguous())
        sim_matrix = torch.exp(sim_matrix) / temperature
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * B, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * B, -1)

        # 分子： *为对应位置相乘，也是点积
        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss


#
class MultiLayerDropoutSampler(dgl.dataloading.BlockSampler):
    def __init__(self, p, num_layers):
        super().__init__(num_layers)

        self.p = p

    def sample_frontier(self, block_id, g, seed_nodes, *args, **kwargs):
        # 获取种 `seed_nodes` 的所有入边
        src, dst = dgl.in_subgraph(g, seed_nodes).all_edges()
        # 以概率p随机选择边
        mask = torch.zeros_like(src).bernoulli_(self.p) == 0
        src = src[mask]
        dst = dst[mask]
        # 返回一个与初始图有相同节点的边界
        frontier = dgl.graph((src, dst), num_nodes=g.number_of_nodes())
        frontier = dgl.add_self_loop(frontier)
        return frontier

    def __len__(self):
        return self.num_layers


# 定义采样函数
class MultiLayerProbsSampler(dgl.dataloading.BlockSampler):
    def __init__(self, probs, max_fanouts, num_layers):
        super().__init__(num_layers)
        self.probs = probs
        self.max_fanouts = max_fanouts

    def sample_frontier(self, block_id, g, seed_nodes, *args, **kwargs):
        max_fanout = self.max_fanouts[block_id]
        prob = self.probs[block_id]

        # 保证边的个数小于
        # return dgl.sampling.sample_neighbors(g, seed_nodes, int(max_fanout * prob))
        # src, dst = dgl.sampling.sample_neighbors(g, seed_nodes, max_fanout).all_edges()
        src, dst = dgl.in_subgraph(g, seed_nodes).all_edges()
        # 以概率p随机选择边
        mask = torch.zeros_like(src).bernoulli_(prob) == 1
        src = src[mask]
        dst = dst[mask]

        # 返回一个与初始图有相同节点的边界
        frontier = dgl.graph((src, dst), num_nodes=g.number_of_nodes())
        # frontier = dgl.add_self_loop(frontier)
        return frontier

    def __len__(self):
        return self.num_layers


# %%
def get_dataloader(dataset_cfg, graph_data):
    graph, labels, train_nid, val_nid, test_nid, node_feat = graph_data
    sampler = MultiLayerNeighborSampler(dataset_cfg['FANOUTS'])  # 50

    # sampler = MultiLayerProbsSampler(probs=[0.2, 0.2], max_fanouts=[60, 100], num_layers=2)  # 50
    # sampler = MultiLayerDropoutSampler(p=0.5, num_layers=2)  # 50

    train_dataloader = NodeDataLoader(graph,
                                      train_nid,
                                      sampler,
                                      batch_size=dataset_cfg['BATCH_SIZE'],
                                      shuffle=True,
                                      drop_last=False,
                                      num_workers=dataset_cfg['NUM_WORKERS'])
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
def train_epoch(epoch, train_cfg, model, train_dataloader, node_feat, labels, optimizer, criterion,
                n_classes, device, global_step, log_step):
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

        predict = model(blocks, input_feats, input_labels)
        tr_batch_pred = torch.sum((torch.argmax(predict, dim=1) == batch_labels).type(
            torch.int)) / torch.tensor(batch_labels.shape[0])
        train_acc_list.append(float(tr_batch_pred.detach().mean()))

        loss = criterion(predict, batch_labels)
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

            writer.add_scalar('Loss/train', train_loss_list[-1], global_step)
            writer.add_scalar('Acc/train', tr_batch_pred.detach().mean(), global_step)

            start_time = time.time()

    total_loss = sum(train_loss_list) / len(train_loss_list)
    total_acc = sum(train_acc_list) / len(train_acc_list)
    return total_loss, total_acc, global_step


# %%
def val_epoch(model, val_dataloader, node_feat, labels, criterion, n_classes, device):
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

            predict = model(blocks, input_feats, input_labels)
            val_batch_pred = torch.sum((torch.argmax(predict, dim=1) == batch_labels).type(torch.int)) /\
                             torch.tensor(batch_labels.shape[0])
            val_acc_list.append(float(val_batch_pred.detach().mean()))

            loss = criterion(predict, batch_labels)
            val_loss_list.append(loss.cpu().detach().numpy())

        total_loss = sum(val_loss_list) / len(val_loss_list)
        total_acc = sum(val_acc_list) / len(val_acc_list)

        return total_loss, total_acc


# %%
# train stage one
def train(model_cfg, dataset_cfg, train_cfg, device, graph_data, k_fold):

    output_folder = train_cfg['OUT_PATH'] + str(k_fold)
    os.makedirs(output_folder, exist_ok=True)

    # 获得数据
    train_dataloader, val_dataloader, node_feat, labels = get_dataloader(dataset_cfg, graph_data)

    gat_cfg = model_cfg['GAT']

    model = SimCLRStage2(in_feats=gat_cfg['IN_FEATS'],
                         hidden_dim=gat_cfg['HIDDEN_DIM'],
                         n_layers=gat_cfg['N_LAYERS'],
                         activation=F.relu,
                         heads=gat_cfg['HEADS'],
                         feat_drop=gat_cfg['FEAT_DROP'],
                         attn_drop=gat_cfg['ATTN_DROP'],
                         label_drop=gat_cfg['LABEL_DROP'],
                         num_class=gat_cfg['N_CLASS'])

    if model_cfg['LOAD_CHECKPOINT']:
        # 加载预训练参数
        pretrained_dict = torch.load(model_cfg['CHECKPOINT'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        if model_cfg['FREEZE']:
            # 冻结预训练参数
            for param in model.encoder.parameters():
                param.requires_grad = False

    logging.info('  Model = %s', str(model))
    logging.info('  Training parameters = %d',
                 sum(p.numel() for p in model.parameters() if p.requires_grad))
    logging.info('  Model parameters = %d', sum(p.numel() for p in model.parameters()))
    logging.info('  Num of train examples = %d', len(train_dataloader))
    logging.info('  Num of val examples = %d', len(val_dataloader))

    model = model.to(device)

    # 加载参数

    lossLR = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=train_cfg['LEARNING_RATE'],
                                 weight_decay=train_cfg['WEIGHT_DECAY'])

    global_step, best_records = 0, [-1, 0]
    for epoch in range(train_cfg['EPOCHS']):
        tr_loss, tr_acc, global_step = train_epoch(epoch, train_cfg, model, train_dataloader,
                                                   node_feat, labels, optimizer, lossLR,
                                                   dataset_cfg['N_CLASS'], device, global_step,
                                                   train_cfg['LOG_STEP'])
        logging.info("Train Epoch %d/%s Finished | Train Loss: %f | Train Acc: %f", epoch + 1,
                     train_cfg['EPOCHS'], tr_loss, tr_acc)
        writer.add_scalar('Loss/train epoch', tr_loss, epoch)
        writer.add_scalar('Acc/train epoch', tr_acc, epoch)

        val_loss, val_acc = val_epoch(model, val_dataloader, node_feat, labels, lossLR,
                                      dataset_cfg['N_CLASS'], device)
        logging.info("Val Epoch %d/%s Finished | Val Loss: %f | Val Acc: %f", epoch + 1,
                     train_cfg['EPOCHS'], val_loss, val_acc)
        writer.add_scalar('Loss/val epoch', val_loss, epoch)
        writer.add_scalar('Acc/val epoch', val_acc, epoch)

        if val_acc > best_records[1]:
            best_records = [epoch + 1, val_acc]
            model_path = os.path.join(
                output_folder, 'simclr_one_layer_load{}_finetune_model_epoch{:02d}'.format(
                    str(model_cfg['LOAD_CHECKPOINT']), epoch + 1) +
                '_valloss_{:.4f}_valacc_{:.4f}'.format(val_loss, val_acc) + '.pth')
            torch.save(model.state_dict(), model_path)


def val(model_cfg, dataset_cfg, device, graph_data):
    train_dataloader, val_dataloader, node_feat, labels = get_dataloader(dataset_cfg, graph_data)

    gat_cfg = model_cfg['GAT']

    model = SimCLRStage2(in_feats=gat_cfg['IN_FEATS'],
                         hidden_dim=gat_cfg['HIDDEN_DIM'],
                         n_layers=gat_cfg['N_LAYERS'],
                         activation=F.relu,
                         heads=gat_cfg['HEADS'],
                         feat_drop=gat_cfg['FEAT_DROP'],
                         attn_drop=gat_cfg['ATTN_DROP'],
                         label_drop=gat_cfg['LABEL_DROP'],
                         num_class=gat_cfg['N_CLASS'])

    model = model.to(device)

    lossLR = nn.CrossEntropyLoss().to(device)

    val_loss, val_acc = val_epoch(model, val_dataloader, node_feat, labels, lossLR,
                                  dataset_cfg['N_CLASS'], device)
    logging.info("Val Finished | Val Loss: %f | Val Acc: %f", val_loss, val_acc)


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
    log_dir = 'logs/finetune/' + TIMESTAMP + str(model_cfg['LOAD_CHECKPOINT'])
    writer = SummaryWriter(log_dir=log_dir)

    # 加载图
    k_fold = 2
    graph_data = load_dgl_graph(dataset_cfg['DATA_PATH'],
                                k_fold,
                                norm_feature=dataset_cfg['NORM_FEATURE'])

    # 训练
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train(model_cfg, dataset_cfg, train_cfg, device, graph_data, k_fold)
