import math
import torch
from torch import nn
from torch.nn import functional as F

from dgl import function as fn
from dgl.utils import expand_as_pair, check_eq_shape, dgl_warning


class ACMSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 pass_type='low',  # low, high, unit
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None):
        super(ACMSAGE, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self._pass_type = pass_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)

        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)

        if bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == 'lstm':
            self.lstm.reset_parameters()

        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _compatibility_check(self):
        if not hasattr(self, 'bias'):
            dgl_warning("You are loading a GraphSAGE model trained from a old version of DGL, "
                        "DGL automatically convert it to be compatible with latest version.")
            bias = self.fc_neigh.bias
            self.fc_neigh.bias = None
            if hasattr(self, 'fc_self'):
                if bias is not None:
                    bias = bias + self.fc_self.bias
                    self.fc_self.bias = None
            self.bias = bias

    def _lstm_reducer(self, nodes):
        m = nodes.mailbox['m']  # (B, L, D)
        batch_size = m.shape[0]
        h = (m.new_zeros((1, batch_size, self._in_src_feats)),
             m.new_zeros((1, batch_size, self._in_src_feats)))
        _, (rst, _) = self.lstm(m, h)
        return {'neigh': rst.squeeze(0)}

    def forward(self, data, edge_weight=None):
        graph, feat = data
        self._compatibility_check()
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            msg_fn = fn.copy_src('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                msg_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            h_self = feat_dst

            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            lin_before_mp = self._in_src_feats > self._out_feats

            if self._pass_type == 'low':
                if self._aggre_type == 'mean':
                    graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
                    graph.update_all(msg_fn, fn.mean('m', 'neigh'))
                    h_neigh = graph.dstdata['neigh']
                    if not lin_before_mp:
                        h_neigh = self.fc_neigh(h_neigh)
                elif self._aggre_type == 'gcn':
                    check_eq_shape(feat)
                    graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
                    if isinstance(feat, tuple):  # heterogeneous
                        graph.dstdata['h'] = self.fc_neigh(feat_dst) if lin_before_mp else feat_dst
                    else:
                        if graph.is_block:
                            graph.dstdata['h'] = graph.srcdata['h'][:graph.num_dst_nodes()]
                        else:
                            graph.dstdata['h'] = graph.srcdata['h']
                    graph.update_all(msg_fn, fn.sum('m', 'neigh'))

                    degs = graph.in_degrees().to(feat_dst)
                    h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
                    if not lin_before_mp:
                        h_neigh = self.fc_neigh(h_neigh)
                elif self._aggre_type == 'pool':
                    graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
                    graph.update_all(msg_fn, fn.max('m', 'neigh'))
                    h_neigh = self.fc_neigh(graph.dstdata['neigh'])
                elif self._aggre_type == 'lstm':
                    graph.srcdata['h'] = feat_src
                    graph.update_all(msg_fn, self._lstm_reducer)
                    h_neigh = self.fc_neigh(graph.dstdata['neigh'])
                else:
                    raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

            elif self._pass_type == 'high':
                if self._aggre_type == 'mean':
                    if lin_before_mp:
                        graph.srcdata['h'] = self.fc_neigh(feat_src)
                        h_self = self.fc_neigh(h_self)
                        graph.update_all(msg_fn, fn.mean('m', 'neigh'))
                        h_neigh = h_self - graph.dstdata['neigh']
                    else:
                        graph.srcdata['h'] = feat_src
                        graph.update_all(msg_fn, fn.mean('m', 'neigh'))
                        h_neigh = self.fc_neigh(h_self - graph.dstdata['neigh'])
                elif self._aggre_type == 'gcn':
                    if lin_before_mp:
                        check_eq_shape(feat)
                        graph.srcdata['h'] = self.fc_neigh(feat_src)
                        h_self = self.fc_neigh(h_self)
                        if isinstance(feat, tuple):  # heterogeneous
                            graph.dstdata['h'] = self.fc_neigh(feat_dst)
                        else:
                            if graph.is_block:
                                graph.dstdata['h'] = graph.srcdata['h'][:graph.num_dst_nodes()]
                            else:
                                graph.dstdata['h'] = graph.srcdata['h']
                        graph.update_all(msg_fn, fn.sum('m', 'neigh'))
                        # divide in_degrees
                        degs = graph.in_degrees().to(feat_dst)
                        h_neigh = h_self - (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
                    else:
                        check_eq_shape(feat)
                        graph.srcdata['h'] = feat_src
                        if isinstance(feat, tuple):  # heterogeneous
                            graph.dstdata['h'] = feat_dst
                        else:
                            if graph.is_block:
                                graph.dstdata['h'] = graph.srcdata['h'][:graph.num_dst_nodes()]
                            else:
                                graph.dstdata['h'] = graph.srcdata['h']
                        graph.update_all(msg_fn, fn.sum('m', 'neigh'))
                        # divide in_degrees
                        degs = graph.in_degrees().to(feat_dst)
                        h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
                        h_neigh = self.fc_neigh(h_self - h_neigh)
                elif self._aggre_type == 'pool':
                    graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
                    h_self = F.relu(self.fc_pool(h_self))
                    graph.update_all(msg_fn, fn.max('m', 'neigh'))
                    h_neigh = self.fc_neigh(h_self - graph.dstdata['neigh'])
                elif self._aggre_type == 'lstm':
                    graph.srcdata['h'] = feat_src
                    graph.update_all(msg_fn, self._lstm_reducer)
                    h_neigh = self.fc_neigh(h_self - graph.dstdata['neigh'])
                else:
                    raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

            elif self._pass_type == 'unit':
                h_neigh = self.fc_self(h_self)

            rst = h_neigh

            if self.bias is not None:
                rst = rst + self.bias

            if self.activation is not None:
                rst = self.activation(rst)

            if self.norm is not None:
                rst = self.norm(rst)
            return rst


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.fill_(1.)
        self.bias.data.fill_(0.)

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, attn_drop):
        super(MultiHeadedAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return context_layer


class ACMSageLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, attn_drop=0.0,
                 aggregator_type='lstm', activation=None):
        super(ACMSageLayer, self).__init__()

        self.low_pass = nn.Sequential(
            ACMSAGE(in_feats=in_feats, out_feats=out_feats, aggregator_type=aggregator_type,
                    pass_type='low', activation=activation),
            LayerNorm(out_feats),
            nn.ELU(),
        )

        self.hig_pass = nn.Sequential(
            ACMSAGE(in_feats=in_feats, out_feats=out_feats, aggregator_type=aggregator_type,
                    pass_type='high', activation=activation),
            LayerNorm(out_feats),
            nn.ELU(),
        )

        self.uni_pass = nn.Sequential(
            ACMSAGE(in_feats=in_feats, out_feats=out_feats, aggregator_type=aggregator_type,
                    pass_type='unit', activation=activation),
            LayerNorm(out_feats),
            nn.ELU(),
        )

        self.attn = MultiHeadedAttention(
            num_attention_heads=num_heads,
            hidden_size=out_feats,
            attn_drop=attn_drop,
        )
        self.norm = LayerNorm(out_feats)

    def forward(self, block, features):
        data = (block, features)
        h_L = self.low_pass(data)
        h_H = self.hig_pass(data)
        h_U = self.uni_pass(data)

        h = torch.stack([h_U, h_L, h_H], dim=1)
        h = self.attn(h)[:, 0]
        h = self.norm(h)
        return h


class UniACM(nn.Module):

    def __init__(self, input_size, num_class, num_layers=2, num_heads=4, hidden_size=512,
                 label_drop=0.1, feat_drop=0., attn_drop=0., drop=0.2, aggregator_type='lstm'
                 ):
        super(UniACM, self).__init__()
        self.num_layers = num_layers

        self.graph = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.graph.append(ACMSageLayer(input_size, hidden_size, num_heads, attn_drop,
                                               aggregator_type=aggregator_type))
            else:
                self.graph.append(ACMSageLayer(hidden_size, hidden_size, num_heads, attn_drop,
                                               aggregator_type=aggregator_type))

        self.shortcut = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            LayerNorm(hidden_size),
        )
        self.attns = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

        self.label_embed = nn.Embedding(num_class + 1, input_size, padding_idx=num_class)

        self.feat_mlp = nn.Sequential(
            nn.Linear(2 * input_size, hidden_size),
            LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_size, input_size),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_size, num_class),
        )

        self.dropout = nn.Dropout(drop)
        self.label_dropout = nn.Dropout(label_drop)
        self.feat_dropout = nn.Dropout(feat_drop)

    def forward(self, blocks, input_feats, input_labels):
        input_labels = self.label_embed(input_labels)
        input_labels = self.label_dropout(input_labels)
        input_feats = self.feat_dropout(input_feats)
        feature = torch.cat([input_labels, input_feats], dim=1)
        feature = self.feat_mlp(feature)

        h, feat_dst = feature, feature
        for l in range(self.num_layers):
            feat_dst = expand_as_pair(feat_dst, blocks[l])[1]
            h = self.graph[l](blocks[l], h)
            h = self.dropout(h)

        feat_dst = self.shortcut(feat_dst)

        h = torch.stack([h, feat_dst], dim=1)
        attn_weights = self.attns(h).transpose(-1, -2)
        h = torch.bmm(attn_weights, h)[:, 0]

        output = self.head(h)
        return output
