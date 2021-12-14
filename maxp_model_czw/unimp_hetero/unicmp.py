import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn as dglnn
from dgl.utils import expand_as_pair


def gcn_init_layers(layer):
    nn.init.xavier_normal_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def linear_init_layers(layer):
    dim = layer.in_features
    bias_bound = 1.0 / math.sqrt(dim)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            nn.init.uniform_(layer.bias, a=-bias_bound, b=bias_bound)

    negative_slope = math.sqrt(5)
    gain = math.sqrt(2.0 / (1 + negative_slope ** 2))
    std = gain / math.sqrt(dim)
    weight_bound = math.sqrt(3.0) * std
    nn.init.uniform_(layer.weight, a=-weight_bound, b=weight_bound)


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


class GraphSageLayer(nn.Module):
    def __init__(self, in_feats, out_feats, edge_type, activation=F.elu,
                 aggregator_type='lstm', dropout=0.):
        super(GraphSageLayer, self).__init__()
        self.edge_type = edge_type

        self.graph = nn.ModuleList()
        self.norm = nn.ModuleList()

        for _ in edge_type:
            self.graph.append(dglnn.SAGEConv(in_feats=in_feats, out_feats=out_feats,
                                             aggregator_type=aggregator_type))
            self.norm.append(LayerNorm(out_feats))

        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, block, feature):
        hidden = []
        for i, edge in enumerate(self.edge_type):
            h = self.graph[i](block[edge], feature)
            h = self.norm[i](h)
            h = self.activation(h)
            h = self.dropout(h)
            hidden.append(h)

        hidden = torch.stack(hidden, dim=1)
        return hidden


class GraphConvLayer(nn.Module):
    def __init__(self, in_feats, out_feats, edge_type, activation=F.elu,
                 norm_type='both', dropout=0.):
        super(GraphConvLayer, self).__init__()
        self.edge_type = edge_type
        self.graph = nn.ModuleList()
        self.norm = nn.ModuleList()

        for _ in edge_type:
            self.graph.append(dglnn.GraphConv(in_feats=in_feats, out_feats=out_feats,
                                              norm=norm_type))
            self.norm.append(LayerNorm(out_feats))

        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, block, feature):
        hidden = []
        for i, edge in enumerate(self.edge_type):
            h = self.graph[i](block[edge], feature)
            h = self.norm[i](h)
            h = self.activation(h)
            h = self.dropout(h)
            hidden.append(h)

        hidden = torch.stack(hidden, dim=1)
        return hidden


class GraphAttnLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, edge_type, activation=F.elu,
                 attn_drop=0., dropout=0.):
        super(GraphAttnLayer, self).__init__()
        self.edge_type = edge_type
        self.graph = nn.ModuleList()
        self.norm = nn.ModuleList()

        for _ in edge_type:
            self.graph.append(dglnn.GATConv(in_feats=in_feats, out_feats=out_feats // num_heads,
                                            num_heads=num_heads, attn_drop=attn_drop))
            self.norm.append(LayerNorm(out_feats))

        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, block, feature):
        hidden = []
        for i, edge in enumerate(self.edge_type):
            h = self.graph[i](block[edge], feature).flatten(1)
            h = self.norm[i](h)
            h = self.activation(h)
            h = self.dropout(h)
            hidden.append(h)

        hidden = torch.stack(hidden, dim=1)
        return hidden


class ShortCutLayer(nn.Module):
    def __init__(self, in_feats, out_feats, edge_type, activation=F.elu, dropout=0.):
        super(ShortCutLayer, self).__init__()
        self.edge_type = edge_type
        self.graph = nn.ModuleList()
        self.norm = nn.ModuleList()

        for _ in edge_type:
            self.graph.append(nn.Linear(in_feats, out_feats))
            self.norm.append(LayerNorm(out_feats))

        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.graph:
            linear_init_layers(m)

    def forward(self, block, feature):
        hidden = []
        for i, edge in enumerate(self.edge_type):
            h = expand_as_pair(feature, block[edge])[1]
            h = self.graph[i](h)
            h = self.norm[i](h)
            h = self.activation(h)
            h = self.dropout(h)
            hidden.append(h)

        hidden = torch.stack(hidden, dim=1)
        return hidden


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

        self.reset_parameters()

    def reset_parameters(self):
        linear_init_layers(self.key)
        linear_init_layers(self.query)
        linear_init_layers(self.value)

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


class UniCMP(nn.Module):
    def __init__(self, input_size, feat_size, num_class, edge_type, num_layers=2, num_heads=8,
                 hidden_size=512, attn_heads=8, label_drop=0.1, feat_drop=0., graph_drop=0.,
                 attn_drop=0., drop=0.2, use_sage=True, use_conv=True, use_attn=True,
                 use_resnet=True, use_densenet=True
                 ):
        super(UniCMP, self).__init__()
        self.edge_type = edge_type
        self.num_layers = num_layers
        self.use_sage = use_sage
        self.use_conv = use_conv
        self.use_attn = use_attn
        self.use_resnet = use_resnet
        self.use_densenet = use_densenet

        if use_sage:
            self.graphsage = nn.ModuleList()
            for i in range(num_layers):
                if i == 0:
                    self.graphsage.append(GraphSageLayer(input_size, hidden_size, edge_type=edge_type, dropout=graph_drop))
                else:
                    self.graphsage.append(GraphSageLayer(hidden_size, hidden_size, edge_type=edge_type, dropout=graph_drop))

        if use_conv:
            self.graphconv = nn.ModuleList()
            for i in range(num_layers):
                if i == 0:
                    self.graphconv.append(GraphConvLayer(input_size, hidden_size, edge_type=edge_type, dropout=graph_drop))
                else:
                    self.graphconv.append(GraphConvLayer(hidden_size, hidden_size, edge_type=edge_type, dropout=graph_drop))

        if use_attn:
            self.graphattn = nn.ModuleList()
            for i in range(num_layers):
                if i == 0:
                    self.graphattn.append(GraphAttnLayer(input_size, hidden_size, num_heads=attn_heads, edge_type=edge_type,
                                                         dropout=graph_drop))
                else:
                    self.graphattn.append(GraphAttnLayer(hidden_size, hidden_size, num_heads=attn_heads, edge_type=edge_type,
                                                         dropout=graph_drop))

        if use_densenet:
            self.graphskip = nn.ModuleList()
            for i in range(num_layers):
                for j in range(i + 1):
                    if j == 0:
                        self.graphskip.append(ShortCutLayer(input_size, hidden_size, edge_type=edge_type, dropout=graph_drop))
                    else:
                        self.graphskip.append(ShortCutLayer(hidden_size, hidden_size, edge_type=edge_type, dropout=graph_drop))
        elif use_resnet:
            self.graphskip = nn.ModuleList()
            for i in range(num_layers):
                if i == 0:
                    self.graphskip.append(ShortCutLayer(input_size, hidden_size, edge_type=edge_type, dropout=graph_drop))
                else:
                    self.graphskip.append(ShortCutLayer(hidden_size, hidden_size, edge_type=edge_type, dropout=graph_drop))

        self.attn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.attn_layers.append(MultiHeadedAttention(num_heads, hidden_size, attn_drop))

        self.norm_layers = nn.ModuleList()
        for i in range(num_layers):
            self.norm_layers.append(LayerNorm(hidden_size))

        self.label_embed = nn.Embedding(num_class + 1, input_size, padding_idx=num_class)
        self.feat_mlp = nn.Sequential(
            nn.Linear(feat_size, hidden_size),
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

        self.label_dropout = nn.Dropout(label_drop)
        self.feat_dropout = nn.Dropout(feat_drop)
        self.dropout = nn.Dropout(drop)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.feat_mlp:
            if isinstance(m, nn.Linear):
                gcn_init_layers(m)
        for m in self.head:
            if isinstance(m, nn.Linear):
                gcn_init_layers(m)

    def forward(self, blocks, input_feats, input_labels):
        input_labels = self.label_embed(input_labels)
        input_labels = self.label_dropout(input_labels)

        input_feats = self.feat_dropout(input_feats)
        feature = torch.cat([input_labels, input_feats], dim=1)
        feature = self.feat_mlp(feature)

        logits = [feature]
        for l in range(self.num_layers):
            hidden = []

            if self.use_sage:
                h1 = self.graphsage[l](blocks[l], logits[-1])
                hidden.append(h1)

            if self.use_conv:
                h2 = self.graphconv[l](blocks[l], logits[-1])
                hidden.append(h2)

            if self.use_attn:
                h3 = self.graphattn[l](blocks[l], logits[-1])
                hidden.append(h3)

            if self.use_densenet:
                for k, tensor in enumerate(logits):
                    h4 = self.graphskip[l * (l + 1) // 2 + k](blocks[l], tensor)
                    hidden.append(h4)
            elif self.use_resnet:
                h4 = self.graphskip[l](blocks[l], logits[-1])
                hidden.append(h4)

            h = torch.cat(hidden, dim=1)
            h = self.attn_layers[l](h)[:, 0]
            h = self.norm_layers[l](h)
            h = self.dropout(h)

            logits = [expand_as_pair(tensor, blocks[l])[1] for tensor in logits]
            logits.append(h)

        output = self.head(logits[-1])
        return output
