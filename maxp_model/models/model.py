import math

import torch
import torch.nn as thnn
import torch.nn.functional as F
import dgl.nn as dglnn


class Se(thnn.Module):
    def __init__(self, feature_1, feature_2):
        super().__init__()
        self.se = thnn.Sequential(
            thnn.Linear(feature_1, feature_2),
            thnn.BatchNorm1d(feature_2),
            thnn.Sigmoid()
        )
    
    def forward(self, features):
        return self.se(features)



class GraphSageModel(thnn.Module):

    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 activation=F.relu,
                 dropout=0):
        super(GraphSageModel, self).__init__()
        self.in_feats = in_feats
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.activation = activation
        self.dropout = thnn.Dropout(dropout)

        self.layers = thnn.ModuleList()

        # build multiple layers
        self.layers.append(dglnn.SAGEConv(in_feats=self.in_feats,
                                          out_feats=self.hidden_dim,
                                          aggregator_type='pool'))
                                          # aggregator_type = 'pool'))
        for l in range(1, (self.n_layers - 1)):
            self.layers.append(dglnn.SAGEConv(in_feats=self.hidden_dim,
                                              out_feats=self.hidden_dim,
                                              aggregator_type='pool'))
                                              # aggregator_type='pool'))
        self.layers.append(dglnn.SAGEConv(in_feats=self.hidden_dim,
                                          out_feats=self.n_classes,
                                          aggregator_type='pool'))
                                          # aggregator_type = 'pool'))

    def forward(self, blocks, features):
        h = features

        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)

        return h


class GraphConvModel(thnn.Module):

    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 norm,
                 activation,
                 dropout=0):
        super(GraphConvModel, self).__init__()
        self.in_feats = in_feats
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.norm = norm
        self.activation = activation
        self.dropout = thnn.Dropout(dropout)

        self.layers = thnn.ModuleList()

        # build multiple layers
        self.layers.append(dglnn.GraphConv(in_feats=self.in_feats,
                                           out_feats=self.hidden_dim,
                                           norm=self.norm,
                                           activation=self.activation,))
        for l in range(1, (self.n_layers - 1)):
            self.layers.append(dglnn.GraphConv(in_feats=self.hidden_dim,
                                               out_feats=self.hidden_dim,
                                               norm=self.norm,
                                               activation=self.activation))
        self.layers.append(dglnn.GraphConv(in_feats=self.hidden_dim,
                                           out_feats=self.n_classes,
                                           norm=self.norm,
                                           activation=self.activation))

    def forward(self, blocks, features):
        h = features

        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.dropout(h)

        return h


class GraphAttnModel(thnn.Module):

    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 heads,
                 activation,
                 feat_drop=0,
                 attn_drop=0
                 ):
        super(GraphAttnModel, self).__init__()
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.heads = heads
        self.feat_dropout = feat_drop
        self.attn_dropout = attn_drop
        self.activation = activation

        self.layers = thnn.ModuleList()

        # build multiple layers
        self.layers.append(dglnn.GATConv(in_feats=self.in_feats,
                                         out_feats=self.hidden_dim,
                                         num_heads=self.heads[0],
                                         feat_drop=self.feat_dropout,
                                         attn_drop=self.attn_dropout,
                                         activation=self.activation))

        for l in range(1, (self.n_layers - 1)):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layers.append(dglnn.GATConv(in_feats=self.hidden_dim * self.heads[l - 1],
                                             out_feats=self.hidden_dim,
                                             num_heads=self.heads[l],
                                             feat_drop=self.feat_dropout,
                                             attn_drop=self.attn_dropout,
                                             activation=self.activation))

        self.layers.append(dglnn.GATConv(in_feats=self.hidden_dim * self.heads[-2],
                                         out_feats=self.n_classes,
                                         num_heads=self.heads[-1],
                                         feat_drop=self.feat_dropout,
                                         attn_drop=self.attn_dropout,
                                         activation=None))

    def forward(self, blocks, features):
        h = features

        for l in range(self.n_layers - 1):
            h = self.layers[l](blocks[l], h).flatten(1)

        logits = self.layers[-1](blocks[-1],h).mean(1)

        return logits


class MultiHeadedAttention(thnn.Module):
    def __init__(self, num_attention_heads, hidden_size, attn_drop):
        super(MultiHeadedAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = thnn.Linear(hidden_size, self.all_head_size)
        self.key = thnn.Linear(hidden_size, self.all_head_size)
        self.value = thnn.Linear(hidden_size, self.all_head_size)
        self.dropout = thnn.Dropout(attn_drop)
        self.softmax = thnn.Softmax(dim=-1)

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

    
class LayerNorm(thnn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = thnn.Parameter(torch.ones(hidden_size))
        self.bias = thnn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
    
    
class GraphModel(thnn.Module):
    def __init__(self, 
                 model_type,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 mlp_dim,
                 num_attention_heads,
                 n_classes,
                 activation,
                 input_drop,
                 drop_rate,
                 attn_drop
                 ):
        super(GraphModel, self).__init__()
        
        self.label_embed = thnn.Embedding(n_classes + 1, in_feats, padding_idx=n_classes)
        self.input_drop = thnn.Dropout(input_drop)
        self.feature_mlp = thnn.Sequential(
            thnn.Linear(2 * in_feats, hidden_dim),
            LayerNorm(hidden_dim),
            thnn.ReLU(),
            thnn.Dropout(drop_rate),
            thnn.Linear(hidden_dim, in_feats),
        )
            
        self.model_type = model_type
        if self.model_type == 0:
            self.graphsage = GraphSageModel(in_feats, hidden_dim, n_layers, mlp_dim,
                                            activation=activation)
            self.graphconv = GraphConvModel(in_feats, hidden_dim, n_layers, mlp_dim,
                                            norm='both', activation=activation)
            self.graphattn = GraphAttnModel(in_feats, hidden_dim, n_layers, mlp_dim,
                                            heads=([5] * n_layers), activation=activation)
            self.attention = MultiHeadedAttention(num_attention_heads, mlp_dim, attn_drop)
            self.se1 = Se(mlp_dim * 3, mlp_dim)
            self.se2 = Se(mlp_dim * 3, mlp_dim)
            self.se3 = Se(mlp_dim * 3, mlp_dim)
        elif self.model_type == 1:
            self.graphsage = GraphSageModel(in_feats, hidden_dim, n_layers, mlp_dim,
                                            activation=activation)
        elif self.model_type == 2:
            self.graphconv = GraphConvModel(in_feats , hidden_dim, n_layers, mlp_dim,
                                            norm='both', activation=activation)
        elif self.model_type == 3: 
            self.graphattn = GraphAttnModel(in_feats * 2, hidden_dim, n_layers, mlp_dim,
                                            heads=([5] * n_layers), activation=activation)
       
        self.head = thnn.Sequential(
            thnn.Linear(mlp_dim, mlp_dim),
            LayerNorm(mlp_dim),
            thnn.ReLU(),
            thnn.Dropout(drop_rate),
            thnn.Linear(mlp_dim, n_classes),
        )
        self.fc = thnn.Linear(mlp_dim, n_classes)
        
    def forward(self, blocks, input_feats, input_labels):
        input_feats = self.input_drop(input_feats)
        
        input_labels = self.label_embed(input_labels)
        input_labels = self.input_drop(input_labels)
        
        feature = torch.cat([input_labels, input_feats], dim=1)
        feature = self.feature_mlp(feature)
        
        if self.model_type == 0:
            logits1 = self.graphsage(blocks, feature)
            logits2 = self.graphconv(blocks, feature)
            logits3 = self.graphattn(blocks, feature)
            # logits = torch.stack([logits1, logits2, logits3], dim=1)
            # logits = self.attention(logits)[:, 0]
            all_logits = torch.cat((logits1, logits2, logits3), dim=1)
            weight1 = self.se1(all_logits)
            weight2 = self.se2(all_logits)
            weight3 = self.se3(all_logits)
            logits = logits1 * weight1 + logits2 * weight2 + logits3 * weight3
            
        elif self.model_type == 1:
            logits = self.graphsage(blocks, feature)
        elif self.model_type == 2:
            logits = self.graphconv(blocks, feature)
        elif self.model_type == 3:
            logits = self.graphattn(blocks, feature)
        
        logits = self.fc(logits)
        return logits