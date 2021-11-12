import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn as dglnn
from dgl.utils import expand_as_pair

class Se(nn.Module):
    def __init__(self, feature_1, feature_2):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(feature_1, feature_2),
            nn.BatchNorm1d(feature_2),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        return self.se(features)

class GraphSageModel(nn.Module):

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
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList()

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


class Mul(nn.Module):
    def __init__(self, feature_size_1, feature_size_2):
        super().__init__()
        # 3 se
        self.se1 = Se(feature_size_1*2, feature_size_2)
        self.se2 = Se(feature_size_1*2, feature_size_2)
        # self.se3 = Se(feature_size_1*2, feature_size_2)
    
    def forward(self, features_1, features_2):
        features = torch.cat((features_1, features_2), dim=1)

        weight_1 = self.se1(features)
        weight_2 = self.se2(features)
        # weight_3 = self.se3(features)
        
        # features_1 = torch.nn.functional.normalize(features_1, p=2, dim=1)
        # features_2 = torch.nn.functional.normalize(features_2, p=2, dim=1)
        # features_3 = torch.nn.functional.normalize(features_3, p=2, dim=1)
        
        features = features_1 * weight_1 + features_2 * weight_2 

        return features

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
    
    
class GNNModel(nn.Module):

    def __init__(self, input_size, num_class, num_layers=2, num_heads=4, hidden_size=512,
                 label_drop=0.2, feat_drop=0., attn_drop=0., drop=0.1):
        super(GNNModel, self).__init__()
        self.num_layers = num_layers
        
        self.skips = nn.ModuleList()
        self.graphsage = nn.ModuleList()
        self.graphconv = nn.ModuleList()
        self.graphattn = nn.ModuleList()
        self.gnn_attns = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.mul = Mul(hidden_size, hidden_size)

        
        
        for i in range(num_layers):
            if i == 0:
                in_feats = input_size
            else:
                in_feats = hidden_size
            
            self.skips.append(nn.Linear(in_feats, hidden_size))
            self.norm_layers.append(LayerNorm(hidden_size))
            
            self.graphsage.append(
                dglnn.SAGEConv(in_feats=in_feats, 
                               out_feats=hidden_size, 
                               aggregator_type='lstm'))
            self.norm_layers.append(LayerNorm(hidden_size))
            
            self.graphconv.append(
                dglnn.GraphConv(in_feats=in_feats,
                                out_feats=hidden_size,
                                norm='both'))
            self.norm_layers.append(LayerNorm(hidden_size))
            
            self.graphattn.append(
                dglnn.GATConv(in_feats=in_feats,
                              out_feats=hidden_size // num_heads,
                              num_heads=num_heads,
                              attn_drop=attn_drop))
            self.norm_layers.append(LayerNorm(hidden_size))
            
            self.gnn_attns.append(nn.Linear(hidden_size, 1))
            self.norm_layers.append(LayerNorm(hidden_size))

            # self.mul.append(Mul(hidden_size, hidden_size))
            
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
        self.fc = nn.Linear(hidden_size, num_class)

        self.dropout = nn.Dropout(drop)
        self.label_dropout = nn.Dropout(label_drop)
        self.feat_dropout = nn.Dropout(feat_drop)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.skips:
            linear_init_layers(m)
        
        for m in self.gnn_attns:
            linear_init_layers(m)
            
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

        h = feature
        h2 = h
        for l in range(self.num_layers):  
            h2 = self.graphsage[l](blocks[l], h2)
            h2 = self.norm_layers[5 * l + 1](h2)
            h2 = F.elu(h2) 

        for l in range(self.num_layers):     
            h1 = expand_as_pair(h, blocks[l])[1]
            h1 = self.skips[l](h1)
            h1 = self.norm_layers[5 * l](h1)
            h1 = F.elu(h1)
            
            
            # h3 = self.graphconv[l](blocks[l], h)
            # h3 = self.norm_layers[5 * l + 2](h3)
            # h3 = F.elu(h3)
            
            h4 = self.graphattn[l](blocks[l], h).flatten(1)
            h4 = self.norm_layers[5 * l + 3](h4)
            h4 = F.elu(h4)
            
            # h = torch.stack([h1, h2, h3, h4], dim=1)
            h = torch.stack([h1, h4], dim=1)
            attn_weights = F.softmax(self.gnn_attns[l](h), dim=1)
            attn_weights = attn_weights.transpose(-1, -2)
            h = torch.bmm(attn_weights, h)[:, 0]
            # 上面是unimp模型的残差连接
            # 下面是se的残差连接
            # h = self.mul[l](h1, h2, h4)
            
            h = self.norm_layers[5 * l + 4](h)
            h = self.dropout(h)

        h = self.mul(h2, h)
        output = self.head(h)
        return output
