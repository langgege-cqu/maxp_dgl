import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn as dglnn
from dgl.utils import expand_as_pair


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
    
    
class GNNModel(nn.Module):

    def __init__(self, input_size, num_class, num_layers=2, num_heads=4, hidden_size=512,
                 label_drop=0.3, feat_drop=0., attn_drop=0., drop=0.5):
        super(GNNModel, self).__init__()
        self.num_layers = num_layers
        
        self.skips = nn.ModuleList()
        self.graphsage = nn.ModuleList()
        self.graphconv = nn.ModuleList()
        self.graphattn = nn.ModuleList()
        self.gnn_attns = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
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
                               aggregator_type='pool'))
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

        h = feature
        for l in range(self.num_layers):     
            h1 = expand_as_pair(h, blocks[l])[1]
            h1 = self.skips(h1)
            h1 = self.norm_layers[5 * l](h1)
            h1 = F.elu(h1)
            
            # h2 = self.graphsage[l](blocks[l], h)
            # h2 = self.norm_layers[5 * l + 1](h2)
            # h2 = F.elu(h2)
            
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
            
            h = self.norm_layers[5 * l + 4](h)
            h = self.dropout(h)

        output = self.head(h)
        return output
