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
    
    
class UniCMP(nn.Module):

    def __init__(self, input_size, num_class, num_layers=2, num_heads=4, hidden_size=512,
                 label_drop=0.1, feat_drop=0., attn_drop=0., drop=0.2, 
                 use_sage=True, use_conv=True, use_attn=True, use_resnet=True,
                 use_densenet=True
                ):
        super(UniCMP, self).__init__()
        self.num_layers = num_layers
        self.use_sage = use_sage
        self.use_conv = use_conv
        self.use_attn = use_attn
        self.use_resnet = use_resnet
        self.use_densenet = use_densenet
        
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
  
            self.gnn_attns.append(MultiHeadedAttention(num_heads, hidden_size, attn_drop))
            self.norm_layers.append(LayerNorm(hidden_size))
        
        self.shortcut = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            LayerNorm(hidden_size),
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
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.skips:
            linear_init_layers(m)
            
        for m in self.shortcut:
            if isinstance(m, nn.Linear):
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

        h, feat_dst = feature, feature
        for l in range(self.num_layers):     
            feat_dst = expand_as_pair(feat_dst, blocks[l])[1]
            hidden = []
            
            if self.use_resnet:
                h1 = expand_as_pair(h, blocks[l])[1]
                h1 = self.skips[l](h1)
                h1 = self.norm_layers[5 * l](h1)
                h1 = F.elu(h1)
                hidden.append(h1)
            
            if self.use_sage:
                h2 = self.graphsage[l](blocks[l], h)
                h2 = self.norm_layers[5 * l + 1](h2)
                h2 = F.elu(h2)
                hidden.append(h2)
            
            if self.use_conv:
                h3 = self.graphconv[l](blocks[l], h)
                h3 = self.norm_layers[5 * l + 2](h3)
                h3 = F.elu(h3)
                hidden.append(h3)

            if self.use_attn:
                h4 = self.graphattn[l](blocks[l], h).flatten(1)
                h4 = self.norm_layers[5 * l + 3](h4)
                h4 = F.elu(h4)
                hidden.append(h4)
            
            if self.use_densenet and l == self.num_layers - 1:
                h5 = self.shortcut(feat_dst)
                hidden.append(h5)

            h = torch.stack(hidden, dim=1)
            h = self.gnn_attns[l](h)[:, 0]
            h = self.norm_layers[5 * l + 4](h)
            h = self.dropout(h)

        output = self.head(h)
        return output
