import dgl
import pickle
import logging
import numpy as np
import torch as th

graph_path = 'graph.bin'
g = dgl.load_graphs(graph_path)[0][0]
print(g)

src_nid = list(g.edges()[0])
dst_nid = list(g.edges()[1])
src_all = src_nid.copy()
src_all.extend(dst_nid)
dst_all = dst_nid.copy()
dst_all.extend(src_nid)

print(len(src_nid), len(dst_nid), len(src_all), len(dst_all))
graph = dgl.heterograph({
    ('paper', 'cite', 'paper'): (src_nid, dst_nid),
    ('paper', 'cited', 'paper'): (dst_nid, src_nid),
    ('paper', 'undirected', 'paper'): (src_all, dst_all),
})
print(graph)

dgl.data.utils.save_graphs('heterograph.bin', [graph])