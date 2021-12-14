import dgl
import copy
from dgl.dataloading.neighbor import MultiLayerNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader


src_nid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
dst_nid = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

src_all = src_nid.copy()
src_all.extend(dst_nid)
dst_all = dst_nid.copy()
dst_all.extend(src_nid)
print('src_nid', src_nid)
print('dst_nid', dst_nid)
print('src_all', src_all)
print('dst_all', dst_all)

graph = dgl.heterograph({
    ('paper', 'cite', 'paper'): (src_nid, dst_nid),
    ('paper', 'cited', 'paper'): (dst_nid, src_nid),
    ('paper', 'undirected', 'paper'): (src_all, dst_all),
})
print(graph)

import torch
graph2 = copy.copy(graph)
node_num = graph2.number_of_edges(etype='cite')
node_index = torch.arange(node_num)
rd_m = torch.rand(node_num)
drop_node = node_index[rd_m < 0.2]
print(node_num, drop_node)
print(graph2)
graph2.remove_edges(drop_node, 'cite')
graph2.remove_edges(drop_node, 'cited')
graph2.remove_edges(drop_node + node_num, 'undirected')
graph2.remove_edges(drop_node, 'undirected')
print(graph2)
'''
print(graph2.nodes())
print(graph2.number_of_edges(etype='cite'))
# graph2.remove_edges(torch.tensor([0, 1, 2, 3]), 'cite')
print(graph2.edges(etype='cite'))
# print(graph2)

nid = list(set(src_nid) | set(dst_nid))
sampler = MultiLayerNeighborSampler([2, 2])
dataloader = NodeDataLoader(graph, nid, sampler, batch_size=2, shuffle=True, drop_last=False)

for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
    print(dir(blocks[1]))
    print(dict(blocks[1]))
    print(blocks[1]['undirected'])
    break
    '''

print(torch.arange(6))