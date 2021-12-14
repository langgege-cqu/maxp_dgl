import torch
import dgl


class MultiLayerInOutSampler(dgl.dataloading.BlockSampler):

    def __init__(self, in_fanouts, out_fanouts, return_eids=False, output_ctx=None):
        super().__init__(len(in_fanouts), return_eids=return_eids, output_ctx=output_ctx)
        self.in_fanouts = in_fanouts
        self.out_fanouts = out_fanouts

    def sample_frontier(self, block_id, g, seed_nodes):
        # 出入分别采样
        in_fanouts, out_fanouts = self.in_fanouts[block_id], self.out_fanouts[block_id]
        in_frontier = dgl.sampling.sample_neighbors(g, seed_nodes, in_fanouts, edge_dir='in')
        out_frontier = dgl.sampling.sample_neighbors(g, seed_nodes, out_fanouts, edge_dir='out')
        in_src, in_det = in_frontier.all_edges()
        out_src, out_det = out_frontier.all_edges()
        src = torch.cat((in_src, out_det), dim=-1)
        det = torch.cat((in_det, out_src), dim=-1)
        frontier = dgl.graph((src, det), num_nodes=g.number_of_nodes())
        return frontier

    def __len__(self):
        return self.num_layers