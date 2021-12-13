from types import new_class
from dgl.dataloading.neighbor import MultiLayerNeighborSampler
import torch
from torch.utils.data import DataLoader, Dataset
import dgl

NID = '_ID'


class IdDataSet(Dataset):

    def __init__(self, nid) -> None:
        self.nid = nid

    def __len__(self):
        return len(self.nid)

    def __getitem__(self, index):
        return self.nid[index]


class InOutDataIter:

    def __init__(self, in_g, out_g, in_sampler, out_sampler, id_dataloader):
        self.in_g = in_g
        self.out_g = out_g
        self.in_sampler = in_sampler
        self.out_sampler = out_sampler
        self.id_dataloader = iter(id_dataloader)

    def __next__(self):
        seeds = next(self.id_dataloader)
        in_blocks = self.in_sampler.sample_blocks(self.in_g, seed_nodes=seeds)
        out_blocks = self.out_sampler.sample_blocks(self.out_g, seed_nodes=seeds)
        in_input_nodes = in_blocks[0].srcdata[NID]
        out_input_nodes = out_blocks[0].srcdata[NID]
        return in_input_nodes, in_blocks, out_input_nodes, out_blocks, seeds

    def __len__(self):
        return len(self.id_dataloader)


class InOutDataLaoder:

    def __init__(self, g, nid, in_fanout, out_fanout, batch_size, shuffle) -> None:
        self.in_sampler = MultiLayerNeighborSampler(in_fanout)
        self.out_sampler = MultiLayerNeighborSampler(out_fanout)
        self.id_dataloader = DataLoader(IdDataSet(nid=nid), batch_size=batch_size, shuffle=shuffle)
        self.in_g = g
        self.out_g = dgl.reverse(g)

    # 该方法必须返回一个迭代器
    def __iter__(self):
        return InOutDataIter(self.in_g, self.out_g, self.in_sampler, self.out_sampler, self.id_dataloader)

    def __len__(self):
        return len(self.id_dataloader)