import numpy as np
import torch
import time
import dgl

from utils import shuffle_walks


def find_connected_nodes(G):
    nodes = G.out_degrees().nonzero().squeeze(-1)
    return nodes


class DeepwalkDataset:
    def __init__(self, net_file, walk_length, window_size, num_walks, batch_size,
                 negative=5, gpus=[0], fast_neg=True):
        self.walk_length = walk_length
        self.window_size = window_size
        self.num_walks = num_walks
        self.batch_size = batch_size
        self.negative = negative
        self.num_procs = len(gpus)
        self.fast_neg = fast_neg

        graph = dgl.load_graphs(net_file)[0][0]
        self.G = dgl.to_bidirected(graph, copy_ndata=True)
        self.num_nodes = self.G.number_of_nodes()

        # random walk seeds
        start = time.time()
        self.valid_seeds = find_connected_nodes(self.G)
        if len(self.valid_seeds) != self.num_nodes:
            print('WARNING: The node ids are not serial. Some nodes are invalid.')

        seeds = torch.cat([torch.LongTensor(self.valid_seeds)] * num_walks)
        self.seeds = torch.split(shuffle_walks(seeds),
                                 int(np.ceil(len(self.valid_seeds) * self.num_walks / self.num_procs)),
                                 0)
        end = time.time()
        t = end - start
        print('%d seeds in %.2fs' % (len(seeds), t))

        # negative table for true negative sampling
        if not fast_neg:
            node_degree = self.G.out_degrees(self.valid_seeds).numpy()
            node_degree = np.power(node_degree, 0.75)
            node_degree /= np.sum(node_degree)
            node_degree = np.array(node_degree * 1e8, dtype=np.int)
            self.neg_table = []

            for idx, node in enumerate(self.valid_seeds):
                self.neg_table += [node] * node_degree[idx]
            self.neg_table_size = len(self.neg_table)
            self.neg_table = np.array(self.neg_table, dtype=np.long)
            del node_degree

    def create_sampler(self, i):
        """ create random walk sampler """
        return DeepwalkSampler(self.G, self.seeds[i], self.walk_length)


class DeepwalkSampler(object):
    def __init__(self, G, seeds, walk_length):
        self.G = G
        self.seeds = seeds
        self.walk_length = walk_length

    def sample(self, seeds):
        walks = dgl.sampling.random_walk(self.G, seeds, length=self.walk_length - 1)[0]
        return walks