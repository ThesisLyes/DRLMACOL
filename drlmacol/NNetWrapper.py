import time

import torch
from tqdm import tqdm

from torch import optim
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

from drlmacol.InvariantColorNNet import InvariantColorNNet


class AverageMeter:
    """Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class NNetWrapper:
    def __init__(self, size, k, dropout, remix, verbose, layers_size):
        self.size = size
        self.k = k
        self.verbose = verbose
        self.nnet = InvariantColorNNet(size, k, dropout, remix, layers_size)
        self.device = ""

    def set_to_device(self, device):
        self.device = device
        self.nnet.to(device)

    def parameters(self):
        return self.nnet.parameters()

    def predict(self, graph, convert=False):
        graph = graph.to(self.device)
        self.nnet.eval()
        with torch.no_grad():
            c = self.nnet(graph)
        if convert:
            return c.data.cpu().numpy()
        return c.data

    def forward_batch(self, all_graphs):
        batch_size = len(all_graphs)
        self.nnet.eval()
        dataloader = DataLoader(
            all_graphs, batch_size=batch_size, shuffle=False)
        data_time = AverageMeter()
        batch_time = AverageMeter()
        end = time.time()
        batch_idx = 0
        all_expected_fit = []
        for sample_batched in tqdm(dataloader):
            graphs = sample_batched
            graphs = graphs.to(self.device)
            data_time.update(time.time() - end)

            # print(graphs.shape)

            out_fit = self.nnet(graphs)
            all_expected_fit.extend(out_fit)
            batch_time.update(time.time() - end)
            end = time.time()
            batch_idx += 1
        outputs = torch.cat(all_expected_fit, dim=0)
        log_output = F.log_softmax(outputs, 0)
        prob_output = F.softmax(outputs, 0)
        return log_output, prob_output

    def loss_fit(self, targets, outputs):
        return torch.mean((targets - outputs) ** 2)
