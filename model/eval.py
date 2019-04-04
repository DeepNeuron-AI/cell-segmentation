import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader


class Eval(object):
    def __init__(self, model: nn.Module, criterion: _Loss, device: torch.device):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.metrics = {'loss': []}  # todo update

    def evaluate(self, data_loader: DataLoader):
        self.model.eval()

        losses = []
        len_dataset = len(data_loader)
        for i, (data, mask, n_cells) in enumerate(data_loader):
            data, mask = data.to(self.device), mask.to(self.device)
            prediction = self.model(data)
            loss = self.criterion(prediction, mask)

            losses.append(loss.item())
            print("[%i/%i] loss: %.4f " % (i, len_dataset, losses[-1]))
        print("Average loss: .4f" % (np.average(np.array(losses))))
