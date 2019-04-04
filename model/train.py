import os
import shutil

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from common.visualise import unnormalise


class Trainer(object):
    def __init__(self, model: nn.Module, optimiser: optim.Optimizer, criterion: _Loss, device: torch.device,
                 writer: SummaryWriter, outf: str):
        self.model = model
        self.optimiser = optimiser
        self.criterion = criterion
        self.device = device
        self.writer = writer
        self.outf = outf
        self.epochs = 0
        self.best_loss = 1e9
        self.best_model_wts = None
        self.len_data_loader = None
        self.prev_save_path = None
        self.results2save = (None, None, None)  # tensor of (data, predictions) respectively
        self.img_out_dir = os.path.join(self.outf, "images")
        self.metrics = {'loss': [], 'acc': []}

    def _eval(self, data_loader: DataLoader):  # todo: something is causing GPU mem overflows
        self.model.eval()

        acc_ttl, loss_ttl = 0, 0
        len_dataset = len(data_loader)
        for i, (data, mask, n_cells) in enumerate(data_loader):
            data, mask = data.to(self.device), mask.to(self.device)
            prediction = self.model(data)

            loss = self.criterion(prediction, mask)
            correct = (prediction == mask).sum().item()
            acc = correct / data_loader.batch_size
            acc_ttl += acc / len_dataset
            loss_ttl += loss / len_dataset

    def _train(self, data_loader: DataLoader, epoch: int):
        self.model.train()
        for i, (data, mask, n_cells) in enumerate(data_loader):
            data, mask = data.to(self.device), mask.to(self.device)

            prediction = self.model(data)

            self.optimiser.zero_grad()
            loss = self.criterion(prediction, mask)
            loss.backward()
            self.optimiser.step()

            correct = (prediction == mask).sum().item()
            self.metrics['acc'].append(correct / data_loader.batch_size)
            self.metrics['loss'].append(loss.item())

            self._log_metrics(epoch, i)
            if i == 0:
                self.results2save = (data, mask, prediction)

    def train(self, data_loader: DataLoader, epochs: int):
        self.epochs = epochs
        self.len_data_loader = len(data_loader)

        self.best_model_wts = self.model.state_dict()
        for epoch in range(epochs):
            self._train(data_loader, epoch)
            self._save_checkpoint()

            # self._eval(data_loader)

        self.model.load_state_dict(self.best_model_wts)
        return self.model

    def _save_checkpoint(self):
        if self.metrics['loss'][-1] < self.best_loss:
            self.best_loss = self.metrics['loss'][-1]
            self.best_model_wts = self.model.state_dict()

            save_path = '%s/net_%.3f.pth' % (self.outf, self.best_loss)
            if self.prev_save_path is not None and os.path.exists(self.prev_save_path):
                os.remove(self.prev_save_path)
            self.prev_save_path = save_path
            torch.save(self.model.state_dict(), save_path)

            shutil.rmtree(self.img_out_dir, ignore_errors=True)
            os.makedirs(self.img_out_dir, exist_ok=True)
            for i, (data, mask, pred) in enumerate(zip(*self.results2save)):
                image = np.hstack([
                    cv2.copyMakeBorder(unnormalise(x.detach()), 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255) for x in
                    [data, mask, pred]])
                cv2.imwrite(os.path.join(self.img_out_dir, '%s.png' % str(i)), image)

    def _log_metrics(self, epoch: int, iter: int):
        global_step = epoch * self.len_data_loader + iter

        # Log metrics
        metrics = {key: value[-1] for key, value in self.metrics.items()}  # get latest metric values
        self.writer.add_scalars("metrics", metrics, global_step)

        # Log metrics by epoch
        if iter == 0:
            self.writer.add_scalars("metrics/by_epoch", metrics, epoch)

        # Print metrics to terminal
        local_step = "[%i/%i][%i/%i]" % (epoch, self.epochs, iter, self.len_data_loader)
        formatted_metrics = ["%s: %.4f " % (key, value) for key, value in metrics.items()]
        print(' '.join([local_step, *formatted_metrics]))

        # # Log gradients
        # for name, param in self.model.named_parameters():
        #     self.writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step)
