# https://github.com/EKami/carvana-challenge/blob/original_unet/src/nn/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_loss(pred: torch.Tensor, target: torch.Tensor, metrics=None, bce_weight=0.5):
    pred, target = pred.double(), target.double()
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    # metrics['bce'].append(bce.data.cpu().numpy() * target.size(0)))
    # metrics['dice'].append(dice.data.cpu().numpy() * target.size(0))
    # metrics['loss'].append(loss.data.cpu().numpy() * target.size(0))

    return loss


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


class BinaryCrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        """
        Binary cross entropy loss 2D
        Args:
            weight:
            size_average:
        """
        super().__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs = F.sigmoid(logits)
        probs_flat = probs.view(-1)  # Flatten
        targets_flat = targets.view(-1)  # Flatten
        return self.bce_loss(probs_flat, targets_flat)


class SoftDiceLoss(nn.Module):
    """ Sørensen–Dice coefficient """

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, logits, targets):
        smooth = 1
        num = targets.size(0)
        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score
