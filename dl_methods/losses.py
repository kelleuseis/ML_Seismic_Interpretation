'''Loss functions modified from TransUNet repository (https://github.com/Beckschen/TransUNet) and boundary-loss repository (https://github.com/LIVIAETS/boundary-loss)'''

import numpy as np
import torch
from torch import Tensor, einsum
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom
from typing import List, cast, Tuple, Callable, Union, Set, Iterable, cast


def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

def simplex(t: Tensor, axis=1) -> bool:
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


class DiceLoss(nn.Module):
    def __init__(self, n_classes, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            if i != self.ignore_index:
                temp_prob = input_tensor == i
                tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            if i != self.ignore_index:
                dice = self._dice_loss(inputs[:, i], target[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        if self.ignore_index is not None:
            return loss / (self.n_classes - 1)
        else:
            return loss / self.n_classes


class BoundaryLoss(torch.nn.Module):
    def __init__(self, idc):
        super(BoundaryLoss, self).__init__()
        self.idc = idc

    def _boundary_loss(self, probs, dist_maps):
        multipled = einsum("bkwh,bkwh->bkwh", probs, dist_maps)
        loss = multipled.mean()
        return loss

    def forward(self, probs, dist_maps, softmax=False):
        if softmax:
            probs = torch.softmax(probs, dim=1)
            
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)
        loss = self._boundary_loss(pc, dc)
        return loss
    
    
    
class GeneralizedDiceLoss(nn.Module):
    def __init__(self, idc):
        super(GeneralizedDiceLoss, self).__init__()
        self.idc = idc

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(len(self.idc)):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, inputs, target, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        
        pc = inputs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)
        
        w = 1 / ((einsum("bkwh->bk", tc).type(torch.float32) + 1e-10) ** 2).sum(dim=0)

        intersection = w * einsum("bkwh,bkwh->bk", pc, tc)
        union = w * (einsum("bkwh->bk", pc) + einsum("bkwh->bk", tc))

        divided = 1 - 2 * (einsum("bk->b", intersection) + 1e-10) / (einsum("bk->b", union) + 1e-10)

        loss = divided.mean()

        return loss

    
def consistency_loss(pred, pred_aug, ignore_value=-1):
    loss = F.mse_loss(pred, pred_aug, reduction='none')
    mask = pred.ne(ignore_value)
    loss = loss * mask.float()
    return loss.sum() / mask.float().sum()