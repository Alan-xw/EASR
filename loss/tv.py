import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class TV(nn.Module):
    def __init__(self):
        super(TV, self).__init__()

    def forward(self, x):
        def _tensor_size(t):
            return t.size()[1] * t.size()[2] * t.size()[3]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = _tensor_size(x[:, :, 1:, :])
        count_w = _tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return h_tv / count_h + w_tv / count_w
