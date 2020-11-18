import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Edge(nn.Module):
    def __init__(self):
        super(Edge, self).__init__()
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        conv_x = np.array([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        conv_y = np.array([[-1.0, -2, -1], [0, 0, 0], [1, 2, 1]])
        self.conv_x.weight = nn.Parameter(torch.from_numpy(conv_x).float().unsqueeze(0).unsqueeze(0))
        self.conv_y.weight = nn.Parameter(torch.from_numpy(conv_y).float().unsqueeze(0).unsqueeze(0))
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, sr, hr):
        tv_sr = self.get_gre(sr)
        tv_hr = self.get_gre(hr)
        edge_loss = F.l1_loss(tv_sr, tv_hr)
        return edge_loss

    def get_gre(self, x):
        # x = self.convert_gray(x)
        print(x.shape)
        grd_x = self.conv_x(x)
        grd_y = self.conv_y(x)
        out = torch.sqrt(grd_x ** 2 + grd_y ** 2) / 2
        return out

    def convert_gray(self, x):
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x = x.mul(convert).sum(dim=1).unsqueeze(1)
        return x
