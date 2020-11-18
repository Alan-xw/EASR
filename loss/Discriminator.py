import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utility


class Discriminator(nn.Module):
    def __init__(self,d=64):
        super(Discriminator, self).__init__()
        n_colors = 3

        def _block(_in_channels, _out_channels, kernel_size=4, stride=1):

            block = [nn.Conv2d(
                    _in_channels,
                    _out_channels,
                    kernel_size,
                    padding=1,
                    stride=stride,
                    bias=False
                )]
            block.append(nn.InstanceNorm2d(_out_channels, momentum=0.9))
            block.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            return nn.Sequential(*block)
        
        self.down_1 = _block(n_colors, d, stride=2)
        self.down_2 = _block(d, d*2, stride=2)
        self.down_3 = _block(d*2, d*4, stride=2)
        self.down_4 = _block(d*4, d*8, stride=2)
        self.validity = nn.Conv2d(d*8, 1, 4, stride=1, padding=1, bias=False)
        self.down_conv = nn.Sequential(
            self.down_1,
            self.down_2,
            self.down_3,
            self.down_4,
            self.validity
        )

    def forward(self, x):
        d_x = self.down_conv(x)
        return d_x
