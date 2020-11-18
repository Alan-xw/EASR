import torch.nn as nn
import torch


class CCA(nn.Module):
    def __init__(self, C, ratio=16):
        super(CCA, self).__init__()
        self.squeeze = nn.Conv2d(C, 1, 1, padding=0)
        self.squeeze_fn = nn.Softmax(dim=-1)
        self.excitation = nn.Sequential(*[
            nn.Conv2d(C, C // ratio, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // ratio, C, 1, padding=0),
            nn.Sigmoid()
        ])

    def spatial_squeeze(self, x):
        b, c, h, w = x.size()
        # squeeze
        input_x = x
        input_x = input_x.view(b, c, h * w)
        input_x = input_x.unsqueeze(1)
        var = x - x.mean(dim=[2, 3], keepdim=True)
        mask = self.squeeze(var)
        mask = mask.view(b, 1, h * w)
        mask = self.squeeze_fn(mask)
        mask = mask.unsqueeze(-1)
        squeeze = torch.matmul(input_x, mask)
        squeeze = squeeze.view(b, c, 1, 1)

        return squeeze

    def forward(self, x):
        # squeeze
        att = self.spatial_squeeze(x)
        # excitation
        att = self.excitation(att)
        x = x * att
        return x


class ESA(nn.Module):
    def __init__(self, Cin, KSize=1, ratio=16):
        super(ESA, self).__init__()
        self.ESA = nn.Sequential(*[
            nn.Conv2d(Cin, Cin // ratio, KSize, padding=(KSize - 1) // 2, stride=1),
            nn.Conv2d(Cin // ratio, Cin // ratio, 3, padding=2, stride=1, groups=Cin // ratio, dilation=2),
            nn.Conv2d(Cin // ratio, Cin // ratio, 3, padding=2, stride=1, groups=Cin // ratio, dilation=2),
            nn.Conv2d(Cin // ratio, 1, KSize, padding=(KSize - 1) // 2, stride=1),
            nn.Sigmoid()
        ])

    def forward(self, x):
        mean = x.mean(dim=[2, 3], keepdim=True)
        x = x - mean
        x = self.ESA(x)
        return x


class SA_squeeze(nn.Module):
    def __init__(self, SA_in, KSize=1, fu_by_conv=True):
        super(SA_squeeze, self).__init__()
        self.conv = nn.Sequential(*[
            nn.Conv2d(SA_in, 1, KSize, padding=(KSize - 1) // 2, stride=1),
            nn.ReLU(inplace=True)
        ])
        self.fu_by_conv = fu_by_conv

    def forward(self, x):
        x = torch.cat(x, dim=1)
        x = self.conv(x)
        return x


class PSA(nn.Module):
    def __init__(self, Cin, SA_in):
        super(PSA, self).__init__()
        self.SA = ESA(Cin)
        self.SA_in = SA_in
        if SA_in > 0:
            self.SA_squeeze = SA_squeeze(SA_in)

    def forward(self, x, prior_SAs):
        current_SA = self.SA(x)
        if self.SA_in > 0:
            prior_SA = self.SA_squeeze(prior_SAs)
            PSA = current_SA + prior_SA
        else:
            PSA = current_SA
        x = x * PSA
        return x, PSA
