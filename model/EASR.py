import torch.nn as nn
import torch
from model import common
from model.attention import *


def make_model(args, parent=False):
    return EASR(args)

class EARB(nn.Module):
    def __init__(
            self, conv, n_feat, SA_in, kernel_size=3, ratio=16,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(EARB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        self.CA = CCA(n_feat, ratio)
        self.SA = PSA(n_feat, SA_in)
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x, prior_SAs):
        res = self.body(x)
        res_CA = self.CA(res)
        res, sa_f = self.SA(res_CA, prior_SAs)
        res += x
        return res, sa_f

class RIRGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, n_resblocks):
        super(RIRGroup, self).__init__()
        self.n_resblocks = n_resblocks
        self.body = nn.ModuleList()
        for i in range(n_resblocks):
            self.body.append(
                EARB(conv, n_feat, SA_in=i, kernel_size=kernel_size)
            )

    def forward(self, x):
        SA_maps = []
        res = x
        for i in range(self.n_resblocks):
            res, sa_f = self.body[i](res, SA_maps)
            SA_maps.append(sa_f)
        res += x
        return res


class GFF_unit(nn.Module):
    def __init__(self, n_feat, feat_in):
        super(GFF_unit, self).__init__()
        self.conv = nn.Conv2d(n_feat * feat_in, n_feat, 1, padding=0)

    def forward(self, current, GF_list):
        GF_concat = torch.cat(GF_list, dim=1)
        GF_concat = torch.cat([current, GF_concat], dim=1)
        out = self.conv(GF_concat)
        return out

class EASR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EASR, self).__init__()
        self.n_resgroups = args.n_resgroups  # 8
        self.n_resblocks = args.n_resblocks  # 16
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale

        self.sub_mean = common.MeanShift(args.rgb_range)

        # define head module
        modules_IFE =[conv(args.n_colors, n_feats, kernel_size)]
        modules_head = [
            conv(n_feats, n_feats, kernel_size)]
        # define body module
        modules_body = nn.ModuleList()
        modules_GFF = nn.ModuleList()
        for i in range(self.n_resgroups):
            modules_body.append(RIRGroup(conv, n_feats, kernel_size, n_resblocks=self.n_resblocks))
            modules_GFF.append(GFF_unit(n_feats, feat_in=(i+2)))

        modules_conv = [conv(n_feats, n_feats, kernel_size)]
        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        self.IFE = nn.Sequential(*modules_IFE)
        self.head = nn.Sequential(*modules_head)
        self.body = modules_body
        self.GFF = modules_GFF
        self.conv = nn.Sequential(*modules_conv)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        GFF_list = []
        x = self.sub_mean(x)
        IFE_x = self.IFE(x)
        x = self.head(IFE_x)
        GFF_list.append(x)
        for i in range(self.n_resgroups):
            res = self.body[i](GFF_list[-1])
            res = self.GFF[i](res, GFF_list)
            GFF_list.append(res)

        res = self.conv(GFF_list[-1])
        res += IFE_x
        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
