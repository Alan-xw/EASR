import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utility
from loss import Discriminator
from config import args
from types import SimpleNamespace

class GAN(nn.Module):
    def __init__(self, args,gan_type='WGAN'):
        super(GAN, self).__init__()
        self.dis = Discriminator.Discriminator()
        self.gan_k = args.gan_k
        if gan_type == 'WGAN_GP':
            optim_dict = {
                'optimizer': 'ADAM',
                'betas': (0, 0.9),
                'epsilon': 1e-8,
                'lr': 1e-5,
                'weight_decay': args.weight_decay,
                'decay': args.decay,
                'gamma': args.gamma}
            optim_args = SimpleNamespace(**optim_dict)
        else:
            optim_args = args
        self.optimizer = utility.make_optimizer(optim_args, self.dis)
        self.gan_type = gan_type
        self.sigmoid = nn.Sigmoid()

    def forward(self, sr, hr):
        self.loss = 0
        fake_detach = sr.detach()
        fake_input = sr
        real_input = hr
        for _ in range(self.gan_k): # discriminator training multi-times D 训练多次，G只训练一次。
            self.optimizer.zero_grad()
            retain_graph = False
            d_fake = self.dis(fake_detach)
            d_real = self.dis(real_input)
            # Discriminator update
            if self.gan_type.find('WGAN')>=0:
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand(1).to(torch.device('cuda'))
                    hat = fake_detach.mul(1 - epsilon) + real_input.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.dis(hat)
                    gradients = torch.autograd.grad(
                        outputs=d_hat.sum(), inputs=hat,
                        retain_graph=True, create_graph=True, only_inputs=True)[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty

            elif self.gan_type == 'ESRGAN':
                better_real = d_real - d_fake.mean(dim=0, keepdim=True)
                better_fake = d_fake - d_real.mean(dim=0, keepdim=True)
                loss_d = self.bce_loss(better_fake,better_real)
                retain_graph = True
            elif self.gan_type == 'RLSGAN':
                better_real = d_real - d_fake.mean(dim=0, keepdim=True)
                better_fake = d_fake - d_real.mean(dim=0, keepdim=True)
                label_one = torch.ones_like(better_fake)
                loss_d = F.mse_loss(better_real,label_one)+ F.mse_loss(better_fake,-label_one)
                retain_graph = True
            else:
                loss_d = self.bce_loss(d_fake, d_real)

            self.loss += loss_d.item()
            loss_d.backward(retain_graph=True)
            self.optimizer.step()

            if self.gan_type == 'WGAN':
                for p in self.dis.parameters():
                    p.data.clamp_(-1, 1)
        self.loss /= self.gan_k

        # Generator updating： The loss can also use the loss_d as back propagation loss for generator.
        d_fake_bp = self.dis(fake_input)
        if self.gan_type.find('WGAN')>=0:
            loss_g = -d_fake_bp.mean()
        elif self.gan_type == 'ESRGAN':
            better_real = d_real - d_fake_bp.mean(dim=0, keepdim=True)
            better_fake = d_fake_bp - d_real.mean(dim=0, keepdim=True)
            loss_g = self.bce_loss(better_fake, better_real)
        elif self.gan_type == 'RLSGAN':
            better_real = d_real - d_fake_bp.mean(dim=0, keepdim=True)
            better_fake = d_fake_bp - d_real.mean(dim=0, keepdim=True)
            label_one = torch.ones_like(better_fake)
            loss_g = loss_d = F.mse_loss(better_fake,label_one)+ F.mse_loss(better_real,-label_one)
        else:
            label_real = torch.ones_like(d_fake_bp)
            loss_g = F.binary_cross_entropy_with_logits(d_fake_bp, label_real)
        return loss_g

    def bce_loss(self, d_fake, d_real):
        # binary_cross_entropy_loss
        label_fake = torch.zeros_like(d_fake)
        label_real = torch.ones_like(d_real)
        d_fake, d_real = self.sigmoid(d_fake), self.sigmoid(d_real)
        dLossfake = F.binary_cross_entropy_with_logits(d_fake, label_fake)
        dLossreal = F.binary_cross_entropy_with_logits(d_real, label_real)
        dLoss = dLossfake + dLossreal
        return dLoss
        

