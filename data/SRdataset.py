import numpy as np
from PIL import Image
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
import glob
import os
from config import args
from data import common
import imageio
import utility


class SRdatasets(Dataset.Dataset):
    def __init__(self, args, name='DIV2K', train=True, transform=None):
        if train:
            self.dir_hr = args.dir_train + name + '_HR'
            self.dir_lr = args.dir_train + name + '_LR_bicubic'
#             self.dir_lr = args.dir_train + name + '_LR_BNoise'
            self.dir_lr = os.path.join(self.dir_lr, 'X{}'.format(args.scale))
        else:
            self.dir_hr = args.dir_test + name + '_HR'
            self.dir_hr = os.path.join(self.dir_hr, 'X{}'.format(args.scale))
            self.dir_lr = args.dir_test + name + '_LR_bicubic'
#             self.dir_lr = args.dir_test + name + '_LR_noise'
            self.dir_lr = os.path.join(self.dir_lr, 'X{}'.format(args.scale))
        self.train = train
        self.name = name
        self.transform = transform
        self.scale = args.scale
        self.format = '.png'

        self.hr, self.lr = self._scan()

        if train:
            n_patches = args.batch_size * args.test_every
            n_images = len(self.hr)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)
    def _scan(self):
        names_hr = sorted(glob.glob(os.path.join(self.dir_hr, '*' + self.format)))
        
        names_lr = []
        if self.train:
            for f in names_hr:
                filename, _ = os.path.splitext(os.path.basename(f))
                front,last = filename.split('_')
                # Noise
                # names_lr.append(os.path.join(self.dir_lr, '{}_noisy_{}x{}{}'.format(front,last, self.scale, self.format)))
                # Bicubic
                # names_lr.append(os.path.join(self.dir_lr, '{}x{}{}'.format(filename, self.scale, self.format)))
                names_lr.append(os.path.join(self.dir_lr, '{}{}'.format(filename, self.format)))
            return names_hr, names_lr
        else:
            for f in names_hr:
                filename, _ = os.path.splitext(os.path.basename(f))

                # names_lr.append(os.path.join(self.dir_lr, '{}_noisyx{}{}'.format(filename, self.scale, self.format)))
                # Bicubic
                names_lr.append(os.path.join(self.dir_lr, '{}{}'.format(filename, self.format)))
            return names_hr, names_lr

    def __len__(self):
        if self.train:
            return len(self.hr) * self.repeat
        else:
            return len(self.hr)

    def _name(self):
        return self.name
    def _get_index(self, idx):
        if self.train:
            return idx % len(self.hr)
        else:
            return idx

    def get_patch(self, lr, hr):
        if self.train:
            lr, hr = common.get_patch(
                lr, hr,
                patch_size=args.input_size,
                scale=args.scale,
                input_large=False
            )
            lr, hr = common.augment(lr, hr)  # 直接实现图像增强
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * args.scale, 0:iw * args.scale]

        return lr, hr

    def __getitem__(self, index):
        hr, lr = self.hr[index], self.lr[index]
        filename, _ = os.path.splitext(os.path.basename(lr))
        lr_img = imageio.imread(lr)
        hr_img = imageio.imread(hr)
        patchs = self.get_patch(lr_img, hr_img)
        patchs = common.set_channel(*patchs, n_channels=3)
        patchs_t = common.np2Tensor(*patchs, rgb_range=args.rgb_range)
        # lr_img = np.transpose(lr_img, (2, 0, 1))
        # hr_img = np.transpose(hr_img, (2, 0, 1))
        return [patchs_t[0], patchs_t[1], filename]


if __name__ == '__main__':

    DIV2K = SRdatasets(args, name='DIV2K_16', train=True)
    name_hr, name_lr = DIV2K._scan()
    print(name_lr[0])
    lr, hr,_ = DIV2K.__getitem__(0)
    lr = utility.quantize(lr, args.rgb_range)
    print(lr)


#     train_loader = DataLoader.DataLoader(DIV2K, batch_size=1, shuffle=True)
#     for i, item in enumerate(train_loader):
#         data_patch = item
#         # a = data_patch[0].numpy().squeeze().transpose((1, 2, 0)).astype(np.uint8)
#         Image.fromarray(
#             data_patch[0].numpy().squeeze().transpose((1, 2, 0)).astype(np.uint8)).show()
#         Image.fromarray(
#             data_patch[1].numpy().squeeze().transpose((1, 2, 0)).astype(np.uint8)).show()
#         #
#         # Image.fromarray(data_patch[1].numpy().squeeze()).show()
#         if i >= 0:
#             break
