import os
from torchvision.models import vgg16, VGG16_Weights
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys
import torch

from model import Model
from dataset import KITTIDataset, TestSet
from attack import AttackWrapper
import args


class Main:
    def __init__(self):
        model = Model(args.num_classes)
        self.attack_frame = AttackWrapper(model, args.attack_lr, args.attack_iter,
                                          args.benign_ratio, args.benign_weight, args.clip_grad)
        self.attack_frame.to(args.device)
        self.to_pil_image = ToPILImage()

    def attack(self):
        dataset = KITTIDataset(args.data_dir, args.img_size)
        dataloader = DataLoader(dataset, args.batch_size, shuffle=False)
        num_success = 0
        idx = 0
        for i, (img, label) in enumerate(dataloader):
            print(f'processing batch {i}')
            x = img.to(args.device)
            y = label.to(args.device)
            target_y = self.setAdvClass(y.clone())
            adv_x, success = self.attack_frame(x, y, target_y)
            num_success += success.sum().item()
            print(f'batch attack success rate: {100 * success.sum().item() / args.batch_size}')
            for j, adv_sample in enumerate(adv_x):
                if args.out_dir is not None:
                    if not os.path.exists(args.out_dir):
                        os.makedirs(args.out_dir)
                    max_pixel = adv_sample.max(dim=1)[0].max(dim=1)[0].unsqueeze(1).unsqueeze(1)
                    min_pixel = adv_sample.min(dim=1)[0].min(dim=1)[0].unsqueeze(1).unsqueeze(1)
                    adv_sample = (adv_sample - min_pixel) / (max_pixel - min_pixel)
                    self.to_pil_image(adv_sample).save(os.path.join(args.out_dir,
                        f'{idx}_ori{y[j].item()}_adv{target_y[j].item()}_success{success[j].item()}.png'))
                idx += 1
        print(f'global attack success rate: {100 * num_success / len(dataset)}')

    def test(self):
        dataset = TestSet(args.test_dir, args.img_size, args.conds,
                          args.illumination_strength, args.illumination_radius_ratio,
                          args.size_ratio,
                          args.gaussian_amp, args.snp_ratio)
        dataloader = DataLoader(dataset, args.batch_size, shuffle=False)
        num_survive = 0
        with tqdm(dataloader, dynamic_ncols=True) as loader:
            for original_img, cond_img in loader:
                original_x = original_img.to(args.device)
                cond_x = cond_img.to(args.device)
                original_pred = self.attack_frame.model(original_x)
                original_pred = original_pred.argmax(dim=1)
                cond_pred = self.attack_frame.model(cond_x)
                cond_pred = cond_pred.argmax(dim=1)
                num_survive += (original_pred == cond_pred).float().sum().item()
        print(f'survival rate under conditions {args.conds}: {100 * num_survive / len(dataset)}')

    def setAdvClass(self, y):
        assert args.adv_class in ['plus1', 'minus1', 'random']
        if args.adv_class == 'plus1':
            return (y + 1) % args.num_classes
        elif args.adv_class == 'minus1':
            y -= 1
            y[y < 0] += args.num_classes
            return y
        else:
            for i in y.size(0):
                gt = y[i].item()
                while y[i] == gt:
                    gt = np.random.randint(args.num_classes)
                y[i] = gt
            return y


if __name__ == "__main__":
    assert len(sys.argv) == 2, f'Number of cmd parameters {len(sys.argv) - 1} not supported'
    assert sys.argv[1] in ['attack', 'test'], f'Mode {sys.argv[1]} not supported'
    main = Main()
    if sys.argv[1] == 'attack':
        main.attack()
    else:
        main.test()

