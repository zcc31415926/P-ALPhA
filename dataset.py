from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import os


def isValidBBox(l, t, r, b):
    if l < 0 or t < 0 or r < 0 or b < 0:
        return False
    elif l >= r or t >= b:
        return False
    else:
        return True


class KITTIDataset(Dataset):
    def __init__(self, root, img_size=224):
        super().__init__()
        image_dir = os.path.join(root, 'kitti_data/training/image_2')
        label_dir = os.path.join(root, 'kitti_data/training/label_2')
        num_data = len(os.listdir(label_dir))
        images = [os.path.join(image_dir, f'{str(i).zfill(6)}.png') for i in range(num_data)]
        labels = [os.path.join(label_dir, f'{str(i).zfill(6)}.txt') for i in range(num_data)]
        self.images = []
        self.bboxes = []
        self.labels = []
        self.target_classes = ['Car', 'Pedestrian', 'Cyclist']
        for i in range(len(images)):
            label_file = labels[i]
            with open(label_file, 'r') as f:
                for line in f.readlines():
                    contents = line.strip().split(' ')
                    if contents[0] in self.target_classes:
                        left, top, right, bottom = np.array(contents[4 : 8]).astype(np.float64).astype(np.int)
                        if isValidBBox(left, top, right, bottom):
                            self.images.append(images[i])
                            self.bboxes.append([left, top, right, bottom])
                            self.labels.append(self.target_classes.index(contents[0]))
        self.transform = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        w, h = img.size
        l, t, r, b = self.bboxes[idx]
        l = max(l, 0)
        t = max(t, 0)
        r = min(r, w)
        b = min(b, h)
        img = np.array(img)[t : b, l : r]
        img = self.transform(Image.fromarray(img))
        return img, self.labels[idx]

    def __len__(self):
        return len(self.images)


class TestSet(Dataset):
    def __init__(self, root, img_size=224, conds=['size'],
                 illumination_strength=[100, 200], illumination_radius_ratio=[0.2, 0.3],
                 size_ratio=[0.25, 0.875],
                 gaussian_amp=[0.02, 0.03], snp_ratio=[1e-4, 3e-4]):
        super().__init__()
        # filename: {idx}_ori{O}_adv{A}_success{S}.png
        # file[-5] (S): boolean token of attack successfulness
        self.images = [os.path.join(root, file) for file in os.listdir(root) if file[-5] == '1']
        self.illumination_strength = illumination_strength
        self.illumination_radius_ratio = illumination_radius_ratio
        self.size_ratio = size_ratio
        self.gaussian_amp = gaussian_amp
        self.snp_ratio = snp_ratio
        cond_funcs = {
            'illumination': self.imbalancedIllumination,
            'size': self.longDistance,
            'noise': self.stochasticError,
        }
        self.original_transform = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.cond_transform = []
        for cond in conds:
            self.cond_transform.append(cond_funcs[cond])
        self.cond_transform += [
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
        self.cond_transform = transforms.Compose(self.cond_transform)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        original_img = self.original_transform(img)
        cond_img = self.cond_transform(img)
        return original_img, cond_img

    def __len__(self):
        return len(self.images)

    def imbalancedIllumination(self, x):
        strength = np.random.rand() * (self.illumination_strength[1] - self.illumination_strength[0]) + \
            self.illumination_strength[0]
        radius = np.random.rand() * (self.illumination_radius_ratio[1] - self.illumination_radius_ratio[0]) + \
            self.illumination_radius_ratio[0]
        x = np.array(x)
        h, w, c = x.shape
        radius *= np.min(h, w)
        center = np.random.randint(w)
        for i in range(h):
            for j in range(w):
                dist = np.sqrt((j - center) ** 2 + i ** 2)
                if dist < radius / 2:
                    x[i, j] = np.min(x[i, j] + strength * (1 - dist / radius), 255)
                else:
                    x[i, j] = np.min(x[i, j] + strength * radius / dist / 4, 255)
        return Image.fromarray(x)

    def longDistance(self, x):
        ratio = np.random.rand() * (self.size_ratio[1] - self.size_ratio[0]) + self.size_ratio[0]
        w, h = x.size
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        new_x = x.resize((new_w, new_h))
        x = new_x.resize((w, h))
        return x

    def stochasticError(self, x):
        x = np.array(x)
        gaussian_amp = np.random.rand() * (self.gaussian_amp[1] - self.gaussian_amp[0]) + self.gaussian_amp[0]
        snp_ratio = np.random.rand() * (self.snp_ratio[1] - self.snp_ratio[0]) + self.snp_ratio[0]
        gaussian_noise = np.random.randn_like(x) * gaussian_amp
        random_map = np.random.rand_like(x[..., 0])
        snp_mask = (random_map <= self.snp_ratio)
        random_map = np.random.randn_like(x[..., 0])
        snp_amp_mask = (random_map > 0).astype(np.float32) * 255 * 2 - 255
        x[snp_mask] += snp_amp_mask[snp_mask]
        x += gaussian_noise
        x[x > 255] = 255
        x[x < 0] = 0
        return Image.fromarray(x)

