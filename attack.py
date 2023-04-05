import torch
import torch.nn as nn
from tqdm import tqdm


class AttackWrapper:
    def __init__(self, model, attack_lr=1e-3, attack_iter=500,
                 benign_ratio=0.8, benign_weight=0.2, clip_grad=10):
        self.model = model
        self.model.eval()
        self.attack_lr = attack_lr
        self.attack_iter = attack_iter
        self.benign_ratio = benign_ratio
        self.benign_weight = benign_weight
        self.clip_grad = clip_grad
        self.cls_criterion = nn.CrossEntropyLoss()

    def __call__(self, x, y, target_y):
        with tqdm(range(self.attack_iter)) as loader:
            for i in loader:
                x.requires_grad_(True)
                pred = self.model(x)
                attack_loss = self.cls_criterion(pred, target_y)
                benign_loss = self.cls_criterion(pred, y)
                attack_loss.backward(retain_graph=True)
                attack_grad = x.grad.clone()
                x.grad.data.zero_()
                benign_loss.backward(retain_graph=True)
                benign_grad = x.grad.clone() * self.benign_weight
                x.grad.data.zero_()
                prob = nn.functional.softmax(pred, dim=1)
                gt_prob = prob[[i for i in range(x.size(0))], y]
                gt_prob.mean().backward()
                saliency = x.grad.clone().abs()
                x.grad.data.zero_()
                benign_mask = []
                for j, s in enumerate(saliency):
                    s = s.contiguous().view(-1).sort(descending=True)[0]
                    threshold = s[int(self.benign_ratio * s.size(0))]
                    benign_mask.append(saliency[j] >= threshold)
                benign_mask = torch.cat([m.float().unsqueeze(0) for m in benign_mask], dim=0)
                x = x.clone().detach()
                attack_grad = torch.clamp(attack_grad, -self.clip_grad, self.clip_grad)
                benign_grad = torch.clamp(benign_grad, -self.clip_grad, self.clip_grad)
                x -= self.attack_lr * (attack_grad * (1 - benign_mask) + benign_grad * benign_mask)
        pred = self.model(x).argmax(dim=1)
        print(pred, target_y)
        return x, (pred == target_y).int().detach()

    def to(self, device):
        self.model.to(device)

