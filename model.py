from torchvision.models import vgg16, VGG16_Weights
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.net.classifier = nn.Linear(25088, num_classes)

    def forward(self, x):
        return self.net(x)

