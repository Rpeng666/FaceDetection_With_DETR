import torch
import torch.nn as nn
from torchvision import models
import types


class Identity(nn.Module):
    def __init__(self) -> None:
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def _forward_impl(self, x):
    # See note [TorchScript super()]
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    # x = self.avgpool(x)

    return x

def build_backbone():
    resnet = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
    
    resnet._forward_impl = types.MethodType(_forward_impl, resnet)

    # resnet.avgpool = nn.AdaptiveAvgPool2d(output_size=(16, 16))

    return resnet  # 输出为[batch_size, 512, 15, 15]

if __name__ == '__main__':
    model = build_backbone()

    print(type(model))
    x = torch.rand((1, 3, 450, 450))

    print(model(x).shape)