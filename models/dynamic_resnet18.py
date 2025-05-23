# models/dynamic_resnet18.py
import torch.nn as nn
from layers import DGCConv2d            # your DGC layer
from .resnet import BasicBlock
         # reuse stem & block shell from resnet.py

def _dgc_conv3x3(in_c, out_c, stride=1, groups=1):
    """3×3 Dynamic Group Conv wrapper (keeps padding=1)."""
    return DGCConv2d(in_c, out_c, kernel_size=3, stride=stride,
                     padding=1, bias=False, groups=groups)

class DGBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_c, out_c, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _dgc_conv3x3(in_c, out_c, stride)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = _dgc_conv3x3(out_c, out_c)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

def _make_layer(block, inplanes, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            DGCConv2d(inplanes, planes * block.expansion, kernel_size=1,
                      stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )
    layers = [block(inplanes, planes, stride, downsample)]
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes))
    return nn.Sequential(*layers), inplanes

class DyResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.inplanes = 64
        self.conv1 = DGCConv2d(3, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1, self.inplanes = _make_layer(DGBasicBlock, self.inplanes, 64, 2)
        self.layer2, self.inplanes = _make_layer(DGBasicBlock, self.inplanes, 128, 2, stride=2)
        self.layer3, self.inplanes = _make_layer(DGBasicBlock, self.inplanes, 256, 2, stride=2)
        self.layer4, self.inplanes = _make_layer(DGBasicBlock, self.inplanes, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512 * DGBasicBlock.expansion, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

def dyresnet18(num_classes=1000):
    return DyResNet18(num_classes=num_classes)
