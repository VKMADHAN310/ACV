# models/dynamic_mobilenetv2.py
import torch.nn as nn
from layers import DGCConv2d

class InvertedResidualDGC(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers += [DGCConv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                       nn.BatchNorm2d(hidden_dim),
                       nn.ReLU6(inplace=True)]
        # depth‑wise 3×3 replaced by DGCConv2d
        layers += [DGCConv2d(hidden_dim, hidden_dim, 3, stride, 1,
                             groups=hidden_dim, bias=False),
                   nn.BatchNorm2d(hidden_dim),
                   nn.ReLU6(inplace=True)]
        # project
        layers += [DGCConv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                   nn.BatchNorm2d(oup)]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)

class DyMobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super().__init__()
        cfg = [(1, 16, 1, 1),
               (6, 24, 2, 2),
               (6, 32, 3, 2),
               (6, 64, 4, 2),
               (6, 96, 3, 1),
               (6, 160, 3, 2),
               (6, 320, 1, 1)]
        input_channel = int(32 * width_mult)
        layers = [DGCConv2d(3, input_channel, 3, 2, 1, bias=False),
                  nn.BatchNorm2d(input_channel),
                  nn.ReLU6(inplace=True)]
        for t, c, n, s in cfg:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidualDGC(
                    input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        layers += [DGCConv2d(input_channel, last_channel, 1, 1, 0, bias=False),
                   nn.BatchNorm2d(last_channel),
                   nn.ReLU6(inplace=True)]
        self.features = nn.Sequential(*layers)
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(last_channel, num_classes)

    def forward(self, x):
        x = self.avgpool(self.features(x)).flatten(1)
        return self.classifier(x)

def dymobilenet_v2(num_classes=1000, width_mult=1.0):
    return DyMobileNetV2(num_classes=num_classes, width_mult=width_mult)
