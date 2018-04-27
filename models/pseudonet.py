from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = ['P3D', 'P3D15', 'P3D31', 'P3D63', 'P3D131', 'P3D199']


def conv_S(in_planes, out_planes, stride=1, padding=(1, 1, 1)):
    # as is descriped, conv S is 1x3x3
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride,
                     padding=padding, bias=False)


def conv_T(in_planes, out_planes, stride=1, padding=(1, 1, 1)):
    # conv T is 3x1x1
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride,
                     padding=padding, bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1),
                             out.size(2), out.size(3),
                             out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv_S(planes, planes, stride=1, padding=(0, 1, 1))
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv_T(planes, planes, stride=1, padding=(1, 0, 0))
        self.bn3 = nn.BatchNorm3d(planes)
        self.conv4 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm3d(planes * 4)

        self.relu = nn.ReLU(inplace=True)

        self.stride = stride


    def ST_A(self, x):
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.ST_A(out)

        out = self.conv4(out)
        out = self.bn4(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # out = self.relu(out)

        return out


class P3D(nn.Module):
    def __init__(self, block, layers, sample_size, sample_duration, shortcut_type='B', num_classes=400):
        self.inplanes = 64
        super(P3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def P3D15(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = P3D(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def P3D31(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = P3D(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def P3D63(**kwargs):
    """Construct a P3D63 modelbased on a ResNet-50-3D model.
    """
    model = P3D(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def P3D131(**kwargs):
    """Construct a P3D131 model based on a ResNet-101-3D model.
    """
    model = P3D(BasicBlock, [3, 4, 23, 3], **kwargs)
    return model


def P3D199(pretrained=False, modality='RGB', **kwargs):
    """construct a P3D199 model based on a ResNet-152-3D model.
    """
    model = P3D(BasicBlock, [3, 8, 36, 3], **kwargs)
    if pretrained == True:
        pretrained_file = 'p3d_rgb_199.checkpoint.pth.tar'

    weights = torch.load(pretrained_file)['state_dict']
    model.load_state_dict(weights)
    return model