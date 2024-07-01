'''
ResNet for CIFAR-10/100 Dataset.

Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
2. https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua
3. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385

'''

import pdb
import torch
import torch.nn as nn

# __all__ = ['ResNet']
from tkinter import X
import torch
import torch.nn as nn
import torch.nn.functional as F

# from parameters import args

import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck, conv1x1
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor
import numpy as np


class Resnet_ft(nn.Module):
    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], num_classes: int = 1000, 
                 data_in_channels: int = 3, zero_init_residual: bool = False, groups: int = 1, width_per_group: int = 64, 
                 replace_stride_with_dilation: Optional[List[bool]] = None, 
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(data_in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)
        
    def forward(self, x: Tensor):
        
        encoded_feature_inputs = self.encode(x)
        encoded_features, layer1_encoded_features, layer2_encoded_features, \
            layer3_encoded_features, layer4_encoded_features= self.decode(encoded_feature_inputs)
        
        x = self.avgpool(encoded_features)
        x = torch.flatten(x, 1)
        
        
        return self.fc(x) #layer1_encoded_features, \
            # layer2_encoded_features, layer3_encoded_features, layer4_encoded_features
    
    def encode(self, x: Tensor) -> Tensor:
        # print(x.shape)
        # print(self.conv1.out_channels)
        # print(self.conv1.weight.shape)
        x = self.conv1(x)
        # print(self.inplanes)
        # print(x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        return x
    
    def decode(self, x: Tensor) -> Tensor:
        
        x = self.layer1(x)
        layer1_encoded_features = x
        x = self.layer2(x)
        layer2_encoded_features = x
        x = self.layer3(x)
        layer3_encoded_features = x
        x = self.layer4(x)
        layer4_encoded_features = x
        
        
        
        return x, layer1_encoded_features, layer2_encoded_features, layer3_encoded_features, layer4_encoded_features


def resnet10(c, args, **kwargs):
    """
    Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """

    # model = Resnet_ft(BasicBlock, [2, 2, 2, 2], num_classes=c, data_in_channels=3, **kwargs)
    if args.dataset == 'cifar10':
        model = Resnet_ft(BasicBlock, [1, 1, 1, 1], num_classes=c, data_in_channels=3, **kwargs)
    elif args.dataset == 'FashionMNIST':
        model = Resnet_ft(BasicBlock, [1, 1, 1, 1], num_classes=c, data_in_channels=1, **kwargs)
    else:
        model = Resnet_ft(BasicBlock, [1, 1, 1, 1], num_classes=c, data_in_channels=3, **kwargs)
    # if pretrained:
    #     checkpoint = torch.load(path)
    #     state_dict = checkpoint['state_dict']

    #     from collections import OrderedDict
    #     new_state_dict = OrderedDict()
    #     for k, v in state_dict.items():
    #         # name = k[7:]  # remove 'module.' of dataparallel
    #         name = k.replace("module.", "")
    #         new_state_dict[name] = v

    #     model.load_state_dict(new_state_dict)
    return model

def resnet14(c, args, **kwargs):
    """
    Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """

    # model = Resnet_ft(BasicBlock, [2, 2, 2, 2], num_classes=c, data_in_channels=3, **kwargs)
    if args.dataset == 'cifar10':
        model = Resnet_ft(BasicBlock, [1, 1, 2, 2], num_classes=c, data_in_channels=3, **kwargs)
    elif args.dataset == 'FashionMNIST':
        model = Resnet_ft(BasicBlock, [1, 1, 2, 2], num_classes=c, data_in_channels=1, **kwargs)
    else:
        model = Resnet_ft(BasicBlock, [1, 1, 2, 2], num_classes=c, data_in_channels=3, **kwargs)
    # if pretrained:
    #     checkpoint = torch.load(path)
    #     state_dict = checkpoint['state_dict']

    #     from collections import OrderedDict
    #     new_state_dict = OrderedDict()
    #     for k, v in state_dict.items():
    #         # name = k[7:]  # remove 'module.' of dataparallel
    #         name = k.replace("module.", "")
    #         new_state_dict[name] = v

    #     model.load_state_dict(new_state_dict)
    return model


def resnet18(c, args, **kwargs):
    """
    Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """

    # model = Resnet_ft(BasicBlock, [2, 2, 2, 2], num_classes=c, data_in_channels=3, **kwargs)
    if args.dataset == 'cifar10':
        model = Resnet_ft(BasicBlock, [2, 2, 2, 2], num_classes=c, data_in_channels=3, **kwargs)
    elif args.dataset == 'FashionMNIST':
        model = Resnet_ft(BasicBlock, [2, 2, 2, 2], num_classes=c, data_in_channels=1, **kwargs)
    else:
        model = Resnet_ft(BasicBlock, [2, 2, 2, 2], num_classes=c, data_in_channels=3, **kwargs)
    # if pretrained:
    #     checkpoint = torch.load(path)
    #     state_dict = checkpoint['state_dict']

    #     from collections import OrderedDict
    #     new_state_dict = OrderedDict()
    #     for k, v in state_dict.items():
    #         # name = k[7:]  # remove 'module.' of dataparallel
    #         name = k.replace("module.", "")
    #         new_state_dict[name] = v

    #     model.load_state_dict(new_state_dict)
    return model

def resnet22(c, args, **kwargs):
    """
    Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """

    # model = Resnet_ft(BasicBlock, [2, 2, 2, 2], num_classes=c, data_in_channels=3, **kwargs)
    if args.dataset == 'cifar10':
        model = Resnet_ft(BasicBlock, [2, 2, 3, 3], num_classes=c, data_in_channels=3, **kwargs)
    elif args.dataset == 'FashionMNIST':
        model = Resnet_ft(BasicBlock, [2, 2, 3, 3], num_classes=c, data_in_channels=1, **kwargs)
    else:
        model = Resnet_ft(BasicBlock, [2, 2, 3, 3], num_classes=c, data_in_channels=3, **kwargs)
    # if pretrained:
    #     checkpoint = torch.load(path)
    #     state_dict = checkpoint['state_dict']

    #     from collections import OrderedDict
    #     new_state_dict = OrderedDict()
    #     for k, v in state_dict.items():
    #         # name = k[7:]  # remove 'module.' of dataparallel
    #         name = k.replace("module.", "")
    #         new_state_dict[name] = v

    #     model.load_state_dict(new_state_dict)
    return model


def resnet26(c, args, **kwargs):
    """
    Constructs a ResNet-26 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """

    # model = ResNet(Bottleneck, [2, 2, 2], num_classes=c, **kwargs)
    if args.dataset == 'cifar10':
        model = Resnet_ft(BasicBlock, [3, 3, 3, 3], num_classes=c, data_in_channels=3, **kwargs)
    elif args.dataset == 'FashionMNIST':
        model = Resnet_ft(BasicBlock, [3, 3, 3, 3], num_classes=c, data_in_channels=1, **kwargs)
    else:
        model = Resnet_ft(BasicBlock, [3, 3, 3, 3], num_classes=c, data_in_channels=3, **kwargs)

    return model

