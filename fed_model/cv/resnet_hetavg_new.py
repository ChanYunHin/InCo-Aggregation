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
    
    def forward_get_all(self, x: Tensor):
        encoded_feature_inputs = self.encode(x)
        encoded_features, layer1_encoded_features, layer2_encoded_features, \
            layer3_encoded_features, layer4_encoded_features= self.decode(encoded_feature_inputs)
        
        return self.fc(encoded_features), encoded_features, layer1_encoded_features, \
            layer2_encoded_features, layer3_encoded_features, layer4_encoded_features
    
    
    def forward_IN(self, x: Tensor):
        
        encoded_feature_inputs = self.encode(x)
        encoded_features, layer1_encoded_features, layer2_encoded_features, \
            layer3_encoded_features, layer4_encoded_features= self.decode(encoded_feature_inputs)
        
        x = self.avgpool(encoded_features)
        x = torch.flatten(x, 1)
        
        return self.fc(x), encoded_feature_inputs, encoded_features #layer1_encoded_features, \
            # layer2_encoded_features, layer3_encoded_features, layer4_encoded_features
    
    
    
    def forward_moon(self, x: Tensor):
        encoded_feature_inputs = self.encode(x)
        encoded_features, layer1_encoded_features, layer2_encoded_features, \
            layer3_encoded_features, layer4_encoded_features= self.decode(encoded_feature_inputs)
        
        return encoded_features, self.fc(encoded_features)
    
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

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out = 10):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden1 = nn.Linear(512, 256)
        self.layer_hidden2 = nn.Linear(256, 256)
        self.layer_hidden3 = nn.Linear(256, 128)
        self.layer_out = nn.Linear(128, dim_out)
        self.softmax = nn.Softmax(dim=1)
        self.weight_keys = [['layer_input.weight', 'layer_input.bias'],
                            ['layer_hidden1.weight', 'layer_hidden1.bias'],
                            ['layer_hidden2.weight', 'layer_hidden2.bias'],
                            ['layer_hidden3.weight', 'layer_hidden3.bias'],
                            ['layer_out.weight', 'layer_out.bias']
                            ]

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.relu(x)

        x = self.layer_hidden1(x)
        x = self.relu(x)

        x = self.layer_hidden2(x)
        x = self.relu(x)

        x = self.layer_hidden3(x)
        x = self.relu(x)

        x = self.layer_out(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class CNNCifar(nn.Module):
    def __init__(self, input_channels, num_classes, layer1=256, layer2=512, layer3=512, dropout_rate=0.2):
        super(CNNCifar, self).__init__()

        # The output size of convolution layer: [(W-K+2P)/S]+1
        # The output size of maxpooling layer: (W-K)/S + 1
        # W is the input volume: image is 32*32*3, W=32
        # K is the kernel size
        # P is the padding
        # S is the stride

        # 1st convolution layer
        # input size: [n, 3, 32, 32]
        # output size: [n, layer1, 32, 32]
        self.conv1 = nn.Conv2d(in_channels=input_channels , out_channels=layer1, 
                               kernel_size=(3, 3), padding='same')

        self.conv1_dim = self.conv_output_dimension(32, 
                                                    kernel=3, 
                                                    padding=1, 
                                                    stride=1)
        
        self.bn1 = nn.Sequential(
            nn.BatchNorm2d(layer1),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # input size: [n, layer1, 32, 32]
        # output size: [n, ]
        self.pooling1 = nn.AvgPool2d((2, 2), stride=1, padding=1)
        self.pool1_dim = self.avgpool_output_dimension(self.conv1_dim, 
                                                       kernel=2, 
                                                       padding=1, 
                                                       stride=1)
        
        # 2nd convolution layer

        self.conv2 = nn.Conv2d(in_channels=layer1, out_channels=layer2, kernel_size=(3, 3), stride=2,
                               padding='valid')
        self.conv2_dim = self.conv_output_dimension(self.pool1_dim, 
                                                    kernel=3,
                                                    padding=0,
                                                    stride=2)


        self.bn2 = nn.Sequential(
            nn.BatchNorm2d(layer2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.pooling2 = nn.AvgPool2d((2, 2), stride=2, padding=0)
        self.pool2_dim = self.avgpool_output_dimension(self.conv2_dim, 
                                                       kernel=2, 
                                                       padding=0, 
                                                       stride=2)
        # 3rd convolution layer

        self.conv3 = nn.Conv2d(in_channels=layer2, out_channels=layer3, kernel_size=(3, 3), stride=2,
                               padding='valid')

        self.conv3_dim = self.conv_output_dimension(self.pool2_dim, 
                                                    kernel=3,
                                                    padding=0,
                                                    stride=2)


        self.bn3 = nn.Sequential(
            nn.BatchNorm2d(layer3),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.sixth_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=layer3*self.conv3_dim*self.conv3_dim, out_features=1024)
        )

        
        self.fc1 = nn.Linear(in_features=1024, 
                             out_features=512, 
                             bias=False)
        
        self.seventh_layer = nn.Linear(in_features=512, 
                                       out_features=num_classes, 
                                       bias=False)

    def encode(self, x):
        x = self.conv1(x)
        # if training:
        x = self.bn1(x)
            # print("in_training")
        x = self.pooling1(x)

        x = self.conv2(x)
        # if training:
        x = self.bn2(x)
        x = self.pooling2(x)

        x = self.conv3(x)
        # if training:
        x = self.bn3(x)
        
        x = self.sixth_layer(x)
        return x

    # def reparametrize(self, mu, logvar):
    #     std = logvar.mul(0.5).exp_()
    #     if torch.cuda.is_available():
    #         eps = torch.cuda.FloatTensor(std.size()).normal_()
    #     else:
    #         eps = torch.FloatTensor(std.size()).normal_()
    #     return eps.mul(std).add_(mu)

    def decode(self, z):
        # z = torch.cat([z, labels], dim=1)
        return self.seventh_layer(z)

    def forward(self, x):
        z = self.encode(x)
        z = self.fc1(z)
        return self.decode(z)

    # def forward(self, inputs):
    #     x = self.conv1(inputs)
    #     # if training:
    #     x = self.bn1(x)
    #         # print("in_training")
    #     x = self.pooling1(x)

    #     x = self.conv2(x)
    #     # if training:
    #     x = self.bn2(x)
    #     x = self.pooling2(x)

    #     x = self.conv3(x)
    #     # if training:
    #     x = self.bn3(x)

    #     x = x.view(x.shape[0], -1)
    #     x = self.seventh_layer(x)
    #     return x

    def conv_output_dimension(self, input_height, kernel, padding, stride=1, dilation=1):
        height_in = input_height
        height_out = np.floor(1 + (height_in + 2 * padding - dilation * (kernel - 1) - 1)/stride)
        return int(height_out)

    def avgpool_output_dimension(self, input_height, kernel, padding, stride):
        height_in = input_height
        height_out = np.floor(1 + (height_in + 2 * padding - kernel)/stride)
        return int(height_out)

# class CNNCifar(nn.Module):
#     def __init__(self):
#         super(CNNCifar, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 100)
#         self.fc3 = nn.Linear(100, 10)

#         # self.weight_keys = [['fc3.weight', 'fc3.bias'],
#         #                     ['fc2.weight', 'fc2.bias'],
#         #                     ['fc1.weight', 'fc1.bias'],
#         #                     ['conv2.weight', 'conv2.bias'],
#         #                     ['conv1.weight', 'conv1.bias'],
#         #                     ]

#         # self.weight_keys = [['conv1.weight', 'conv1.bias'],
#         #                     ['conv2.weight', 'conv2.bias'],
#         #                     ['fc2.weight', 'fc2.bias'],
#         #                     ['fc3.weight', 'fc3.bias'],
#         #                     ['fc1.weight', 'fc1.bias'],
#         #                     ]

#         self.weight_keys = [['fc1.weight', 'fc1.bias'],
#                             ['fc2.weight', 'fc2.bias'],
#                             ['fc3.weight', 'fc3.bias'],
#                             ['conv2.weight', 'conv2.bias'],
#                             ['conv1.weight', 'conv1.bias'],
#                             ]

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x