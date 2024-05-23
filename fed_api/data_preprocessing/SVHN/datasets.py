import argparse
import json
import os
import shutil
import sys

import numpy as np
from torchvision.datasets import SVHN
import torch.utils.data as data
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))


def add_args(parser):
    parser.add_argument('--client_num_per_round', type=int, default=3, metavar='NN',
                        help='number of workers')
    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we should use')
    args = parser.parse_args()
    return args



class SVHN_custom(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        if self.train is True:
            # svhn_dataobj1 = SVHN(self.root, 'train', self.transform, self.target_transform, self.download)
            # svhn_dataobj2 = SVHN(self.root, 'extra', self.transform, self.target_transform, self.download)
            # data = np.concatenate((svhn_dataobj1.data, svhn_dataobj2.data), axis=0)
            # target = np.concatenate((svhn_dataobj1.labels, svhn_dataobj2.labels), axis=0)

            svhn_dataobj = SVHN(self.root, 'train', self.transform, self.target_transform, self.download)
            data = svhn_dataobj.data
            target = svhn_dataobj.labels
        else:
            svhn_dataobj = SVHN(self.root, 'test', self.transform, self.target_transform, self.download)
            data = svhn_dataobj.data
            target = svhn_dataobj.labels

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        # print("svhn data:", data)
        # print("len svhn data:", len(data))
        # print("type svhn data:", type(data))
        # print("svhn target:", target)
        # print("type svhn target", type(target))
        return data, target

    # def truncate_channel(self, index):
    #     for i in range(index.shape[0]):
    #         gs_index = index[i]
    #         self.data[gs_index, :, :, 1] = 0.0
    #         self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        # print("svhn img:", img)
        # print("svhn target:", target)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np.transpose(img, (1, 2, 0))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
