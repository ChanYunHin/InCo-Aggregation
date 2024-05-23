import logging

import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from .datasets import FashionMNIST_truncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# generate the non-IID distribution for all methods


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_FashionMNIST():

    
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize(32),
    ])

    # train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize(32),
    ])

    return train_transform, valid_transform


from PIL import Image

# Define the custom Replicate transformation
class Replicate(object):
    def __init__(self, num_reps):
        self.num_reps = num_reps

    def __call__(self, image):
        image_rgb = image.convert('RGB')
        channels = [image] * self.num_reps
        return Image.merge('RGB', channels)
    

def _data_transforms_FashionMNIST_vit():

    
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        Replicate(num_reps=3),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x.reshape(1, 28, 28)),
        transforms.Resize(224),
        # transforms.Lambda(lambda x: x.expand(3, -1, -1)),
        
    ])

    # train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        Replicate(num_reps=3),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x.reshape(1, 28, 28)),
        transforms.Resize(224),
        # transforms.Lambda(lambda x: x.expand(3, -1, -1)),
        
    ])

    return train_transform, valid_transform


def load_FashionMNIST_data(datadir):
    train_transform, test_transform = _data_transforms_FashionMNIST()

    FashionMNIST_train_ds = FashionMNIST_truncated(datadir, train=True, download=True, transform=train_transform)
    FashionMNIST_test_ds = FashionMNIST_truncated(datadir, train=False, download=True, transform=test_transform)

    X_train, y_train = FashionMNIST_train_ds.data, FashionMNIST_train_ds.target
    X_test, y_test = FashionMNIST_test_ds.data, FashionMNIST_test_ds.target

    return (X_train, y_train, X_test, y_test)


def partition_data(dataset, datadir, partition, n_nets, alpha):
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_FashionMNIST_data(datadir)
    n_train = X_train.shape[0]
    # n_test = X_test.shape[0]
    net_dataidx_map = {}

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = 10
        N = y_train.shape[0]
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    else:
        traindata_cls_counts = 0

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts


# for centralized training
def get_dataloader(dataset, datadir, train_bs, test_bs, vit_flag, dataidxs=None):
    return get_dataloader_FashionMNIST(datadir, train_bs, test_bs, vit_flag, dataidxs)


# for local devices
def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    return get_dataloader_test_FashionMNIST(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


def get_dataloader_FashionMNIST(datadir, train_bs, test_bs, vit_flag, dataidxs=None):
    dl_obj = FashionMNIST_truncated

    if vit_flag:
        transform_train, transform_test = _data_transforms_FashionMNIST_vit()
    else:
        transform_train, transform_test = _data_transforms_FashionMNIST()

    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

    if vit_flag:
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True, num_workers=1, prefetch_factor=2)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True, num_workers=1, prefetch_factor=2)
    else:
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def get_dataloader_test_FashionMNIST(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = FashionMNIST_truncated

    transform_train, transform_test = _data_transforms_FashionMNIST()

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def load_partition_data_distributed_FashionMNIST(process_id, dataset, data_dir, partition_method, partition_alpha,
                                            client_number, batch_size):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
        logging.info("train_dl_global number = " + str(len(train_data_global)))
        logging.info("test_dl_global number = " + str(len(test_data_global)))
        train_data_local = None
        test_data_local = None
        local_data_num = 0
    else:
        # get local dataset
        dataidxs = net_dataidx_map[process_id - 1]
        local_data_num = len(dataidxs)
        logging.info("rank = %d, local_sample_number = %d" % (process_id, local_data_num))
        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                 dataidxs)
        logging.info("process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            process_id, len(train_data_local), len(test_data_local)))
        train_data_global = None
        test_data_global = None
    return train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num


def load_partition_data_FashionMNIST(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                 dataidxs)
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num



# def read_data(train_data_dir, test_data_dir):
#     '''parses data in given train and test data directories

#     assumes:
#     - the data in the input directories are .json files with 
#         keys 'users' and 'user_data'
#     - the set of train set users is the same as the set of test set users

#     Return:
#         clients: list of non-unique client ids
#         groups: list of group ids; empty list if none found
#         train_data: dictionary of train data
#         test_data: dictionary of test data
#     '''
#     clients = []
#     groups = []
#     train_data = {}
#     test_data = {}

#     train_files = os.listdir(train_data_dir)
#     train_files = [f for f in train_files if f.endswith('.json')]
#     for f in train_files:
#         file_path = os.path.join(train_data_dir, f)
#         with open(file_path, 'r') as inf:
#             cdata = json.load(inf)
#         clients.extend(cdata['users'])
#         if 'hierarchies' in cdata:
#             groups.extend(cdata['hierarchies'])
#         train_data.update(cdata['user_data'])

#     test_files = os.listdir(test_data_dir)
#     test_files = [f for f in test_files if f.endswith('.json')]
#     for f in test_files:
#         file_path = os.path.join(test_data_dir, f)
#         with open(file_path, 'r') as inf:
#             cdata = json.load(inf)
#         test_data.update(cdata['user_data'])

#     clients = sorted(cdata['users'])

#     return clients, groups, train_data, test_data


# def batch_data(data, batch_size):
#     '''
#     data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
#     returns x, y, which are both numpy array of length: batch_size
#     '''
#     data_x = data['x']
#     data_y = data['y']

#     # randomly shuffle data
#     np.random.seed(100)
#     rng_state = np.random.get_state()
#     np.random.shuffle(data_x)
#     np.random.set_state(rng_state)
#     np.random.shuffle(data_y)

#     # loop through mini-batches
#     batch_data = list()
#     for i in range(0, len(data_x), batch_size):
#         batched_x = data_x[i:i + batch_size]
#         batched_y = data_y[i:i + batch_size]
#         batched_x = torch.from_numpy(np.asarray(batched_x)).float()
#         batched_y = torch.from_numpy(np.asarray(batched_y)).long()
#         batch_data.append((batched_x, batched_y))
#     return batch_data


# def load_partition_data_FashionMNIST(batch_size,
#                               train_path="./../../../data/FashionMNIST/train",
#                               test_path="./../../../data/FashionMNIST/test"):
#     users, groups, train_data, test_data = read_data(train_path, test_path)

#     if len(groups) == 0:
#         groups = [None for _ in users]
#     train_data_num = 0
#     test_data_num = 0
#     train_data_local_dict = dict()
#     test_data_local_dict = dict()
#     train_data_local_num_dict = dict()
#     train_data_global = list()
#     test_data_global = list()
#     client_idx = 0
#     logging.info("loading data...")
#     for u, g in zip(users, groups):
#         user_train_data_num = len(train_data[u]['x'])
#         user_test_data_num = len(test_data[u]['x'])
#         train_data_num += user_train_data_num
#         test_data_num += user_test_data_num
#         train_data_local_num_dict[client_idx] = user_train_data_num

#         # transform to batches
#         train_batch = batch_data(train_data[u], batch_size)
#         test_batch = batch_data(test_data[u], batch_size)

#         # index using client index
#         train_data_local_dict[client_idx] = train_batch
#         test_data_local_dict[client_idx] = test_batch
#         train_data_global += train_batch
#         test_data_global += test_batch
#         client_idx += 1
#     logging.info("finished the loading data")
#     client_num = client_idx
#     class_num = 10

#     return client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
#            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num


