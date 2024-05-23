from copy import deepcopy
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
import torch.distributed as dist
import pdb
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from fed_api.data_preprocessing.cifar10.data_loader import get_dataloader as get_dataloader_cifar10
from fed_api.data_preprocessing.cinic10.data_loader import get_dataloader as get_dataloader_cinic10
from fed_api.data_preprocessing.FashionMNIST.data_loader import get_dataloader as get_dataloader_FashionMNIST
from fed_api.data_preprocessing.SVHN.data_loader import get_dataloader as get_dataloader_SVHN

def get_train_test_dataloader(args, dataidxs):
    if args.dataset == 'cifar10':
        train_loader, test_loader = get_dataloader_cifar10(args.dataset, args.data_dir, 
                                                           args.batch_size, args.batch_size, 
                                                           args.vit, dataidxs)
    elif args.dataset == 'cinic10':
        train_loader, test_loader = get_dataloader_cinic10(args.dataset, args.data_dir, 
                                                           args.batch_size, args.batch_size, 
                                                           args.vit, dataidxs)
    elif args.dataset == 'FashionMNIST':
        train_loader, test_loader = get_dataloader_FashionMNIST(args.dataset, args.data_dir, 
                                                                args.batch_size, args.batch_size, 
                                                                args.vit, dataidxs)
    elif args.dataset == 'SVHN':
        train_loader, test_loader = get_dataloader_SVHN(args.dataset, args.data_dir, 
                                                        args.batch_size, args.batch_size, 
                                                        args.vit, dataidxs)

    return train_loader, test_loader


def train(rank, args, model, device, dataidxs, message_grads_dict, comm_round):
    
    train_loader, test_loader = get_train_test_dataloader(args, dataidxs)
    
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    old_model_params = get_model_params(model)

    for epoch in range(args.epochs_client):
        train_epoch(epoch, args, model, device, train_loader, optimizer, rank)
    model_weights = deepcopy(get_model_params(model))
    
    updated_gradients = deepcopy(old_model_params)
    for layer_name, val in old_model_params.items():
        if args.vit:
            updated_gradients[layer_name] = (model_weights[layer_name] - val) / args.epochs_client
        else:
            updated_gradients[layer_name] = (model_weights[layer_name] - val)
    message_grads_dict[rank] = updated_gradients
    


def test(rank, args, model, device, dataidxs, message_acc_dict, message_loss_dict):
    
    if args.dataset == 'cifar10':
        train_loader, test_loader = get_dataloader_cifar10(args.dataset, args.data_dir, 
                                                           512, 512, 
                                                           args.vit, dataidxs)
    elif args.dataset == 'cinic10':
        train_loader, test_loader = get_dataloader_cinic10(args.dataset, args.data_dir, 
                                                           512, 512, 
                                                           args.vit, dataidxs)
    elif args.dataset == 'FashionMNIST':
        train_loader, test_loader = get_dataloader_FashionMNIST(args.dataset, args.data_dir, 
                                                                512, 512, 
                                                                args.vit, dataidxs)
    elif args.dataset == 'SVHN':
        train_loader, test_loader = get_dataloader_SVHN(args.dataset, args.data_dir, 
                                                        512, 512, 
                                                        args.vit, dataidxs)
    
    test_loss, test_acc = test_epoch(rank, model, device, test_loader)
    message_acc_dict[rank] = test_acc
    message_loss_dict[rank] = test_loss.cpu().detach().numpy()



def get_model_params(client_model):
        return client_model.cpu().state_dict()


def train_epoch(epoch, args, model, device, data_loader, 
                optimizer, rank):
    model.train()
    model.to(device)
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = F.cross_entropy(output, target.to(device))
        loss.backward()
        optimizer.step()
        if batch_idx == 0:
            print('client: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                rank, epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))
            
            

def test_epoch(rank, model, device, data_loader):
    model.eval()
    model.to(device)
    test_loss = 0
    correct = 0
    batch_idx = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            test_loss += F.cross_entropy(output, target.to(device))
            pred = output.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.to(device)).sum().item()
            batch_idx += 1


    test_loss /= batch_idx
    test_acc = 100. * correct / (batch_idx * 512)
    print('client: {} Test set: Average loss: {:.4f}, Accuracy: ({:.3f}%)'.format(
        rank, test_loss, 100. * correct / (batch_idx * 512)))
    return test_loss, test_acc

