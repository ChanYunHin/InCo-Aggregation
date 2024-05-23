


import logging
import torch
import numpy as np
from copy import deepcopy
import random
import wandb
import os
import torch.multiprocessing as mp

from train import train, test
from fed_api.data_preprocessing.cifar10.data_loader import partition_data as partition_data_cifar10
from fed_api.data_preprocessing.FashionMNIST.data_loader import partition_data as partition_data_FashionMNIST
from fed_api.data_preprocessing.cinic10.data_loader import partition_data as partition_data_cinic10
from fed_api.data_preprocessing.SVHN.data_loader import partition_data as partition_data_SVHN

from fed_model.cv.resnet_hetavg_new import resnet10, resnet14, resnet18, resnet22, resnet26
from fed_model.cv.VIT_hetavg_new import vit_small_16_Layers
import copy


def count_model_parameters(NNmodels):
    return sum(p.numel() for p in NNmodels.parameters())

def safely_savemodels(directory, client_models, worker_number):
    if not os.path.exists(directory):
        os.makedirs(directory)
    group_num = int((worker_number - 1) / 5)
    for idx, rank in enumerate(range(1, worker_number, group_num)):
    # for client_id, params in client_models.items():
        torch.save(client_models[rank].state_dict(), directory + '/group{}'.format(str(idx)))
    

def create_client_model(args, n_classes, process_id):
    if args.vit:
        # Initialize the client model following the Layer Splitting.
        group_nums = int(args.client_number / 5)
        if process_id < group_nums * 1 + 0.5:
            client_model = vit_small_16_Layers(layer_num=8)
        elif process_id > group_nums * 1 + 0.5 and process_id < group_nums * 2 + 0.5:
            client_model = vit_small_16_Layers(layer_num=9)
        elif process_id > group_nums * 2 + 0.5 and process_id < group_nums * 3 + 0.5:
            client_model = vit_small_16_Layers(layer_num=10)
        elif process_id > group_nums * 3 + 0.5 and process_id < group_nums * 4 + 0.5:
            client_model = vit_small_16_Layers(layer_num=11)
        elif process_id > group_nums * 4 + 0.5:
            client_model = vit_small_16_Layers(layer_num=12)
    
    else:
        # Initialize the client model following the Stage Splitting.
        group_nums = int(args.client_number / 5)
        if process_id < group_nums * 1 + 0.5:
            client_model = resnet10(n_classes, args)
        elif process_id > group_nums * 1 + 0.5 and process_id < group_nums * 2 + 0.5:
            client_model = resnet14(n_classes, args)
        elif process_id > group_nums * 2 + 0.5 and process_id < group_nums * 3 + 0.5:
            client_model = resnet18(n_classes, args)
        elif process_id > group_nums * 3 + 0.5 and process_id < group_nums * 4 + 0.5:
            client_model = resnet22(n_classes, args)
        elif process_id > group_nums * 4 + 0.5:
            client_model = resnet26(n_classes, args)
    
    logging.info("The number of model parameters: {}".format(count_model_parameters(client_model)))
    return client_model


def init_training_device(process_ID, gpu_num_per_machine, beginning_gpu):
    # initialize the mapping from process ID to GPU ID: <process ID, GPU ID>    
    # We have 4 GPUs.
    gpu_index = (process_ID % gpu_num_per_machine) % gpu_num_per_machine + beginning_gpu
    device = torch.device("cuda:" + str(gpu_index) if torch.cuda.is_available() else "cpu")
    logging.info(device)
    return device

def create_server_model(n_classes, args):
    if args.vit:
        server_model = vit_small_16_Layers(layer_num=12)
    else:
        server_model = resnet26(n_classes, args)
    return server_model


def load_data(args, dataset_name):
    if dataset_name == "cifar10":
        get_dataset = partition_data_cifar10
    elif dataset_name == "cinic10":
        get_dataset = partition_data_cinic10
    elif dataset_name == "FashionMNIST":
        get_dataset = partition_data_FashionMNIST
    elif dataset_name == "SVHN":
        get_dataset = partition_data_SVHN
    else:
        get_dataset = partition_data_cifar10

    X_train, y_train, X_test, y_test, net_dataidx_map, \
        traindata_cls_counts = get_dataset(args.dataset, args.data_dir, args.partition_method,
                               args.client_number, args.partition_alpha)
        
    class_num = len(np.unique(y_train))

    # train_data_num, test_data_num, train_data_global, test_data_global, \
    # train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
    # class_num = data_loader(args.dataset, args.data_dir, args.partition_method,
    #                         args.partition_alpha, args.client_number, args.batch_size)

    dataset = [X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts, class_num]
    return dataset


def aggregate_gradients(gradients_dict, comm_round, args, net_dataidx_map=None):
    global_gradients_dict = dict()
    global_gradients_cnt_dict = dict()
    # aggregate the gradients from all selected clients
    for client_id, client_id_gradients in gradients_dict.items():
        for layer_name, layer_gradients in client_id_gradients.items():
            if layer_name not in global_gradients_dict:
                global_gradients_dict[layer_name] = layer_gradients
                global_gradients_cnt_dict[layer_name] = 1
            else:
                global_gradients_dict[layer_name] += layer_gradients
                global_gradients_cnt_dict[layer_name] += 1
                
    for layer_name, layer_gradients in global_gradients_dict.items():
        global_gradients_dict[layer_name] = layer_gradients / global_gradients_cnt_dict[layer_name]
    
    # Using InCo to aggregate the gradients or not
    if args.vit:
        if args.InCo_training == 1:
            global_gradients_dict = aggregate_IN_vit_gradients(global_gradients_dict, comm_round, args)
    else:
        if args.InCo_training == 1:
            global_gradients_dict = aggregate_IN_gradients(global_gradients_dict, args)
    return global_gradients_dict



def aggregate_IN_vit_gradients(global_gradients_dict, comm_round, args):
    # ViT
    new_global_gradients_dict = deepcopy(global_gradients_dict)
    for layer_name, layer_gradients in global_gradients_dict.items():
        # update the gradients of the second norm layer
        if 'blocks' in layer_name:
            # get the gradients from the first layer 
            if ('norm2.weight' in layer_name):
                last_layer_name = layer_name.replace('norm2', 'norm1')
            
                norm_this_layer = torch.linalg.vector_norm(global_gradients_dict[layer_name])
                norm_last_layer = torch.linalg.vector_norm(global_gradients_dict[last_layer_name])

                normalized_gradients_this_layer = global_gradients_dict[layer_name] / (norm_this_layer + 1e-7)
                normalized_gradients_last_layer = global_gradients_dict[last_layer_name] / (norm_last_layer + 1e-7)
                    
                # The operation of diagonal is the trace of this matrix.
                alphaa = torch.dot(global_gradients_dict[last_layer_name], global_gradients_dict[last_layer_name])
                beta = torch.dot(global_gradients_dict[last_layer_name], global_gradients_dict[layer_name])
                
                new_beta = torch.abs(beta)
                
                # Avoid the case of dividing by zero
                calibrated_weights = new_beta/(alphaa + 1e-7)
                calibrated_weights = torch.clamp(calibrated_weights, min=0, max=args.clamp_max)
                
                # InCo gradients
                updated_gradients = (normalized_gradients_this_layer +
                                        calibrated_weights * normalized_gradients_last_layer) * ((norm_this_layer + norm_last_layer) / 2)
                
                
                # gradient clipping to avoid unstable training
                updated_gradients = torch.clamp(updated_gradients, min=-0.1, max=0.1)
                new_global_gradients_dict[layer_name] = updated_gradients
    return new_global_gradients_dict



def aggregate_IN_gradients(global_gradients_dict, args):
    # ResNet
    new_global_gradients_dict = deepcopy(global_gradients_dict)
    for layer_name, layer_gradients in global_gradients_dict.items():
        if ((('.1.' in layer_name) or ('.2.' in layer_name) or ('.3.' in layer_name)) and ('bn' not in layer_name)
            and ('downsample' not in layer_name)):
            # get the gradients from the first layer 
            assert 'bn' not in layer_name
            str_layer_name = "0"
            if '.1.' in layer_name:
                if 'conv1' in layer_name:
                    last_layer_name = layer_name[:6] + '.0.conv2' + layer_name[14:]
                    str_layer_name = layer_name[:6] + "block_1_conv1"
                else:
                    last_layer_name = layer_name[:6] + '.0.' + layer_name[9:]
                    str_layer_name = layer_name[:6] + "block_1_conv2"
            elif '.2.' in layer_name:
                if 'conv1' in layer_name:
                    last_layer_name = layer_name[:6] + '.0.conv2' + layer_name[14:]
                    str_layer_name = layer_name[:6] + "block_2_conv1"
                else:
                    last_layer_name = layer_name[:6] + '.0.' + layer_name[9:]
                    str_layer_name = layer_name[:6] + "block_2_conv2"
            elif '.3.' in layer_name:
                if 'conv1' in layer_name:
                    last_layer_name = layer_name[:6] + '.0.conv2' + layer_name[14:]
                    str_layer_name = layer_name[:6] + "block_3_conv1"
                else:
                    last_layer_name = layer_name[:6] + '.0.' + layer_name[9:]
                    str_layer_name = layer_name[:6] + "block_3_conv2"
            
            norm_this_layer = torch.linalg.matrix_norm(global_gradients_dict[layer_name])
            norm_last_layer = torch.linalg.matrix_norm(global_gradients_dict[last_layer_name])
            
            norm_this_layer = norm_this_layer.reshape(norm_this_layer.shape[0],
                                                        norm_this_layer.shape[1], 
                                                        1, 1)
            
            norm_this_layer = norm_this_layer.expand(norm_this_layer.shape[0],
                                                        norm_this_layer.shape[1], 
                                                        3, 3)
            
            norm_last_layer = norm_last_layer.reshape(norm_last_layer.shape[0],
                                                        norm_last_layer.shape[1], 
                                                        1, 1)
            
            norm_last_layer = norm_last_layer.expand(norm_last_layer.shape[0],
                                                        norm_last_layer.shape[1], 
                                                        3, 3)
            
            
            normalized_gradients_this_layer = global_gradients_dict[layer_name] / (norm_this_layer + 1e-7)
            normalized_gradients_last_layer = global_gradients_dict[last_layer_name] / (norm_last_layer + 1e-7)
                
            
            last_T_gradients = torch.transpose(global_gradients_dict[last_layer_name], -2, -1)
            # The operation of diagonal is the trace of this matrix.
            alphaa = torch.matmul(last_T_gradients, global_gradients_dict[last_layer_name]).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
            beta = torch.matmul(last_T_gradients, global_gradients_dict[layer_name]).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
            
            alphaa = alphaa.reshape(alphaa.shape[0], alphaa.shape[1], 1, 1).expand(alphaa.shape[0], alphaa.shape[1], 3, 3)
            tmp_beta = beta.reshape(beta.shape[0], beta.shape[1], 1, 1).expand(beta.shape[0], beta.shape[1], 3, 3)
            beta = torch.abs(tmp_beta)
            
            
            calibrated_weights = beta/(alphaa + 1e-7)
            calibrated_weights = torch.clamp(calibrated_weights, min=0, max=args.clamp_max)
            
            # InCo gradients
            updated_gradients = (normalized_gradients_this_layer +
                                    (calibrated_weights) * normalized_gradients_last_layer) * ((norm_this_layer + norm_last_layer) / 2)
            
            # gradient clipping to avoid unstable training
            updated_gradients = torch.clamp(updated_gradients, min=-0.1, max=0.1)
            new_global_gradients_dict[layer_name] = updated_gradients
    return new_global_gradients_dict

    

def updated_params(model_params, gradients_dict):
    for layer_name, layer_gradients in gradients_dict.items():
        if 'tracked' not in layer_name:
            model_params[layer_name] += gradients_dict[layer_name]
            if 'var' in layer_name:
                model_params[layer_name] = torch.clamp(model_params[layer_name], min=0)
        else:
            model_params[layer_name] += gradients_dict[layer_name].long()
    return model_params



def set_model_params(global_model_parameters, client_model):
    new_model_parameters = dict()
    # selete the weights this model has.
    for key in client_model.state_dict().keys():
        new_model_parameters[key] = global_model_parameters[key]
    client_model.load_state_dict(new_model_parameters)
    return client_model


def deep_copy_model_state_dict(model_dict):
    new_model_state_dict = dict()
    for client_id, client_model in model_dict.items():
        new_model_state_dict[client_id] = client_model.state_dict()
    return new_model_state_dict



def set_all_client_models_params(model_dict, global_params, args, set_best_flag=False):
    new_client_models_dict = deepcopy(model_dict)
    for client_id, params in model_dict.items():
        new_client_models_dict[client_id] = set_model_params(global_params, params)
    return new_client_models_dict



def randomly_sampling(args):
    client_idx_list = list(range(1, args.client_number + 1))
    random_idx_list = random.sample(client_idx_list, int((args.client_number)*args.client_sample_ratio))
    return random_idx_list



def clients_training(args, model_dict, 
                     device_dict, net_dataidx_map, message_gradients_dict,
                     worker_number, sample_client_idx, comm_round):
    # processes = []
    train_count = 0
    processes = [] # only need to record the last process
    for rank in range(1, worker_number):
        if rank in sample_client_idx:
            # if train_count < args.gpu_num_per_server + 1:
            dataidxs = net_dataidx_map[rank-1]
            device_id = (train_count % args.gpu_num_per_server + args.gpu_starting_point)
            device = torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")
            p = mp.Process(target=train, args=(rank, args, model_dict[rank], 
                                                device, dataidxs, message_gradients_dict, comm_round))
            train_count += 1
            p.start()
            processes.append(p)
    for p in processes:
        p.join()
            
    return message_gradients_dict




def clients_testing(args, model_dict, 
                    device_dict, net_dataidx_map, message_acc_dict,
                    message_loss_dict, worker_number):
    test_count = 0 # the number of concurrent running clients is not larger than 10
    group_num = int((worker_number - 1) / 5)
    for rank in range(1, worker_number, group_num):
        processes = [] # only need to record the last process
        device_id = (test_count % args.gpu_num_per_server + args.gpu_starting_point)
        device = torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")
        dataidxs = net_dataidx_map[0] # unavailable in testing
        p = mp.Process(target=test, args=(rank, args, model_dict[rank], 
                                          device, dataidxs, message_acc_dict, message_loss_dict))
        test_count += 1
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    return message_acc_dict, message_loss_dict



def clients_ViT_training(args, model_dict, 
                         device_dict, net_dataidx_map, message_gradients_dict,
                         worker_number, sample_client_idx, comm_round):
    # processes = []
    train_count = 0
    processes = [] # only need to record the last process
    for rank in range(1, worker_number):
        if rank in sample_client_idx:
            # if train_count < args.gpu_num_per_server + 1:
            dataidxs = net_dataidx_map[rank-1]
            device_id = (train_count % args.gpu_num_per_server + args.gpu_starting_point)
            device = torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")
            p = mp.Process(target=train, args=(rank, args, model_dict[rank], 
                                                device, dataidxs, message_gradients_dict, comm_round))
            train_count += 1
            p.start()
            processes.append(p)
            # if train_count >= args.gpu_num_per_server + 1:
                # for p in processes:
                #     p.join()
                # train_count = 0
                # processes = []
    for p in processes:
        p.join()
            
    return message_gradients_dict


def clients_ViT_testing(args, model_dict, 
                        device_dict, net_dataidx_map, message_acc_dict,
                        message_loss_dict, worker_number):
    test_count = 0 # the number of concurrent running clients is not larger than 10
    group_num = int((worker_number - 1) / 5)
    for rank in range(1, worker_number, group_num):
        processes = [] # only need to record the last process
        # if test_count < args.gpu_num_per_server:
        device_id = (test_count % args.gpu_num_per_server + args.gpu_starting_point)
        device = torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")
        dataidxs = net_dataidx_map[0] # unavailable in testing
        p = mp.Process(target=test, args=(rank, args, model_dict[rank], 
                                          device, dataidxs, message_acc_dict, message_loss_dict))
        test_count += 1
        p.start()
        processes.append(p)
        # if test_count >= args.gpu_num_per_server:
    for p in processes:
        p.join()
    return message_acc_dict, message_loss_dict



def get_avg_metrics(message_acc_dict, message_loss_dict, comm_round):
    avg_acc = 0
    for client_id, acc_val in message_acc_dict.items():
        avg_acc += acc_val
        wandb.log({"Test_client/Acc_{}".format(client_id): acc_val, "epoch": comm_round + 1})
    avg_loss = 0
    for client_id, loss_val in message_loss_dict.items():
        avg_loss += loss_val
        wandb.log({"Test_client/Loss_{}".format(client_id): loss_val, "epoch": comm_round + 1})
    avg_acc /= len(message_acc_dict)
    avg_loss /= len(message_loss_dict)
    return avg_acc, avg_loss
