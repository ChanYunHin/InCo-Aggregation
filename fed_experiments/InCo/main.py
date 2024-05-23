import argparse
import numpy as np
import random
import logging
import torch
import torch.multiprocessing as mp

# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from utils import create_client_model, init_training_device, create_server_model, load_data, safely_savemodels
from utils import aggregate_gradients, updated_params, set_model_params, set_all_client_models_params
from utils import randomly_sampling, clients_training, clients_testing, get_avg_metrics

import wandb
import time
import copy

def set_all_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)

# Training settings

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='../../data/cifar10', help='data directory')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=5e-4)

    parser.add_argument('--epochs_client', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--client_number', type=int, default=20, metavar='NN',
                        help='number of workers in a distributed cluster')
    
    parser.add_argument('--client_sample_ratio', type=float, default=0.1, metavar='NN',
                        help='number of sampled clients in a FL')

    parser.add_argument('--comm_round', type=int, default=200,
                        help='how many round of communications we shoud use')

    parser.add_argument('--vit', action='store_true',
                        help='Using VIT or not.')
    
    parser.add_argument('--alpha', default=1.0, type=float, help='The coefficient of gradients')
    
    parser.add_argument('--InCo_training', type=int, default=1, help='0 means not using InCo, 1 means using InCo')
    
    parser.add_argument('--optimizer', default="Adam", type=str, help='optimizer: AdamW, Adam, etc.')
    
    parser.add_argument('--running_name', default="fedml_resnet56_homo_cifar10", type=str)

    parser.add_argument('--gpu_num_per_server', type=int, default=4,
                        help='gpu_num_per_server')
    
    parser.add_argument('--gpu_starting_point', type=int, default=0, help='the beginning gpu number')
    
    parser.add_argument('--clamp_max', type=int, default=5, help='clamp the max sigma to 5')
    
    parser.add_argument('--rand_seed', type=int, default=0, help='random seed')
    
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    model_dict = dict()
    device_dict = dict()
    
    # get data
    dataset = load_data(args, args.dataset)
    [X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts, class_num] = dataset
    # A Server + clients
    worker_number = args.client_number + 1
    
    
    if args.partition_method == 'hetero-fixed':
        if args.dataset == "cinic10":
            net_dataidx_map = np.load('./fix_distribution/cinic10_hetero-fix-{}-{}.npy'.format(args.client_number, args.partition_alpha),
                                       allow_pickle=True).item()
        elif args.dataset == "SVHN":
            net_dataidx_map = np.load('./fix_distribution/SVHN_hetero-fix-{}-{}.npy'.format(args.client_number, args.partition_alpha),
                                       allow_pickle=True).item()
        else:
            net_dataidx_map = np.load('./fix_distribution/hetero-fix-{}-{}.npy'.format(args.client_number, args.partition_alpha),
                                    allow_pickle=True).item()
        
    
    for process_id in range(args.client_number + 1):
        
        # customize the log format
        logging.basicConfig(level=logging.INFO,
                            format=str(process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S')
        logging.info("########process ID = " + str(process_id) + "########")

        # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
        if process_id == 0:
            print("testing")
            wandb.init(
                project="new_fed",
                name="HeteroAvgMP-" + str(args.running_name),
                config=args,
            )

        logging.info("process_id = %d, size = %d" % (process_id, worker_number))
        device = init_training_device(process_id, args.gpu_num_per_server, args.gpu_starting_point)
        device_dict[process_id] = device
        
        # create model
        if process_id == 0:
            model = create_server_model(class_num, args)
            model_dict[process_id] = model
            global_params = model_dict[process_id].cpu().state_dict()
        else:
            model = create_client_model(args, class_num, process_id)
            model_dict[process_id] = set_model_params(global_params, model)


    # Training FL with multiprocessing
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    old_global_gradients = dict()
        
    for comm_round in range(args.comm_round):
        # manager.dict only can change the first depth value.
        message_gradients_dict = manager.dict()
        message_acc_dict = manager.dict()
        message_loss_dict = manager.dict()
        sample_client_idx = randomly_sampling(args)
        logging.info("comm round: {}".format(comm_round))
        
        
        message_gradients_dict = clients_training(args, model_dict, device_dict, net_dataidx_map, 
                                                    message_gradients_dict, worker_number, sample_client_idx, comm_round)

    
        global_gradients = aggregate_gradients(message_gradients_dict, comm_round, args, net_dataidx_map)
        global_params = updated_params(global_params, global_gradients)
        model_dict = set_all_client_models_params(model_dict, global_params, args)
                    
        message_acc_dict, message_loss_dict = \
            clients_testing(args, model_dict, 
                            device_dict, net_dataidx_map, message_acc_dict,
                            message_loss_dict, worker_number)
        

        avg_acc, avg_loss = get_avg_metrics(message_acc_dict, message_loss_dict, comm_round)
        
        wandb.log({"Test/Loss": avg_loss, "epoch": comm_round + 1})
        wandb.log({"Test/AccTop1": avg_acc, "epoch": comm_round + 1})
        
    safely_savemodels("./models/{}_mdhmin_{}".format(str(args.running_name),
                                                     str(time.localtime(time.time()).tm_mon) + 
                                                     str(time.localtime(time.time()).tm_mday) + 
                                                     str(time.localtime(time.time()).tm_hour) + 
                                                     str(time.localtime(time.time()).tm_min)), 
                                                     model_dict,
                                                     worker_number)