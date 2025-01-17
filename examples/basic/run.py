"""
Created on February 2023

@author: Arnaud Yarga

This script train and evaluate ParaLIF and LIF networks over various datasets.
"""

import os
import json
import torch
import random
import argparse
import numpy as np
from datetime import datetime
from datasets import Dataset
from network import create_network, train_test





# --------------------------------------------- SHD

def get_configs_shd(PARAMS):
    nb_layers,hidden_size = [1,256] if PARAMS.recurrent else [3,128]
    #PARAMS.seed = 0
    PARAMS.dir = f"heidelberg/{'recurrent' if PARAMS.recurrent else 'feedforward'}/{PARAMS.neuron}/"
    PARAMS.dataset = "heidelberg"
    PARAMS.window_size =  1e-3
    PARAMS.batch_size = 64
    PARAMS.nb_epochs = 200
    PARAMS.nb_layers = nb_layers
    PARAMS.hidden_size = hidden_size
    PARAMS.data_augmentation = True
    PARAMS.save_model = True
    return PARAMS

# --------------------------------------------- ntidigits

def get_configs_ntidigits(PARAMS):
    nb_layers,hidden_size = [1,256] if PARAMS.recurrent else [3,128]
    #PARAMS.seed = 0
    PARAMS.dir = f"ntidigits/{'recurrent' if PARAMS.recurrent else 'feedforward'}/{PARAMS.neuron}/"
    PARAMS.dataset = "ntidigits"
    PARAMS.window_size =  1e-3
    PARAMS.batch_size = 64
    PARAMS.nb_epochs = 300
    PARAMS.nb_layers = nb_layers
    PARAMS.hidden_size = hidden_size
    PARAMS.data_augmentation = False
    PARAMS.save_model = True
    return PARAMS

# --------------------------------------------- dvsgesture

def get_configs_dvsgesture(PARAMS):
    nb_layers,hidden_size = [1,128] if PARAMS.recurrent else [2,128]
    #PARAMS.seed = 0
    PARAMS.dir = f"dvsgesture/{'recurrent' if PARAMS.recurrent else 'feedforward'}/{PARAMS.neuron}/"
    PARAMS.dataset = "dvsgesture"
    PARAMS.window_size =  5e-3
    PARAMS.batch_size = 16
    PARAMS.nb_epochs = 50
    PARAMS.nb_layers = nb_layers
    PARAMS.hidden_size = hidden_size
    PARAMS.data_augmentation = True
    PARAMS.save_model = True
    return PARAMS


# --------------------------------------------- NMNIST

def get_configs_nmnist(PARAMS):
    nb_layers,hidden_size = [1,128] if PARAMS.recurrent else [2,128]
    #PARAMS.seed = 0
    PARAMS.dir = f"nmnist/{'recurrent' if PARAMS.recurrent else 'feedforward'}/{PARAMS.neuron}/"
    PARAMS.dataset = "nmnist"
    PARAMS.window_size =  5e-3
    PARAMS.batch_size = 256
    PARAMS.nb_epochs = 50
    PARAMS.nb_layers = nb_layers
    PARAMS.hidden_size = hidden_size
    PARAMS.data_augmentation = False
    PARAMS.save_model = True
    return PARAMS

# --------------------------------------------- yinyang

def get_configs_yinyang(PARAMS):
    window_size = [1e4, 5e3, 1e3, 1e2, 1e1][0] # Window size will define the input spike train length
    #PARAMS.seed = 0
    PARAMS.dir = f"yinyang/input_size/feedforward/{PARAMS.neuron}/"
    PARAMS.dataset = "yinyang"
    PARAMS.window_size =  window_size
    PARAMS.batch_size = 64
    PARAMS.nb_epochs = 20
    PARAMS.recurrent = False
    PARAMS.nb_layers = 1
    PARAMS.hidden_size = 128
    PARAMS.data_augmentation = False
    PARAMS.save_model = True
    return PARAMS

# ---------------------------------------------

def save_results(train_results, test_results, PARAMS, model):
    """
    This function creates a dictionary of results from the training and testing and save it
    as a json file. 
    If the 'save_model' parameter is set to True, the trained model is also saved. 
    """
    PARAMS.device = str(PARAMS.device)
    outputs = {
       'loss_hist':train_results['loss'],
       'train_accuracies':train_results['acc'], 
       'train_duration':train_results['dur'], 
       'test_accuracies': test_results['acc'],
       'nb_spikes':test_results['spk'],
       'test_duration':test_results['dur'],
       'PARAMS': vars(PARAMS)
      }
    
    output_dir = f"outputs/{PARAMS.dir}"
    timestamp = int(datetime.timestamp(datetime.now()))
    filename = output_dir+f"results_{PARAMS.neuron}_{str(timestamp)}.json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(outputs, f)
    
    if PARAMS.save_model: 
        modelname = output_dir+f"model_{PARAMS.neuron}_{str(timestamp)}.pt"
        torch.save(model.state_dict(), modelname)



def main(PARAMS):
    """
    This function :
        - Enable or not the reproductibility by setting a seed
        - Loads the train and test sets
        - Create the network
        - Train and test the network
        - Save the results
    """
    print("\n-- Start --\n")
    if PARAMS.best_config:
        if PARAMS.dataset == "heidelberg":
            PARAMS = get_configs_shd(PARAMS)
        elif PARAMS.dataset == "ntidigits":
            PARAMS = get_configs_ntidigits(PARAMS)
        elif PARAMS.dataset == "nmnist":
            PARAMS = get_configs_nmnist(PARAMS)
        elif PARAMS.dataset == "dvsgesture":
            PARAMS = get_configs_dvsgesture(PARAMS)
        elif PARAMS.dataset == "yinyang":
            PARAMS = get_configs_yinyang(PARAMS)
    print(PARAMS)
            
            
    #To enable reproductibility
    if PARAMS.seed is not None:
        seed=PARAMS.seed
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PARAMS.device = device
    
    # Loads the train and test sets
    dataset_class = Dataset(window_size=PARAMS.window_size, device=device, augment=PARAMS.data_augmentation, 
                            shift=PARAMS.shift, scale=PARAMS.scale)
    (train_set, test_set, input_size, nb_class, collate_fn_train, collate_fn_test) = dataset_class.create_dataset(PARAMS.dataset)
    PARAMS.input_size=input_size
    PARAMS.nb_class=nb_class
    
    if PARAMS.debug :
        batch_size = PARAMS.batch_size
        train_set.data = train_set.data[:2*batch_size]
        train_set.targets = train_set.targets[:2*batch_size]
        test_set.data = test_set.data[:2*batch_size]
        test_set.targets = test_set.targets[:2*batch_size]
        
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=PARAMS.batch_size, shuffle=True, collate_fn=collate_fn_train)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=PARAMS.batch_size, shuffle=False, collate_fn=collate_fn_test)

    model = create_network(PARAMS)
    print(model, sum(p.numel() for p in model.parameters() if p.requires_grad))
    

    # Train and test the network
    print("\n-- Training - Testing --\n")
    train_results, test_results = train_test(model, train_loader, test_loader, PARAMS.nb_epochs, 
                                PARAMS.loss_mode, PARAMS.reg_thr, PARAMS.reg_thr_r, 
                                lr=PARAMS.lr, weight_decay=PARAMS.weight_decay, eval_each_epoch=PARAMS.nb_epochs//1)
    
    
    # Save train and test results
    save_results(train_results, test_results, PARAMS, model)
    print("\n-- End --\n")
    
def get_argparser():
    parser = argparse.ArgumentParser(description="SNN training")
    parser.add_argument('--seed', type=int)
    parser.add_argument('--dataset', type=str, default='heidelberg', choices=["heidelberg", "ntidigits", "nmnist", "dvsgesture", "yinyang"])
    parser.add_argument('--neuron', type=str, default="ParaLIF-D", choices=["LIF","LIF-LT","ParaLIF-GS","ParaLIF-SB","ParaLIF-TRB","ParaLIF-D","ParaLIF-SD","ParaLIF-TD","ParaLIF-T","ParaLIF-ST","ParaLIF-TT"])
    parser.add_argument('--nb_epochs', type=int, default=200)
    parser.add_argument('--tau_mem', type=float, default=2e-2, help='neuron membrane time constant')
    parser.add_argument('--tau_syn', type=float, default=2e-2, help='neuron synaptic current time constant')
    parser.add_argument('--learn_tau', action="store_true", default=False)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=128, help='nb of neurons in the hidden layer')
    parser.add_argument('--nb_layers', type=int, default=3, help='nb of hidden layers')
    parser.add_argument('--recurrent', action="store_true", default=False)
    parser.add_argument('--recurrent_relu', action="store_true", default=False)
    parser.add_argument('--reg_thr', type=float, default=0., help='spiking frequency regularization threshold')
    parser.add_argument('--reg_thr_r', type=float, default=0., help='spiking frequency regularization threshold')
    parser.add_argument('--loss_mode', type=str, default='mean', choices=["last", "max", "mean", "cumsum"])
    parser.add_argument('--data_augmentation', action="store_true", default=False)
    parser.add_argument('--shift', type=float, default=0.05, help='data augmentation random shift factor')
    parser.add_argument('--scale', type=float, default=0.15, help='data augmentation random scale factor')
    parser.add_argument('--dir', type=str, default='')
    parser.add_argument('--save_model', action="store_true", default=False)
    parser.add_argument('--debug', action="store_true", default=False)
    parser.add_argument('--window_size', type=float, default=1e-3)
    
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0., help='weight decay')
    
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--best_config", action="store_true", default=False)
    return parser

if __name__ == '__main__':
    parser = get_argparser()
    main(parser.parse_args())




