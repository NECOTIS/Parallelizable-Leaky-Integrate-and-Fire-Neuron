# -*- coding: utf-8 -*-
"""
Created on December 2023

@author: Arnaud Yarga
"""

import os
import json
import torch
import random
import argparse
import numpy as np
from datetime import datetime
from datasets import Dataset
from network import create_network, train, test, train_test


parser = argparse.ArgumentParser(description="SNN training")
parser.add_argument('--seed', type=int)
parser.add_argument('--dataset', type=str, default='heidelberg', choices=["heidelberg", "ntidigits", "nmnist", "dvsgesture", "yinyang"])
parser.add_argument('--neuron', type=str, default='LIF')
parser.add_argument('--nb_epochs', type=int, default=200)
parser.add_argument('--tau_mem', type=float, default=2e-2, help='neuron membrane time constant')
parser.add_argument('--tau_syn', type=float, default=2e-2, help='neuron synaptic current time constant')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--hidden_size', type=int, default=128, help='nb of neurons in the hidden layer')
parser.add_argument('--nb_layers', type=int, default=3, help='nb of hidden layers')
parser.add_argument('--recurrent', action="store_true", default=False, help="Whether to use recurrent architecture or not")
parser.add_argument('--reg_thr', type=float, default=0., help='spiking frequency regularization threshold')
parser.add_argument('--reg_thr_r', type=float, default=0., help='spiking frequency regularization threshold for recurrent spikes')
parser.add_argument('--loss_mode', type=str, default='mean', choices=["last", "max", "mean", "cumsum"])
parser.add_argument('--data_augmentation', action="store_true", default=False)
parser.add_argument('--shift', type=float, default=0.05, help='data augmentation random shift factor')
parser.add_argument('--scale', type=float, default=0.15, help='data augmentation random scale factor')
parser.add_argument('--window_size', type=float, default=1e-3)
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0., help='weight decay')

parser.add_argument('--dir', type=str, default='')
parser.add_argument('--save_model', action="store_true", default=False)
parser.add_argument('--debug', action="store_true", default=False)
parser.add_argument('--best_config', action="store_true", default=False)


args = parser.parse_args()
PARAMS = {
    "seed" : args.seed,
    "dataset" : args.dataset,
    "neuron" : args.neuron,
	"nb_epochs" : args.nb_epochs,
    "tau_mem" : args.tau_mem,
    "tau_syn" : args.tau_syn,
    "batch_size" : args.batch_size,
    "hidden_size" : args.hidden_size,
    "nb_layers" : args.nb_layers,
    "recurrent" : args.recurrent,
    "reg_thr" : args.reg_thr,
    "reg_thr_r" : args.reg_thr_r,
    "loss_mode" : args.loss_mode,
    "data_augmentation" : args.data_augmentation,
	"shift" : args.shift,
    "scale" : args.scale,
    "window_size" : args.window_size,
    "lr" : args.lr,
    "weight_decay" : args.weight_decay,
    "dir" : args.dir,
	"save_model" : args.save_model,
    "debug" : args.debug,
	"best_config" : args.best_config,
   }



def get_best_configs(dataset):
    CONFIGS = {}
    
    CONFIGS["dvsgesture"] = {
        'neuron': "ParaLIF-SD",
         "dir": "dvsgesture/",
         "dataset": "dvsgesture",
         "window_size" : 5e-3,
         "batch_size": 16,
         "nb_epochs": 50,
         'recurrent': False,
         'nb_layers': 2,
         'hidden_size': 128,
         'data_augmentation': True,
     }
    
    CONFIGS["nmnist"] = {
        'neuron': "ParaLIF-SD",
         "dir": "nmnist/",
         "dataset": "nmnist",
         "window_size" : 1e-3,
         "batch_size": 256,
         "nb_epochs": 50,
         'recurrent': False,
         'nb_layers': 2,
         'hidden_size': 128,
         'data_augmentation': False,
     }
    
    CONFIGS["ntidigits"] = {
        'neuron': "ParaLIF-D",
         "dir": "ntidigits/",
         "dataset": "ntidigits",
         "window_size" : 1e-3,
         "batch_size": 64,
         "nb_epochs": 300,
         'recurrent': True,
         'nb_layers': 1,
         'hidden_size': 256,
         'data_augmentation': False,
     }
    
    CONFIGS["heidelberg"] = {
        'neuron': "ParaLIF-SD",
         "dir": "heidelberg/",
         "dataset": "heidelberg",
         "window_size" : 1e-3,
         "batch_size": 64,
         "nb_epochs": 200,
         'recurrent': True,
         'nb_layers': 1,
         'hidden_size': 256,
         'data_augmentation': True,
     }
    
    CONFIGS["yinyang"] = {
        'neuron': "ParaLIF-GS",
         "dir": "yinyang/",
         "dataset": "yinyang",
         "window_size" : 1e4,
         "batch_size": 64,
         "nb_epochs": 20,
         'recurrent': False,
         'nb_layers': 1,
         'hidden_size': 128,
         'data_augmentation': False,
     }
    
    return CONFIGS[dataset]



# ---------------------------------------------

def save_results(train_results, test_results, PARAMS, model):
    """
    This function creates a dictionary of results from the training and testing and save it
    as a json file. 
    If the 'save_model' parameter is set to True, the trained model is also saved. 
    """
    outputs = {
       'train_accuracies':train_results['acc'], 
       'train_duration':train_results['dur'], 
       'test_accuracies': test_results['acc'],
       'nb_spikes':test_results['spk'],
       'test_duration':test_results['dur'],
       'PARAMS': PARAMS
      }
    
    output_dir = f"outputs/{PARAMS['dir']}"
    timestamp = int(datetime.timestamp(datetime.now()))
    filename = output_dir+f"results_{PARAMS['neuron']}_{str(timestamp)}.json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(outputs, f)
    
    if PARAMS['save_model']: 
        modelname = output_dir+f"model_{PARAMS['neuron']}_{str(timestamp)}.pt"
        torch.save(model.state_dict(), modelname)



def main():
    """
    This function :
        - Enable or not the reproductibility by setting a seed
        - Loads the train and test sets
        - Create the network
        - Train and test the network
        - Save the results
    """
    print("\n-- Start --\n")
    
    if PARAMS["best_config"]:
        for k,v in get_best_configs(PARAMS["dataset"]).items():
            PARAMS[k] = v
    print(PARAMS)
            
            
    #To enable reproductibility
    if PARAMS["seed"] is not None:
        seed=PARAMS["seed"]
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Loads the train and test sets
    dataset_class = Dataset(window_size=PARAMS["window_size"], device=device, augment=PARAMS["data_augmentation"], 
                            shift=PARAMS["shift"], scale=PARAMS["scale"])
    (train_set, test_set, input_size, nb_class, collate_fn) = dataset_class.create_dataset(PARAMS["dataset"])
    PARAMS["input_size"] = input_size
    PARAMS["nb_class"] = nb_class
    
    if PARAMS["debug"]:
        PARAMS['nb_epochs'] = 10
        batch_size = PARAMS['batch_size']
        train_set.data = train_set.data[:2*batch_size]
        train_set.targets = train_set.targets[:2*batch_size]
        test_set.data = test_set.data[:2*batch_size]
        test_set.targets = test_set.targets[:2*batch_size]
            
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=PARAMS['batch_size'], shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=PARAMS['batch_size'], shuffle=False, collate_fn=collate_fn)
    

    # Create the network
    model = create_network(PARAMS, device)
    

    # Train and test the network
    print("\n-- Training --\n")
    train_results = train(model, train_loader, nb_epochs=PARAMS['nb_epochs'], loss_mode=PARAMS['loss_mode'], 
                          reg_thr=PARAMS['reg_thr'], reg_thr_r=PARAMS['reg_thr_r'], lr=PARAMS['lr'], weight_decay=PARAMS['weight_decay'])
    print("\n-- Testing --\n")
    test_results = test(model, test_loader, loss_mode=PARAMS['loss_mode'])

    # Save train and test results
    save_results(train_results, test_results, PARAMS, model)
    print("\n-- End --\n")
    


if __name__ == "__main__":
    main()




