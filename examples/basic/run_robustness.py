# -*- coding: utf-8 -*-
"""
Created on December 2024

@author: Arnaud Yarga

This script evaluate generalization and robustness of ParaLIF and LIF on SHD dataset.
Classification accuracy was evaluated on the training set, the clean
test set, and noisy test sets with varying noise levels. Noise
was introduced by randomly flipping bits in the input spike
train based on the desired signal-to-noise ratio (SNR), ensuring
the spike train remained binary. The same experiments can be
repeated with noise added to the training set.
"""

import os
import json
import torch
import random
import argparse
import numpy as np
from datetime import datetime
from datasets import Dataset
from network import create_network
from tqdm import tqdm
import time
import torch.nn.functional as F



def get_configs(PARAMS):
    recurrent,nb_layers,hidden_size = [False,3,128]
    #PARAMS.seed = 0
    PARAMS.dir = f"heidelberg/train_robustness/noise_Flip_{PARAMS.train_noise_level}/{'recurrent' if recurrent else 'feedforward'}/{PARAMS.neuron}/robustness_"
    PARAMS.dataset = "heidelberg"
    PARAMS.window_size =  1e-3
    PARAMS.batch_size = 64
    PARAMS.nb_epochs = 100
    PARAMS.noise_type = "Flip"
    PARAMS.recurrent = recurrent
    PARAMS.nb_layers = nb_layers
    PARAMS.hidden_size = hidden_size
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
        PARAMS = get_configs(PARAMS)
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
    

    # Create the network
    model = create_network(PARAMS)
    print(model)
    if PARAMS.checkpoint!= "":
        model.load_state_dict(torch.load(PARAMS.checkpoint, map_location=PARAMS.device))
    

    # Train and test the network
    print("\n-- Training - Testing --\n")
    def add_white_noise_to_spike_train(spike_train, snr_db):
        if snr_db==None: return spike_train
        # Calculate the signal power (fraction of 1's in the spike_train)
        signal_power = torch.mean(spike_train.float())**2
        # Convert SNR from dB to linear scale
        snr_linear = 10 ** (snr_db / 10)
        # Calculate the noise power based on the desired SNR (in linear scale)
        noise_power = torch.sqrt(signal_power / snr_linear)
        # Create a random tensor with the same shape as the spike_train (with values between 0 and 1)
        random_noise = torch.rand_like(spike_train.float())
        should_flip = (random_noise < noise_power)
        # Flip the bits where the random noise is less than the flip probability
        noisy_spike_train = torch.logical_not(should_flip).float()*spike_train + torch.logical_and(should_flip, torch.logical_not(spike_train)).float()
        # Return the noisy spike_train (still binary, values 0 or 1)
        return noisy_spike_train
    
    def train(model, data_loader, nb_epochs=100, loss_mode='mean', reg_thr=0., reg_thr_r=0., optimizer=None, lr=1e-3, weight_decay=0., lr_scheduler=None, noise_level=None, noise_type=""):
        """
        This function Train the given model on the train data.
        """
        model.train()
        optimizer = optimizer if optimizer else torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=weight_decay)
        #lr_scheduler = lr_scheduler if lr_scheduler else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, nb_epochs)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # If a regularization threshold is set we compute the theta_reg*N parameter of Equation (21)
        if reg_thr>0 or reg_thr_r>0: 
            N = np.sum([layer.hidden_size for layer in model if (layer.__class__.__name__ in ['LIF', 'ParaLIF'] and layer.fire)])
            reg_thr_sum = reg_thr * N
            reg_thr_sum_r = reg_thr_r * N

        loss_hist = []
        acc_hist = []
        progress_bar = tqdm(range(nb_epochs), desc=f"Train {nb_epochs} epochs")
        start_time = time.time()
        # Loop over the number of epochs
        for i_epoch in progress_bar:
            local_loss = 0
            local_acc = 0
            total = 0
            nb_batch = len(data_loader)
            # Loop over the batches
            for i_batch,(x,y) in enumerate(data_loader):
                total += len(y)
                if noise_level is not None:
                    if noise_type=="Flip":
                        x = add_white_noise_to_spike_train(x, noise_level)
                output = model(x)
                # Select the relevant function to process the output based on loss mode
                if loss_mode=='last' : output = output[:,-1,:]
                elif loss_mode=='max': output = torch.max(output,1)[0] 
                elif loss_mode=='cumsum': output = F.softmax(output,dim=2).sum(1)
                else: output = torch.mean(output,1)

                # Here we set up our regularizer loss as in Equation (21)
                reg_loss_val = 0
                if reg_thr>0:
                    spks = torch.stack([layer.nb_spike_per_neuron.sum() for layer in model if (layer.__class__.__name__ in ['LIF', 'ParaLIF'] and layer.fire)])
                    reg_loss_val += F.relu(spks.sum()-reg_thr_sum)**2
                if reg_thr_r>0:
                    spks_r = torch.stack([layer.nb_spike_per_neuron_rec.sum() for layer in model if (layer.__class__.__name__ in ['ParaLIF'] and layer.fire)])
                    reg_loss_val += F.relu(spks_r.sum()-reg_thr_sum_r)**2

                # Here we combine supervised loss and the regularizer
                loss_val = loss_fn(output, y) + reg_loss_val

                # Backpropagation and weights update
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

                local_loss += loss_val.detach().cpu().item()
                if len(y.shape)==2: y = y.argmax(1)
                local_acc += torch.sum((y==output.argmax(1))).detach().cpu().numpy()
                progress_bar.set_postfix(loss=local_loss/total, accuracy=local_acc/total, _batch=f"{i_batch+1}/{nb_batch}")
            
            loss_hist.append(local_loss/total)
            acc_hist.append(local_acc/total)

            if lr_scheduler:
                lr_scheduler.step()
        train_duration = (time.time()-start_time)/nb_epochs
        return {'loss':loss_hist, 'acc':acc_hist, 'dur':train_duration}

    
    def test(model, data_loader, loss_mode='mean', noise_level=None, noise_type="Flip"):
        """
        This function Computes classification accuracy for the given model on the test data.
        """
        model.eval()
        acc = 0
        total = 0
        spk_per_layer = []
        spk_per_layer_r = []
        progress_bar = tqdm(data_loader, desc="Test")
        start_time = time.time()
        # loop through the test data
        for x,y in progress_bar:
            total += len(y)
            with torch.no_grad():
                if noise_level is not None:
                    if noise_type=="Flip":
                        x = add_white_noise_to_spike_train(x, noise_level)
                output = model(x)
                # Select the relevant function to process the output based on loss mode
                if loss_mode=='last' : output = output[:,-1,:]
                elif loss_mode=='max': output = torch.max(output,1)[0] 
                elif loss_mode=='cumsum': output = F.softmax(output,dim=2).sum(1)
                else: output = torch.mean(output,1)
                # get the predicted label
                _,y_pred = torch.max(output,1)
                acc += torch.sum((y==y_pred)).cpu().numpy()
                # get the number of spikes per layer for LIF and ParaLIF layers
                spk_per_layer.append([layer.nb_spike_per_neuron.sum().cpu().item() for layer in model if (layer.__class__.__name__ in ['LIF', 'ParaLIF'] and layer.fire)])
                spk_per_layer_r.append([layer.nb_spike_per_neuron_rec.sum().cpu().item() for layer in model if (layer.__class__.__name__ in ['ParaLIF'] and layer.fire)])
                progress_bar.set_postfix(accuracy=acc/total)
        test_duration = (time.time()-start_time)
        
        return {'acc':acc/total, 'spk':[np.mean(spk_per_layer,axis=0).tolist(), np.mean(spk_per_layer_r,axis=0).tolist()], 'dur':test_duration}
    
    noise_range = [None, 0, 5, 10, 15, 20]
    print("Train results")
    train_results = train(model, train_loader, PARAMS.nb_epochs,
                                PARAMS.loss_mode, PARAMS.reg_thr, PARAMS.reg_thr_r, 
                                lr=PARAMS.lr, weight_decay=PARAMS.weight_decay, 
                                noise_level=PARAMS.train_noise_level, noise_type=PARAMS.noise_type)
    print("Test results")
    final_test_results = {}
    for noise_level in noise_range:
        final_test_results[noise_level] = []
        for repeat in range(5):
            test_results_ = test(model, test_loader, PARAMS.loss_mode, noise_level=noise_level, noise_type=PARAMS.noise_type)
            final_test_results[noise_level].append(test_results_['acc'])
        print(f"Noise level {noise_level}, mean acc {np.mean(final_test_results[noise_level])}, std acc {np.std(final_test_results[noise_level])}")
    # Save train and test results
    save_results(train_results, {'acc':final_test_results, 'spk':None, 'dur':None}, PARAMS, model)
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
    parser.add_argument('--checkpoint', type=str, default="")
    parser.add_argument('--noise_type', type=str, default="Flip")
    parser.add_argument('--train_noise_level', type=float, default=None)
    return parser

if __name__ == '__main__':
    parser = get_argparser()
    main(parser.parse_args())




