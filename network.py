# -*- coding: utf-8 -*-
"""
Created on February 2023

@author: Arnaud Yarga
"""

import time
import torch
import numpy as np
from tqdm import tqdm
from neurons import LIF
from neurons import ParaLIF
import torch.nn.functional as F
    
    
def create_network(params, device):
    """
    This function creates a neural network based on the given parameters
    """
    neuron = params["neuron"]
    nb_layers = params["nb_layers"]
    input_size = params["input_size"]
    hidden_size = params["hidden_size"]
    nb_class = params["nb_class"]
    tau_mem = params["tau_mem"]
    tau_syn = params["tau_syn"]
    recurrent = params["recurrent"]

    modules = []
    if neuron in ["LIF", "LIF-LT"]:
        learn_th = neuron=="LIF-LT"
        modules.append(LIF(input_size, hidden_size, device, recurrent=recurrent, tau_mem=tau_mem, tau_syn=tau_syn, learn_th=learn_th))
        for i in range(nb_layers-1):
            modules.append(LIF(hidden_size, hidden_size, device, recurrent=recurrent, tau_mem=tau_mem, tau_syn=tau_syn, learn_th=learn_th))
        modules.append(LIF(hidden_size, nb_class, device, tau_mem=tau_mem, tau_syn=tau_syn, fire=False))
        
    elif neuron.split('-')[0]=="ParaLIF":
        spike_mode = neuron.split('-')[-1]
        modules.append(ParaLIF(input_size, hidden_size, device, spike_mode, recurrent=recurrent, tau_mem=tau_mem, tau_syn=tau_syn))
        for i in range(nb_layers-1):
            modules.append(ParaLIF(hidden_size, hidden_size, device, spike_mode, recurrent=recurrent, tau_mem=tau_mem, tau_syn=tau_syn))
        modules.append(ParaLIF(hidden_size, nb_class, device, spike_mode, tau_mem=tau_mem, tau_syn=tau_syn, fire=False))

    model = torch.nn.Sequential(*modules)
    return model


def train(model, data_loader, nb_epochs=100, loss_mode='mean', reg_thr=0., reg_thr_r=0., optimizer=None, lr=1e-3, weight_decay=0.):
    """
    This function Train the given model on the train data.
    """
    model.train()
    optimizer = optimizer if optimizer else torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=weight_decay)
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
            _,y_pred = torch.max(output,1)
            local_acc += torch.sum((y==y_pred)).detach().cpu().numpy()
            progress_bar.set_postfix(loss=local_loss/total, accuracy=local_acc/total, _batch=f"{i_batch+1}/{nb_batch}")
        
        loss_hist.append(local_loss/total)
        acc_hist.append(local_acc/total)

    train_duration = (time.time()-start_time)/nb_epochs
    
    return {'loss':loss_hist, 'acc':acc_hist, 'dur':train_duration}



def test(model, data_loader, loss_mode='mean'):
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






def train_test(model, data_loader_train, data_loader_test, nb_epochs, loss_mode, reg_thr, reg_thr_r, lr=1e-3, weight_decay=0.):
    loss_all, train_acc_all, train_dur_all = [],[],[]
    test_acc_all, test_spk_all, test_dur_all = [],[],[]  
    
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=weight_decay)
    for i in range(nb_epochs):
        print(f'\nEpoch: {i}/{nb_epochs}')
        res_train = train(model, data_loader_train, nb_epochs=1, loss_mode=loss_mode, reg_thr=reg_thr, reg_thr_r=reg_thr_r, optimizer=optimizer)
        loss_all += res_train['loss']
        train_acc_all += res_train['acc']
        train_dur_all.append(res_train['dur'])
        res_test = test(model, data_loader_test, loss_mode=loss_mode)
        test_acc_all.append(res_test['acc'])
        test_spk_all.append(res_test['spk'])
        test_dur_all.append(res_test['dur'])
        
    return (
            {'loss':loss_all, 'acc':train_acc_all, 'dur': np.mean(train_dur_all)},
            {'acc':test_acc_all, 'spk':test_spk_all, 'dur': np.mean(test_dur_all)},
        )

