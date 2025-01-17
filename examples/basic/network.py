# -*- coding: utf-8 -*-
"""
Created on February 2023

@author: Arnaud Yarga
"""
import sys
sys.path.append('../ParaLIF')

import time
import torch
import numpy as np
from tqdm import tqdm
from ParaLIF import LIF,ParaLIF
import torch.nn.functional as F
import random
import argparse


class MyRandErase(torch.nn.Module):
    def __init__(self, p: float = 0.5, ratio: float = 0.5, vertical=True):
        super(MyRandErase, self).__init__()
        self.p = p
        self.ratio = ratio
        self.vertical = vertical

    def forward(self, X):
        if self.training:
            return torch.stack([self.line_erasing(x) for x in X])
        return X
    
    def line_erasing(self, x):
        if random.random() < self.p:
            h, w = x.size()
            if self.vertical:
                erase_width = int(self.ratio*w)
                # Erase vertical lines (columns)
                start = random.randint(0, w - erase_width)
                x[:, start:start + erase_width] = 0
            else:
                erase_width = int(self.ratio*h)
                # Erase horizontal lines (rows)
                start = random.randint(0, h - erase_width)
                x[start:start + erase_width, :] = 0
        return x
    

    
class DELAY(torch.nn.Module):
    def __init__(self, n_input, delay_max=10, device=None, mode="uniform"):
        super().__init__()
        delay_max = int(delay_max)
        if mode=="normal":
            self.register_buffer('delays', torch.clamp(torch.normal(0., delay_max//2, size=(n_input,), device=device).abs().long(), max=delay_max))
        else: 
            self.register_buffer('delays', torch.randint(0, delay_max+1, (n_input,), device=device).to(torch.long))
        self.register_buffer('delay_max', torch.tensor(delay_max))
        self.device = device
    
    def roll(self, x, shifts):
        indices = (torch.arange(x.shape[0], device=self.device)[:, None] - shifts[None, :]) % x.shape[0]
        return torch.gather(x, 0, indices.long())

    def forward(self, x):
        x = F.pad(x, (0,0,0,self.delay_max), "constant", 0)
        return torch.stack([self.roll(x_i, self.delays) for x_i in x])



    
    
def create_network(params):
    """
    This function creates a neural network based on the given parameters
    """
    device = params.device
    nb_layers = params.nb_layers
    input_size = params.input_size
    nb_class = params.nb_class
    neuron = params.neuron
    
    hidden_size = params.hidden_size if (type(params.hidden_size) == list) else (nb_layers)*[params.hidden_size]
    tau_mem = params.tau_mem if (type(params.tau_mem) == list) else (nb_layers+1)*[params.tau_mem]
    tau_syn = params.tau_syn if (type(params.tau_syn) == list) else (nb_layers+1)*[params.tau_syn]
    recurrent = params.recurrent if (type(params.recurrent) == list) else (nb_layers+1)*[params.recurrent]
    recurrent[-1] = False #output layer
    recurrent_fire = (not params.recurrent_relu) if "recurrent_relu" in vars(params) else True
    
    surrogate_mode = params.surrogate_mode if "surrogate_mode" in vars(params) else "fastsig"
    surrogate_scale = params.surrogate_scale if "surrogate_scale" in vars(params) else 100.
    learn_tau = params.learn_tau if "learn_tau" in vars(params) else False
    learn_th = neuron!="LIF"
    spk_threshold = params.spk_threshold if "spk_threshold" in vars(params) else 1.0
    delay = params.delay if "delay" in vars(params) else None
    if delay is not None and not isinstance(delay, (tuple, list)): delay = nb_layers*[delay]

    model = torch.nn.Sequential()
    
    if neuron.split('-')[0]=="ParaLIF":
        spike_mode = neuron.split('-')[-1]
        if "D" in spike_mode: spk_threshold = params.spk_threshold if "spk_threshold" in vars(params) else 0.1
    if isinstance(input_size, (tuple, list)):
        model.append(torch.nn.Flatten(2,-1))
        input_size = np.prod(input_size)
    for i in range(nb_layers+1):
        in_d = input_size if i==0 else hidden_size[i-1]
        out_d = nb_class if i==nb_layers else hidden_size[i]
        model.append(torch.nn.Linear(in_d, out_d, device=device))
        if neuron in ["LIF", "LIF-LT"]:
            model.append(LIF(out_d, recurrent=recurrent[i], fire=(i!=nb_layers), recurrent_fire=recurrent_fire, spk_threshold=spk_threshold, 
                         learn_threshold=learn_th, tau_mem=tau_mem[i], tau_syn=tau_syn[i], learn_tau=learn_tau, refractory=None,
                         device=device, surrogate_mode=surrogate_mode, surrogate_scale=surrogate_scale))
        elif neuron.split('-')[0]=="ParaLIF":
            model.append(ParaLIF(out_d, spike_mode, recurrent=recurrent[i], fire=(i!=nb_layers), recurrent_fire=recurrent_fire, spk_threshold=spk_threshold, 
                                 learn_threshold=learn_th, tau_mem=tau_mem[i], tau_syn=tau_syn[i], learn_tau=learn_tau,
                                 device=device, surrogate_mode=surrogate_mode, surrogate_scale=surrogate_scale))
        else:
            model.append(torch.nn.ReLU())
        if delay is not None and i!=nb_layers: model.append(DELAY(out_d, delay_max=delay[i], device=device, mode="normal"))

    for layer in model: 
        if layer.__class__.__name__ == 'Linear':
            torch.nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='linear')
            torch.nn.init.zeros_(layer.bias)

    return model





def train(model, data_loader, nb_epochs=100, loss_mode='mean', reg_thr=0., reg_thr_r=0., optimizer=None, lr=1e-3, weight_decay=0., lr_scheduler=None):
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






def train_test(model, data_loader_train, data_loader_test, nb_epochs, loss_mode, reg_thr, reg_thr_r, lr=1e-3, weight_decay=0., eval_each_epoch=10):
    loss_all, train_acc_all, train_dur_all = [],[],[]
    test_acc_all, test_spk_all, test_dur_all = [],[],[]  
    
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5) #torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, nb_epochs)
    for i in range(0,nb_epochs,eval_each_epoch):
        print(f'\nEpoch: {i}-{i+eval_each_epoch-1}/{nb_epochs}')
        res_train = train(model, data_loader_train, nb_epochs=eval_each_epoch, loss_mode=loss_mode, 
                          reg_thr=reg_thr, reg_thr_r=reg_thr_r, optimizer=optimizer, lr_scheduler=lr_scheduler)
        loss_all += res_train['loss']
        train_acc_all += res_train['acc']
        train_dur_all.append(res_train['dur'])
        if (i)%eval_each_epoch==0:
            res_test = test(model, data_loader_test, loss_mode=loss_mode)
            test_acc_all.append(res_test['acc'])
            test_spk_all.append(res_test['spk'])
            test_dur_all.append(res_test['dur'])
            if res_test['acc']<0.2 and i>=9:
                print("*********** Early stopped")
                break
        
    return (
            {'loss':loss_all, 'acc':train_acc_all, 'dur': np.mean(train_dur_all)},
            {'acc':test_acc_all, 'spk':test_spk_all, 'dur': np.mean(test_dur_all)},
        )

