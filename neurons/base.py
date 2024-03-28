# -*- coding: utf-8 -*-
"""
Created on December 2023

@author: Arnaud Yarga
"""

import torch
import numpy as np



class Base(torch.nn.Module):
    """
    Base class for creating a spiking neural network using PyTorch.

    Parameters:
    - input_size (int): size of input tensor
    - hidden_size (int): size of hidden layer
    - device (torch.device): device to use for tensor computations, such as 'cpu' or 'cuda'
    - recurrent (bool): flag to determine if the neurons should be recurrent
    - fire (bool): flag to determine if the neurons should fire spikes or not
    - tau_mem (float): time constant for the membrane potential
    - tau_syn (float): time constant for the synaptic potential
    - time_step (float): step size for updating the LIF model
    - debug (bool): flag to turn on/off debugging mode
    """
    def __init__(self, input_size, hidden_size, device, recurrent,
                 fire, tau_mem, tau_syn, time_step, debug):
        super(Base, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.recurrent = recurrent
        self.v_th = torch.tensor(1.0)
        self.fire = fire
        self.debug = debug
        self.nb_spike_per_neuron = torch.zeros(self.hidden_size, device=self.device)

        # Neuron time constants
        self.alpha = float(np.exp(-time_step/tau_syn))
        self.beta = float(np.exp(-time_step/tau_mem))

        # Fully connected layer for feedforward synapses
        self.fc = torch.nn.Linear(self.input_size, self.hidden_size, device=self.device)
        # Initializing weights
        torch.nn.init.kaiming_uniform_(self.fc.weight, a=0, mode='fan_in', nonlinearity='linear')
        torch.nn.init.zeros_(self.fc.bias)
        if self.debug: torch.nn.init.ones_(self.fc.weight)
        
        # Fully connected for recurrent synapses 
        if self.recurrent:
            self.fc_recu = torch.nn.Linear(self.hidden_size, self.hidden_size, device=self.device)
            # Initializing weights
            torch.nn.init.kaiming_uniform_(self.fc_recu.weight, a=0, mode='fan_in', nonlinearity='linear')
            torch.nn.init.zeros_(self.fc_recu.bias)
            if self.debug: torch.nn.init.ones_(self.fc_recu.weight)




# Surrogate gradient implementation from https://github.com/fzenke/spytorch/blob/main/notebooks/SpyTorchTutorial1.ipynb
class SurrGradSpike(torch.autograd.Function):
    scale = 100.0
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
        return grad
        

