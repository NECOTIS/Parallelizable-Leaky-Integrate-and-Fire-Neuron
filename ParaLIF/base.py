# -*- coding: utf-8 -*-
"""
Created on August 2024

@author: Arnaud Yarga
"""

import torch
import numpy as np


class Base(torch.nn.Module):
    """
    Base class for creating a spiking neural layer using PyTorch.

    Parameters:
    - n_neuron (int): Number of neurons in the layer.
    - recurrent (bool): Flag to determine if the neurons should be recurrent.
    - fire (bool): Flag to determine if the neurons should fire spikes.
    - recurrent_fire (bool): Flag to determine if recurrent neurons should fire.
    - spk_threshold (float or list): Spiking threshold(s) for the neurons. 
                                     Can be a single value or a list with length equal to n_neuron.
    - learn_threshold (bool): Whether the spiking threshold is learnable.
    - tau_mem (float or list): Membrane time constant. Can be a single value or a list of values.
    - tau_syn (float or list): Synaptic time constant. Can be a single value or a list of values.
    - time_step (float): Step size for updating the neuron model.
    - learn_tau (bool): Whether the time constants are learnable.
    - device (torch.device): Device to use for tensor computations, such as 'cpu' or 'cuda'.
    - debug (bool): Flag to turn on/off debugging mode.
    """
    def __init__(self, n_neuron, recurrent, fire, recurrent_fire, spk_threshold, learn_threshold, tau_mem, tau_syn, time_step, learn_tau, device, debug):
        
        super(Base, self).__init__()
        # Validate input sizes if lists are provided for spk_threshold, tau_mem, or tau_syn
        if type(spk_threshold) == list:
            assert len(spk_threshold) == n_neuron, f"'spk_threshold' size ({len(spk_threshold)}) should be the same as 'n_neuron' ({n_neuron})"
        if type(tau_mem) == list:
            assert len(tau_mem) == n_neuron, f"'tau_mem' size ({len(tau_mem)}) should be the same as 'n_neuron' ({n_neuron})"
        if type(tau_syn) == list:
            assert len(tau_syn) == n_neuron, f"'tau_syn' size ({len(tau_syn)}) should be the same as 'n_neuron' ({n_neuron})"
        
        self.n_neuron = n_neuron
        self.recurrent = recurrent
        self.fire = fire
        self.recurrent_fire = recurrent_fire
        self.learn_threshold = learn_threshold
        self.learn_tau = learn_tau
        self.device = device
        self.debug = debug
        self.register_buffer('nb_spike_per_neuron', torch.zeros(self.n_neuron, device=self.device))
        
        # Initialize spiking threshold parameter
        if self.learn_threshold:
            self._spk_threshold = torch.nn.Parameter(torch.tensor(spk_threshold, device=self.device))
        else:
            self.register_buffer('_spk_threshold', torch.tensor(spk_threshold, device=self.device))
        
        # Neuron time constants for synaptic and membrane potentials
        alpha = torch.exp(-time_step / torch.tensor(tau_syn, device=self.device)) if tau_syn is not None else torch.tensor(0., device=self.device)
        beta = torch.exp(-time_step / torch.tensor(tau_mem, device=self.device))
        if self.learn_tau:
            if tau_syn is not None: self._alpha = torch.nn.Parameter(alpha)  
            else : self.register_buffer('_alpha', alpha) #if tau_syn is None its not learned
            self._beta = torch.nn.Parameter(beta)
        else:
            self.register_buffer('_alpha', alpha)
            self.register_buffer('_beta', beta)
            
        # Fully connected layer for recurrent synapses
        if self.recurrent:
            self.fc_recu = torch.nn.Linear(self.n_neuron, self.n_neuron, device=self.device)
            # Initialize weights for the recurrent layer
            torch.nn.init.kaiming_uniform_(self.fc_recu.weight, a=0, mode='fan_in', nonlinearity='linear')
            torch.nn.init.zeros_(self.fc_recu.bias)
            if self.debug: 
                torch.nn.init.ones_(self.fc_recu.weight)  # Optionally initialize to ones for debugging
    
    
    # Clamp the spk_threshold when learning to avoid negative thresholds
    @property
    def spk_threshold(self):
        return self._spk_threshold.relu() if self.learn_threshold else self._spk_threshold
    
    @spk_threshold.setter
    def spk_threshold(self, val):
        if isinstance(self._spk_threshold, torch.nn.Parameter):
            self._spk_threshold.data = val
        else:
            self._spk_threshold = val
    
    # Apply sigmoid as a clamp function to keep alpha in [0, 1] range as suggested in PLIF
    #  (https://arxiv.org/abs/2007.05785)
    @property
    def alpha(self):
        # if tau_syn is None, self._alpha will not be a parameter then alpha will stay zero
        return self._alpha.sigmoid() if (self.learn_tau and isinstance(self._alpha, torch.nn.Parameter)) else self._alpha
    
    @alpha.setter
    def alpha(self, val):
        if isinstance(self._alpha, torch.nn.Parameter):
            self._alpha.data = val
        else:
            self._alpha = val
            
    # Apply sigmoid as a clamp function to keep beta in [0, 1] range as suggested in PLIF
    #  (https://arxiv.org/abs/2007.05785)
    @property
    def beta(self):
        return self._beta.sigmoid() if self.learn_tau else self._beta
    
    @beta.setter
    def beta(self, val):
        if isinstance(self._beta, torch.nn.Parameter):
            self._beta.data = val
        else:
            self._beta = val

    def extra_repr(self):
        """
        Provides additional representation for the module, useful for printing.
        """
        return f"n_neuron={self.n_neuron}, recurrent={self.recurrent}, fire={self.fire}, recurrent_fire={self.recurrent_fire}, learn_threshold={self.learn_threshold}, learn_tau={self.learn_tau}"



# Fast Sigmoid Surrogate gradient implementation from https://github.com/fzenke/spytorch/blob/main/notebooks/SpyTorchTutorial1.ipynb
class SurrGradSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.save_for_backward(input)
        ctx.scale = scale
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (ctx.scale * torch.abs(input) + 1.0) ** 2
        return grad, None
        

# ATan surrogate gradient from https://proceedings.neurips.cc/paper/2021/hash/afe434653a898da20044041262b3ac74-Abstract.html
class AtanSurrGradSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.save_for_backward(input)
        ctx.scale = scale
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.scale
        grad = alpha / 2 / (1 + (np.pi / 2 * alpha * x).pow_(2)) * grad_output
        return grad, None




class SurrogateFunction:
    def __init__(self, mode="fastsig", scale=None):
        """
        Surrogate function selector class to handle different types of surrogate gradients.
        
        Parameters:
        - mode (str): Type of surrogate gradient function ('fastsig' or 'atan').
        - scale (float): Scaling factor for the surrogate gradient function. Default depends on mode.
        """
        super(SurrogateFunction, self).__init__()
        self.mode = mode
        if self.mode == "atan":
            self.surrogate_fn = AtanSurrGradSpike
            self.scale = 2.0
        else:
            self.surrogate_fn = SurrGradSpike
            self.scale = 100.0
        if scale:
            self.scale = scale
    
    def __call__(self, inputs):
        """
        Apply the surrogate gradient function to the inputs.
        
        Parameters:
        - inputs (torch.Tensor): The input tensor on which to apply the surrogate gradient function.
        
        Returns:
        - torch.Tensor: The output after applying the surrogate gradient.
        """
        return self.surrogate_fn.apply(inputs, self.scale)
