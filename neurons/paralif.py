# -*- coding: utf-8 -*-
"""
Created on December 2023

@author: Arnaud Yarga
"""

import torch
from neurons import Base, SurrGradSpike
import torch.nn.functional as F


class ParaLIF(Base):
    """
    Class for implementing a Parallelizable Leaky Integrate-and-Fire (ParaLIF) neuron model

    Parameters:
    - input_size (int): The number of expected features in the input
    - hidden_size (int): The number of neurons on the layer
    - device (torch.device): device to use for tensor computations, such as 'cpu' or 'cuda'
    - spike_mode (str): "GS", "SB", "TRB", "D", "SD", "TD", "TRD", "T", "ST", "TT" or "TRT"
    - recurrent (bool, optional): flag to determine if the neurons should be recurrent (default: False)
    - fire (bool, optional): flag to determine if the neurons should fire spikes or not (default: True)
    - tau_mem (float, optional): time constant for the membrane potential (default: 1e-3)
    - tau_syn (float, optional): time constant for the synaptic potential (default: 1e-3)
    - time_step (float, optional): step size for updating the LIF model (default: 1e-3)
    - debug (bool, optional): flag to turn on/off debugging mode (default: False)
    """
	
    def __init__(self, input_size, hidden_size, device, spike_mode, recurrent=False, 
                 fire=True, tau_mem=1e-3, tau_syn=1e-3, time_step=1e-3, debug=False):

        super(ParaLIF, self).__init__(input_size, hidden_size, device, recurrent,
                 fire, tau_mem, tau_syn, time_step, debug)
        # Set the spiking function
        self.spike_mode = spike_mode
        if self.fire: self.spike_fn = SpikingFunction(self.device, self.spike_mode)
        else: self.spike_fn = None
        self.nb_spike_per_neuron_rec = torch.zeros(self.hidden_size, device=self.device)
        self.nb_steps = None
        

    def compute_params_fft(self):
        """
        Compute the FFT of the leakage parameters for parallel Leaky Integration

        Returns:
        fft_l_k: Product of FFT of parameters l and k
        """
        if self.nb_steps is None: return None

        l = torch.pow(self.alpha,torch.arange(self.nb_steps,device=self.device))
        k = torch.pow(self.beta,torch.arange(self.nb_steps,device=self.device))*(1-self.beta)
        fft_l = torch.fft.rfft(l, n=2*self.nb_steps).unsqueeze(1)
        fft_k = torch.fft.rfft(k, n=2*self.nb_steps).unsqueeze(1)
        return fft_l*fft_k


    def forward(self, inputs, parallel=True):
        """
        Perform forward pass of the network

        Parameters:
        - inputs (tensor): Input tensor with shape (batch_size, nb_steps, input_size)
        - parallel (bool, optional): If 'True' (default) the parallel forward is used and if 'False' the sequential forward is used

        Returns:
        - Return membrane potential tensor with shape (batch_size, nb_steps, hidden_size) if 'fire' is False
        - Return spiking tensor with shape (batch_size, nb_steps, hidden_size) if 'fire' is True
        - Return the tuple (spiking tensor, membrane potential tensor) if 'debug' is True and 'fire' is True
        """

        X = self.fc(inputs)
        if not parallel: return self.forward_sequential(X) # Run on sequential mode
        batch_size,nb_steps,_ = X.shape

        # Compute FFT params if nb_steps has changed
        if self.nb_steps!=nb_steps: 
            self.nb_steps = nb_steps
            self.fft_l_k = self.compute_params_fft()

        # Perform parallel leaky integration - Equation (15)
        fft_X = torch.fft.rfft(X, n=2*nb_steps, dim=1)
        mem_pot_hidden = torch.fft.irfft(fft_X*self.fft_l_k, n=2*nb_steps, dim=1)[:,:nb_steps:,]
        
        if self.recurrent:
            mem_pot_hidden_ = F.pad(mem_pot_hidden, (0,0,1,0), "constant", 0)[:,:-1]
            # Computing hidden state - Equation (22)
            hidden_state = self.spike_fn(mem_pot_hidden_) if self.fire else mem_pot_hidden_
            self.nb_spike_per_neuron_rec = torch.mean(torch.mean(hidden_state,dim=0),dim=0)
            # Perform parallel leaky integration for input and hidden state - Equation (23)
            fft_X_hidden_state = torch.fft.rfft(X + self.fc_recu(hidden_state), n=2*nb_steps, dim=1)
            mem_pot_temp = torch.fft.irfft(fft_X_hidden_state*self.fft_l_k, n=2*nb_steps, dim=1)[:,:nb_steps:,]
            mem_pot_final = mem_pot_hidden + mem_pot_temp
        else: mem_pot_final = mem_pot_hidden
            
        if self.fire:
        	# Perform firing - Equation (24)
            spikes = self.spike_fn(mem_pot_final)
            self.nb_spike_per_neuron = torch.mean(torch.mean(spikes,dim=0),dim=0)
            return (spikes, mem_pot_final) if self.debug else spikes
        return mem_pot_final
    
    # Sequential ParaLIF forward function
    def forward_sequential(self, X):
        batch_size,nb_steps,_ = X.shape
        syn_cur_hidden = torch.zeros_like(X[:,0]) # shape: [batch_size, hidden_size]
        mem_pot_hidden = torch.zeros_like(X[:,0]) # shape: [batch_size, hidden_size]
        mem_pot_hidden_prev = torch.zeros_like(X[:,0]) # shape: [batch_size, hidden_size]
        if self.recurrent:
            syn_cur_temp = torch.zeros_like(X[:,0]) # shape: [batch_size, hidden_size]
            mem_pot_temp = torch.zeros_like(X[:,0]) # shape: [batch_size, hidden_size]
            hidden_state = torch.zeros_like(X) # shape: [batch_size, nb_steps, hidden_size]
        mem_pot_final = torch.zeros_like(X) # shape: [batch_size, nb_steps, hidden_size]
        spikes = torch.zeros_like(X) # shape: [batch_size, nb_steps, hidden_size]
        
        for t in range(nb_steps):
            # Integrating input to synaptic current
            syn_cur_hidden = self.alpha*syn_cur_hidden + X[:,t]
            mem_pot_hidden_prev = mem_pot_hidden
            # Integrating synaptic current to membrane potential - Equation (7)
            mem_pot_hidden = self.beta*mem_pot_hidden_prev + (1-self.beta)*syn_cur_hidden
            if self.recurrent:
                # Computing hidden state - Equation (22)
                hidden_state[:,t] = self.spike_fn(torch.stack((mem_pot_hidden_prev,mem_pot_hidden), dim=1))[:,-1]  if self.fire else mem_pot_hidden
                # Integrating input and hidden state to recurrent synaptic current
                syn_cur_temp = self.alpha*syn_cur_temp + X[:,t] + self.fc_recu(hidden_state[:,t-1])
                # Integrating recurrent synaptic current to recurrent membrane potential
                mem_pot_temp = self.beta*mem_pot_temp + (1-self.beta)*syn_cur_temp
                mem_pot_final[:,t] = mem_pot_hidden + mem_pot_temp
            else: mem_pot_final[:,t] = mem_pot_hidden
            if self.fire: spikes[:,t] = self.spike_fn(mem_pot_final[:,[t-1,t]])[:,-1]
        
        if self.fire:
            self.nb_spike_per_neuron = torch.mean(torch.mean(spikes,dim=0),dim=0)
            if self.recurrent: self.nb_spike_per_neuron_rec = torch.mean(torch.mean(hidden_state,dim=0),dim=0)
            return (spikes, mem_pot_final) if self.debug else spikes
        return mem_pot_final
    
    def extra_repr(self):
        return f"spike_mode={self.spike_mode}, recurrent={self.recurrent}, fire={self.fire}, alpha={self.alpha:.2f}, beta={self.beta:.2f}"








class SpikingFunction(torch.nn.Module):
    """
    Perform spike generation. There is 4 main spiking methods:
        - GS : Gumbel Softmax
        - SB : Sigmoid Bernouilli
        - D: Delta
        - T: Threshold
    Some variants of these methods can be performed by first normalizing the input trough Sigmoid or Hyperbolic Tangeant
    """
    def __init__(self, device, spike_mode):
        super(SpikingFunction, self).__init__()
            
        if spike_mode in ["SB", "SD", "ST"]: self.normalise = torch.sigmoid
        elif spike_mode in ["TD", "TT"]: self.normalise = torch.tanh
        elif spike_mode in ["TRB", "TRD", "TRT"]: self.normalise = lambda inputs : F.relu(torch.tanh(inputs))
        else: self.normalise = lambda inputs : inputs
        
        if spike_mode in ["SB", "TRB"]: self.generate = StochasticStraightThrough.apply
        elif spike_mode =="GS": self.generate = GumbelSoftmax(device)
        elif spike_mode in ["D", "SD", "TD", "TRD"]: 
            self.generate = self.delta_fn
            self.threshold = torch.nn.Parameter(torch.tensor(0.01, device=device))
        elif spike_mode in ["T", "ST", "TT", "TRT"]: 
            self.generate = self.threshold_fn
            self.threshold = torch.nn.Parameter(self.normalise(torch.tensor(1., device=device)))
            
    def forward(self, inputs):
        inputs = self.normalise(inputs) 
        return self.generate(inputs)
    # Delta Spikes Generation - Equation (19)
    def delta_fn(self, inputs):
        inputs_previous = F.pad(inputs, (0,0,1,0), "constant", 0)[:,:-1]
        return SurrGradSpike.apply((inputs - inputs_previous) - self.threshold)
    # Threshold Spikes Generation
    def threshold_fn(self, inputs):
        return SurrGradSpike.apply(inputs - self.threshold)




# Sigmoid Bernoulli Spikes Generation
class StochasticStraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.bernoulli(input) # Equation (17)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input*input # Equation (18)


# Gumbel Softmax Spikes Generation
class GumbelSoftmax(torch.nn.Module):
    def __init__(self, device, hard=True, tau=1.0):
        super().__init__()
        self.hard = hard
        self.tau = tau
        self.uniform = torch.distributions.Uniform(torch.tensor(0.0).to(device),torch.tensor(1.0).to(device))
        self.softmax = torch.nn.Softmax(dim=0)
  
    def forward(self, logits):
        # Sample uniform noise
        unif = self.uniform.sample(logits.shape + (2,))
        # Compute Gumbel noise from the uniform noise
        gumbels = -torch.log(-torch.log(unif))
        # Apply softmax function to the logits and Gumbel noise
        y_soft = self.softmax(torch.stack([(logits + gumbels[...,0]) / self.tau,
                                                     (-logits + gumbels[...,1]) / self.tau]))[0]
        if self.hard:
            # Use straight-through estimator
            y_hard = torch.where(y_soft > 0.5, 1.0, 0.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Use reparameterization trick
            ret = y_soft
        return ret




if __name__=='__main__':
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    import matplotlib.pyplot as plt
    input_signal = torch.rand((2,20,1))    
   
    paralif = ParaLIF(1, 5, device=None, spike_mode="T", recurrent=True,
                 fire=True, tau_mem=1e-3, tau_syn=1e-3, time_step=1e-3, debug=True)
    o_par,mem_par = paralif(input_signal)
    o_seq,mem_seq = paralif(input_signal, parallel=False)
    print(o_seq.shape)
    print("diff spikes =", torch.mean((o_par-o_seq)**2).item())
    print("diff mem =", torch.mean((mem_par-mem_seq)**2).item())
    
    plt.plot(input_signal[0,:,0], label='Input')
    plt.plot(mem_par[0,:,0].detach(), label='Mem. potential')
    plt.plot(o_par[0,:,0].detach(), linestyle='--', label='Spikes')
    plt.legend()
    plt.show()
    

