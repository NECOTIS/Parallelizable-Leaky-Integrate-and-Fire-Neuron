# -*- coding: utf-8 -*-
"""
Created on December 2023

@author: Arnaud Yarga
"""

import torch
from neurons import Base, SurrGradSpike


class LIF(Base):
    """
    Class for implementing a Leaky Integrate and Fire (LIF) neuron model

    Parameters:
    - input_size (int): The number of expected features in the input
    - hidden_size (int): The number of neurons on the layer
    - device (torch.device): device to use for tensor computations, such as 'cpu' or 'cuda'
    - recurrent (bool, optional): flag to determine if the neurons should be recurrent (default: False)
    - fire (bool, optional): flag to determine if the neurons should fire spikes or not (default: True)
    - tau_mem (float, optional): time constant for the membrane potential (default: 1e-3)
    - tau_syn (float, optional): time constant for the synaptic potential (default: 1e-3)
    - time_step (float, optional): step size for updating the LIF model (default: 1e-3)
    - debug (bool, optional): flag to turn on/off debugging mode (default: False)
    """
    def __init__(self, input_size, hidden_size, device, recurrent=False,
                 fire=True, tau_mem=1e-3, tau_syn=1e-3, time_step=1e-3, learn_th=False,
                 debug=False):
        super(LIF, self).__init__(input_size, hidden_size, device, recurrent,
                 fire, tau_mem, tau_syn, time_step, debug)
        
        self.learn_th = learn_th
        if self.learn_th:
            self.v_th = torch.nn.Parameter(torch.tensor(1.0, device=device))
            
        # Set the spiking function
        if not self.fire: self.spike_fn = None
        else: self.spike_fn = SurrGradSpike.apply


    def forward(self, inputs):
        """
        Perform forward pass of the network

        Parameters:
        - inputs (tensor): Input tensor with shape (batch_size, nb_steps, input_size)

        Returns:
        - Return membrane potential tensor with shape (batch_size, nb_steps, hidden_size) if 'fire' is False
        - Return spiking tensor with shape (batch_size, nb_steps, hidden_size) if 'fire' is True
        - Return the tuple (spiking tensor, membrane potential tensor) if 'debug' is True
        """
        X = self.fc(inputs)
        batch_size,nb_steps,_ = X.shape
        syn_cur = torch.zeros_like(X[:,0]) # shape: [batch_size, hidden_size]
        mem_pot = torch.zeros_like(X) # shape: [batch_size, nb_steps, hidden_size]
        spikes = torch.zeros_like(X) # shape: [batch_size, nb_steps, hidden_size]
        
        # Iterate over each time step
        for t in range(nb_steps):
            # Integrating input to synaptic current - Equation (5)
            syn_cur = self.alpha*syn_cur + X[:,t] 
            # Adding recurrent input to synaptic current - Equation (20)
            if self.recurrent: syn_cur += self.fc_recu(spikes[:,t-1] if self.fire else mem_pot[:,t-1])
            # Integrating synaptic current to membrane potential - Equation (6)
            mem_pot[:,t] = self.beta*mem_pot[:,t-1] + (1-self.beta)*syn_cur 
            if self.fire:
                # Spikes generation - Equation (3)
                spikes[:,t] = self.spike_fn(mem_pot[:,t]-self.v_th)
                # Membrane potential reseting - Equation (6)
                mem_pot[:,t] = mem_pot[:,t] * (1-spikes[:,t].detach())
        
        if self.fire:
            self.nb_spike_per_neuron = torch.mean(torch.mean(spikes,dim=0),dim=0)
            return (spikes, mem_pot) if self.debug else spikes
        return mem_pot
    
    def extra_repr(self):
        return f"recurrent={self.recurrent}, fire={self.fire}, alpha={self.alpha:.2f}, beta={self.beta:.2f}"


if __name__=='__main__':
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    import matplotlib.pyplot as plt
    input_signal = torch.rand((2,20,1))
   
    lif = LIF(1, 5, device=None, recurrent=True, fire=True, tau_mem=1e-3, tau_syn=1e-3, time_step=1e-3, debug=True)
    o, mem = lif(input_signal)
    print(o.shape)
    
    plt.plot(input_signal[0,:,0], label='Input')
    plt.plot(mem[0,:,0].detach(), label='Mem. potential')
    plt.plot(o[0,:,0].detach(), linestyle='--', label='Spikes')
    plt.legend()
    plt.show()

