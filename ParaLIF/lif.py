# -*- coding: utf-8 -*-
"""
Created on August 2024

@author: Arnaud Yarga
"""

import torch
from ParaLIF.base import Base, SurrogateFunction
import torch.nn.functional as F

class LIF(Base):
    """
    Leaky Integrate and Fire (LIF) neuron model implementation.

    Parameters:
    - n_neuron (int): Number of neurons in the layer.
    - recurrent (bool, optional): flag to determine if the neurons should be recurrent (default: False).
    - fire (bool, optional): flag to determine if the neurons should fire spikes or not (default: True).
    - recurrent_fire (bool): Flag to determine if recurrent neurons should fire (default: True).
    - spk_threshold (float or list): Spiking threshold(s) for the neurons. Can 
                    be a single value or a list with length equal to n_neuron (default: 1.).
    - learn_threshold (bool): Whether the spiking threshold is learnable (default: False).
    - tau_mem (float or list): Membrane time constant. Can be a single value or a list of values (default: 1e-3).
    - tau_syn (float or list): Synaptic time constant. Can be a single value or a list of values (default: 1e-3).
    - time_step (float): Step size for updating the neuron model (default: 1e-3).
    - learn_tau (bool): Whether the time constants are learnable (default: False).
    - refractory (float): Refractory period after spiking (default: None).
    - device (torch.device): Device to use for tensor computations, such as 'cpu' or 'cuda'.
    - surrogate_mode (str): Type of surrogate gradient function ('fastsig' or 'atan') (default: 'fastsig').
    - surrogate_scale (float): Scaling factor for the surrogate gradient function. Default depends on mode.
    - debug (bool, optional): flag to turn on/off debugging mode (default: False).
    """

    def __init__(self, n_neuron, recurrent=False, fire=True, recurrent_fire=True, spk_threshold=1.0, 
                 learn_threshold=False, tau_mem=1e-3, tau_syn=1e-3, time_step=1e-3, learn_tau=False, refractory=None,
                 device=None, surrogate_mode="fastsig", surrogate_scale=None, debug=False):
        """
        Initializes the LIF neuron model with the specified parameters.
        """
        super(LIF, self).__init__(n_neuron, recurrent, fire, recurrent_fire, spk_threshold, learn_threshold, 
                                  tau_mem, tau_syn, time_step, learn_tau, device, debug)
        
        self.refractory = (refractory/time_step) if refractory else refractory
        # Set the spiking function if spike generation is enabled (`fire` is True)
        self.spike_fn = SurrogateFunction(mode=surrogate_mode, scale=surrogate_scale) if self.fire else None


    def forward(self, X):
        """
        Perform the forward pass of the LIF neuron model.

        Parameters:
        - X (tensor): Input tensor with shape (batch_size, nb_steps, input_size).

        Returns:
        - If 'fire' is False: Returns the membrane potential tensor with shape (batch_size, nb_steps, n_neuron).
        - If 'fire' is True: Returns the spiking tensor with shape (batch_size, nb_steps, n_neuron).
        - If 'debug' is True: Returns a tuple (spiking tensor, membrane potential tensor) for additional insight.
        """
        # Initialize batch size and number of time steps from the input tensor shape
        batch_size, nb_steps, _ = X.shape
        
        # Initialize synaptic current, membrane potential, and spike tensors
        syn_cur = torch.zeros_like(X[:, 0])  # shape: [batch_size, n_neuron]
        mem_pot_t = torch.zeros_like(X[:, 0])  # shape: [batch_size, n_neuron]
        spikes_t = torch.zeros_like(X[:, 0])  # shape: [batch_size, n_neuron]
        mem_pot = []
        spikes = []
        if self.refractory: refractory_time = torch.zeros_like(X[:, 0])  # shape: [batch_size, n_neuron]
        
        # Iterate over each time step to update synaptic currents and membrane potentials
        for t in range(nb_steps):
            if self.refractory: refractory_time -= 1
            # Update synaptic current with input contribution - Integrating input to synaptic current (Equation 5)
            syn_cur = self.alpha * syn_cur + X[:, t]
            
            # Add recurrent input to synaptic current if recurrent connections are enabled (Equation 20)
            if self.recurrent:
                # Choose the source of recurrent input based on `recurrent_fire` flag
                prev_output = spikes_t if self.recurrent_fire else F.relu(mem_pot_t)
                syn_cur += self.fc_recu(prev_output)
            
            # Update membrane potential with synaptic current contribution - Integrating synaptic current (Equation 6)
            mem_pot_t = self.beta * mem_pot_t + (1 - self.beta) * syn_cur * (torch.where(refractory_time > 0, 0.0, 1.0) if self.refractory else 1.)
            
            if self.fire:
                # Generate spikes based on the membrane potential exceeding the threshold (Equation 3)
                spikes_t = self.spike_fn(mem_pot_t - self.spk_threshold)
                if self.refractory: refractory_time[spikes_t==1] = self.refractory
                # Reset membrane potential where spikes occurred (Equation 6)
                mem_pot_t = mem_pot_t * (1 - spikes_t.detach())
            
            mem_pot.append(mem_pot_t)
            spikes.append(spikes_t)
            
        mem_pot = torch.stack(mem_pot, dim=1)
        spikes = torch.stack(spikes, dim=1)
        if self.fire:
            # Calculate the average number of spikes per neuron for potential analysis
            self.nb_spike_per_neuron = torch.mean(torch.mean(spikes, dim=0), dim=0)
            return (spikes, mem_pot) if self.debug else spikes  # Return spikes or (spikes, mem_pot) based on `debug`
        
        # Return membrane potential if no spiking is performed
        return mem_pot



if __name__=='__main__':
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    
    import matplotlib.pyplot as plt
    
    # Define the input signal tensor
    input_signal = torch.zeros((2, 20, 5))
    input_signal[:, 3] = 2.0  # Set a specific time step to a higher value to observe spike generation
    
    # Instantiate the LIF model
    lif = LIF(
        n_neuron=5,
        recurrent=True,
        fire=True,
        recurrent_fire=True,
        spk_threshold=[1.0, 0.3, 1.0, 1.0, 0.5],
        learn_threshold=True,
        tau_mem=[2e-3, 2e-3, 2e-3, 2e-3, 5e-3],
        tau_syn=[1e-3, 1e-3, 1e-3, 1e-3, 2e-3],
        learn_tau=False,
        refractory=2e-3,
        device=None,
        surrogate_mode="atan",
        surrogate_scale=50.0,
        debug=True
    )
    
    # Perform forward pass
    spikes, mem_pot = lif(input_signal)
    
    # Print the shapes of the output tensors
    print("Spikes shape:", spikes.shape)
    print("Membrane potential shape:", mem_pot.shape)
    
    # Plot the results for each neuron
    for i in range(5):
        plt.figure(figsize=(10, 6))
        plt.plot(input_signal[0, :, i].detach(), label='Input', color='blue')
        plt.plot(mem_pot[0, :, i].detach(), label='Membrane Potential', color='orange')
        #plt.plot(spikes[0, :, i].detach(), linestyle='--', label='Spikes', color='red')
        plt.eventplot(torch.where(spikes[0, :, i].detach()==1), linestyle='--', label='Spikes', color='red', lineoffsets=0.5)
        plt.title(f'LIF - Neuron {i}')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

















