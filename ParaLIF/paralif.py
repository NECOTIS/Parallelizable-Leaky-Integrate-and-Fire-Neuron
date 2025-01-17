# -*- coding: utf-8 -*-
"""
Created on August 2024

@author: Arnaud Yarga
"""

import torch
from ParaLIF.base import Base, SurrogateFunction
import torch.nn.functional as F


class ParaLIF(Base):
    """
    Class for implementing a Parallelizable Leaky Integrate-and-Fire (ParaLIF) neuron model

    Parameters:
    - n_neuron (int): number of neurons in the layer.
    - spike_mode (str): "GS", "SB", "TRB", "D", "SD", "TD", "TRD", "T", "ST", "TT" or "TRT".
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
    - device (torch.device): Device to use for tensor computations, such as 'cpu' or 'cuda'.
    - surrogate_mode (str): Type of surrogate gradient function ('fastsig' or 'atan') (default: 'fastsig').
    - surrogate_scale (float): Scaling factor for the surrogate gradient function. Default depends on mode.
    - debug (bool, optional): flag to turn on/off debugging mode (default: False).
    """
	
    def __init__(self, n_neuron, spike_mode, recurrent=False, fire=True, recurrent_fire=True, spk_threshold=1., 
                 learn_threshold=False, tau_mem=1e-3, tau_syn=1e-3, time_step=1e-3, learn_tau=False,
                 device=None, surrogate_mode="fastsig", surrogate_scale=None, debug=False):
        """
        Initializes the ParaLIF model with the specified parameters.
        """

        # Call the constructor of the Base class
        super(ParaLIF, self).__init__(n_neuron, recurrent, fire, recurrent_fire, spk_threshold, learn_threshold, 
                                  tau_mem, tau_syn, time_step, learn_tau, device, debug)
        
        # Set the spiking function based on the spike_mode and other parameters
        self.spike_mode = spike_mode
        self.spike_fn = SpikingFunction(self.spike_mode, surrogate_mode, surrogate_scale, self.device) if (self.fire or (self.recurrent and self.recurrent_fire)) else None

        # Initialize the number of spikes per neuron for recording
        self.register_buffer('nb_spike_per_neuron_rec', torch.zeros(self.n_neuron, device=self.device))
        self.nb_steps = None
        

    def compute_params_fft(self, nb_steps):
        """
        Compute the FFT of the leakage parameters for parallel Leaky Integration.

        Returns:
        - fft_l_k: Product of FFT of parameters l and k
        """
        if nb_steps is None: return None
        self.device = self.alpha.device
        # Compute FFT for alpha parameter
        if self.alpha.dim() == 0:
            l = torch.pow(self.alpha, torch.arange(nb_steps, device=self.device))
            fft_l = torch.fft.rfft(l, n=2*nb_steps).unsqueeze(1)
        elif self.alpha.dim() > 0:
            l = torch.pow(self.alpha, torch.arange(nb_steps, device=self.device).expand(self.n_neuron, nb_steps).T)
            fft_l = torch.fft.rfft(l, n=2*nb_steps, dim=0)

        # Compute FFT for beta parameter
        if self.beta.dim() == 0:
            k = torch.pow(self.beta, torch.arange(nb_steps, device=self.device)) * (1-self.beta)
            fft_k = torch.fft.rfft(k, n=2*nb_steps).unsqueeze(1)
        elif self.beta.dim() > 0:
            k = torch.pow(self.beta, torch.arange(nb_steps, device=self.device).expand(self.n_neuron, nb_steps).T) * (1-self.beta)
            fft_k = torch.fft.rfft(k, n=2*nb_steps, dim=0)
        
        # Return the product of FFTs of l and k
        return fft_l * fft_k


    def forward(self, X, parallel=True):
        """
        Perform forward pass of the network.

        Parameters:
        - X (tensor): Input tensor with shape (batch_size, nb_steps, input_size)
        - parallel (bool, optional): If 'True' (default) the parallel forward is used, 
          and if 'False' the sequential forward is used

        Returns:
        - Return membrane potential tensor with shape (batch_size, nb_steps, n_neuron) if 'fire' is False
        - Return spiking tensor with shape (batch_size, nb_steps, n_neuron) if 'fire' is True
        - Return the tuple (spiking tensor, membrane potential tensor) if 'debug' is True and 'fire' is True
        """

        # If parallel is False, use the sequential forward method
        if not parallel: return self.forward_sequential(X) 
        
        batch_size, nb_steps, _ = X.shape

        # Compute FFT parameters if the number of steps has changed or if tau params are learnable
        if self.nb_steps != nb_steps or self.learn_tau: 
            self.fft_l_k = self.compute_params_fft(nb_steps)
            self.nb_steps = nb_steps

        # Perform parallel leaky integration - Equation (15)
        fft_X = torch.fft.rfft(X, n=2*nb_steps, dim=1)
        mem_pot_hidden = torch.fft.irfft(fft_X * self.fft_l_k, n=2*nb_steps, dim=1)[:, :nb_steps, :]
        
        # Handle recurrent connections if needed
        if self.recurrent:
            mem_pot_hidden_ = F.pad(mem_pot_hidden, (0, 0, 1, 0), "constant", 0)[:, :-1]
            # Compute hidden state - Equation (22)
            hidden_state = self.spike_fn(mem_pot_hidden_, self.spk_threshold) if self.recurrent_fire else F.relu(mem_pot_hidden_)
            if self.recurrent_fire: self.nb_spike_per_neuron_rec = torch.mean(torch.mean(hidden_state, dim=0), dim=0)
            # Perform parallel leaky integration for input and hidden state - Equation (23)
            fft_X_hidden_state = torch.fft.rfft(X + self.fc_recu(hidden_state), n=2*nb_steps, dim=1)
            mem_pot_temp = torch.fft.irfft(fft_X_hidden_state * self.fft_l_k, n=2*nb_steps, dim=1)[:, :nb_steps, :]
            mem_pot_final = mem_pot_hidden + mem_pot_temp
        else:
            mem_pot_final = mem_pot_hidden
            
        # Handle firing if enabled
        if self.fire:
            # Perform firing - Equation (24)
            spikes = self.spike_fn(mem_pot_final, self.spk_threshold)
            self.nb_spike_per_neuron = torch.mean(torch.mean(spikes, dim=0), dim=0)
            return (spikes, mem_pot_final) if self.debug else spikes
        
        return mem_pot_final
    
    def forward_sequential(self, X):
        """
        Sequential implementation of the ParaLIF forward function.
        """
        batch_size, nb_steps, _ = X.shape
        syn_cur_hidden = torch.zeros_like(X[:, 0]) # Initialize synaptic current for hidden layer
        mem_pot_hidden = torch.zeros_like(X[:, 0]) # Initialize membrane potential for hidden layer
        mem_pot_hidden_prev = torch.zeros_like(X[:, 0]) # Initialize previous membrane potential

        if self.recurrent:
            syn_cur_temp = torch.zeros_like(X[:, 0]) # Initialize synaptic current for recurrent connections
            mem_pot_temp = torch.zeros_like(X[:, 0]) # Initialize membrane potential for recurrent connections
            hidden_state = torch.zeros_like(X[:, 0]) # Initialize hidden state tensor

        #mem_pot_final = torch.zeros_like(X) # Initialize final membrane potential
        mem_pot_final = []
        #spikes = torch.zeros_like(X) # Initialize spikes tensor
        spikes_t = torch.zeros_like(X[:, 0])
        spikes = []
        
        for t in range(nb_steps):
            # Integrate input to synaptic current
            syn_cur_hidden = self.alpha * syn_cur_hidden + X[:, t]
            mem_pot_hidden_prev = mem_pot_hidden

            # Integrate synaptic current to membrane potential - Equation (7)
            mem_pot_hidden = self.beta * mem_pot_hidden_prev + (1-self.beta) * syn_cur_hidden

            if self.recurrent:
                # Integrate input and hidden state to recurrent synaptic current
                syn_cur_temp = self.alpha * syn_cur_temp + X[:, t] + self.fc_recu(hidden_state)
                
                # Integrate recurrent synaptic current to recurrent membrane potential
                mem_pot_temp = self.beta * mem_pot_temp + (1-self.beta) * syn_cur_temp
                #mem_pot_final[:, t] = mem_pot_hidden + mem_pot_temp
                mem_pot_final.append(mem_pot_hidden + mem_pot_temp)
                
                # Compute hidden state - Equation (22)
                hidden_state = self.spike_fn(torch.stack((mem_pot_hidden_prev, mem_pot_hidden), dim=1), self.spk_threshold)[:, -1] if self.recurrent_fire else F.relu(mem_pot_hidden)
            else:
                #mem_pot_final[:, t] = mem_pot_hidden
                mem_pot_final.append(mem_pot_hidden)
            
            # Handle firing
            if self.fire: 
                #spikes[:,t] = self.spike_fn(mem_pot_final[:,[t-1,t]], self.spk_threshold)[:,-1]
                mem_pot_final_prev = mem_pot_final[-2] if len(mem_pot_final)>1 else torch.zeros_like(mem_pot_final[-1])
                spikes_t = self.spike_fn(torch.stack((mem_pot_final_prev, mem_pot_final[-1]), dim=1), self.spk_threshold)[:,-1]
            spikes.append(spikes_t)
        
        mem_pot_final = torch.stack(mem_pot_final, dim=1)
        spikes = torch.stack(spikes, dim=1)
        # Save average spikes and return ouputs tensors
        if self.fire:
            self.nb_spike_per_neuron = torch.mean(torch.mean(spikes,dim=0),dim=0)
            if self.recurrent_fire: self.nb_spike_per_neuron_rec = torch.mean(torch.mean(spikes, dim=0), dim=0)
            return (spikes, mem_pot_final) if self.debug else spikes
        
        return mem_pot_final
    
    def extra_repr(self):
        """
        Provides additional representation for the module, useful for printing.
        """
        return f"spike_mode={self.spike_mode}, " + super().extra_repr()





def identity_fn(x):
    return x

def tanh_relu_fn(x):
    return F.relu(torch.tanh(x))

class SpikingFunction(torch.nn.Module):
    """
    Perform spike generation using various methods. The main spiking methods include:
        - GS : Gumbel Softmax
        - SB : Sigmoid Bernoulli
        - D : Delta
        - T : Threshold
    Variants of these methods can be applied by first normalizing the input using Sigmoid or Hyperbolic Tangent functions.
    """
    
    def __init__(self, spike_mode, surrogate_mode, surrogate_scale, device):
        """
        Initializes the SpikingFunction with the specified spike generation mode and other parameters.
        
        Parameters:
        - spike_mode (str): The spiking method to use. Must be one of the available modes in `spike_mode_list`.
        - surrogate_mode (str): The surrogate gradient mode used in some spiking methods.
        - surrogate_scale (float): Scale parameter for the surrogate gradient.
        - device (torch.device): Device to use for computations, such as 'cpu' or 'cuda'.
        """
        super(SpikingFunction, self).__init__()
        self.spike_mode_list = ["GS", "SB", "TRB", "D", "SD", "TD", "TRD", "T", "ST", "TT", "TRT"]
        assert spike_mode in self.spike_mode_list, f"'{spike_mode}' spike mode is not available. The available options are {self.spike_mode_list}"
        self.spike_mode = spike_mode
        
        # Input normalization function based on the selected spiking mode
        if spike_mode in ["SB", "SD", "ST"]:
            self.normalize = torch.sigmoid  # Sigmoid normalization for certain spike modes
        elif spike_mode in ["TD", "TT"]:
            self.normalize = torch.tanh  # Tanh normalization for certain spike modes
        elif spike_mode in ["TRB", "TRD", "TRT"]:
            self.normalize = tanh_relu_fn  # Combination of Tanh and ReLU for specific modes
        else:
            self.normalize = identity_fn  # No normalization for other modes

        # Spike generation function selection based on the spike_mode
        if spike_mode in ["SB", "TRB"]:
            self.generate = self.bernoulli_fn  # Use Bernoulli spike generation
        elif spike_mode in ["D", "SD", "TD", "TRD"]:
            self.surrogate_fn = SurrogateFunction(mode=surrogate_mode, scale=surrogate_scale)
            self.generate = self.delta_fn  # Use Delta spike generation
        elif spike_mode in ["T", "ST", "TT", "TRT"]:
            self.surrogate_fn = SurrogateFunction(mode=surrogate_mode, scale=surrogate_scale)
            self.generate = self.threshold_fn  # Use Threshold spike generation
        elif spike_mode == "GS":
            self.gs = GumbelSoftmax(device)
            self.generate = self.gumbel_softmax_fn  # Use Gumbel Softmax spike generation
            
    def forward(self, inputs, spk_threshold):
        """
        Perform spike generation on the input tensor based on the selected spiking method.
        
        Parameters:
        - inputs (tensor): The input tensor to be processed.
        - spk_threshold (float): The threshold value for spike generation.
        
        Returns:
        - tensor: The output tensor representing generated spikes.
        """
        inputs = self.normalize(inputs)  # Normalize the input
        return self.generate(inputs, spk_threshold)  # Generate spikes based on the selected method
     
    # Delta Spikes Generation - Uses surrogate gradient
    def delta_fn(self, inputs, spk_threshold):
        """
        Generate spikes using the Delta method with a surrogate gradient.
        
        Parameters:
        - inputs (tensor): The input tensor with shape (batch_size, time_steps, input_size).
        - spk_threshold (float): The threshold for spike generation.
        
        Returns:
        - tensor: The generated spike tensor.
        """
        inputs_previous = F.pad(inputs, (0, 0, 1, 0), "constant", 0)[:, :-1]  # Previous time step inputs
        return self.surrogate_fn((inputs - inputs_previous) - spk_threshold)  # Spike based on change from previous step
    
    # Threshold Spikes Generation - Simple threshold comparison
    def threshold_fn(self, inputs, spk_threshold):
        """
        Generate spikes using a simple threshold comparison.
        
        Parameters:
        - inputs (tensor): The input tensor.
        - spk_threshold (float): The threshold for spike generation.
        
        Returns:
        - tensor: The generated spike tensor.
        """
        return self.surrogate_fn(inputs - spk_threshold)
    
    # Sigmoid Bernoulli Spikes Generation - Stochastic spike generation
    def bernoulli_fn(self, inputs, _):
        """
        Generate spikes stochastically using the Sigmoid Bernoulli method.
        
        Parameters:
        - inputs (tensor): The input tensor after normalization.
        
        Returns:
        - tensor: The stochastically generated spike tensor.
        """
        return StochasticStraightThrough.apply(inputs)
    
    # Gumbel Softmax Spikes Generation - Sample spikes using Gumbel-Softmax trick
    def gumbel_softmax_fn(self, inputs, _):
        """
        Generate spikes using the Gumbel Softmax method.
        
        Parameters:
        - inputs (tensor): The input tensor after normalization.
        
        Returns:
        - tensor: The Gumbel Softmax generated spike tensor.
        """
        return self.gs(inputs)
    
    def extra_repr(self):
        """
        Provides additional representation for the module, useful for printing.
        """
        desc = {"D": "Delta", "T": "Threshold", "B": "Bernoulli", "S": "Gumbel Softmax"}
        spike_mode_text = ""
        if self.spike_mode in ["SB", "SD", "ST"]:
            spike_mode_text = "Sigmoid "
        elif self.spike_mode in ["TD", "TT"]:
            spike_mode_text = "Tanh "
        elif self.spike_mode in ["TRB", "TRD", "TRT"]:
            spike_mode_text = "Tanh-ReLU "
        spike_mode_text += desc[self.spike_mode[-1]]
        return f"spike_mode: {spike_mode_text}"


# Sigmoid Bernoulli Spikes Generation
class StochasticStraightThrough(torch.autograd.Function):
    """
    Custom autograd function for Sigmoid Bernoulli spiking with straight-through estimation.
    """

    @staticmethod
    def forward(ctx, input):
        """
        Forward pass: Perform stochastic sampling using the Bernoulli distribution.
        
        Parameters:
        - input (tensor): The input tensor after Sigmoid normalization.
        
        Returns:
        - tensor: The output tensor with sampled spikes.
        """
        ctx.save_for_backward(input)
        out = torch.bernoulli(input)  # Sample spikes using Bernoulli distribution
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Compute gradients using the straight-through estimator.
        
        Parameters:
        - grad_output (tensor): The gradient of the loss with respect to the output.
        
        Returns:
        - tensor: The gradient of the loss with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input * input  # Gradient computation with straight-through estimator


# Gumbel Softmax Spikes Generation
class GumbelSoftmax(torch.nn.Module):
    """
    Generate spikes using the Gumbel-Softmax method, which allows for differentiable sampling.
    """

    def __init__(self, device=None, hard=True, tau=1.0):
        """
        Initialize the GumbelSoftmax module.
        
        Parameters:
        - device (torch.device): Device to use for tensor operations.
        - hard (bool): Whether to use hard sampling with straight-through estimation.
        - tau (float): Temperature parameter for the Gumbel-Softmax distribution.
        """
        super().__init__()
        self.device = device
        self.hard = hard  # Use hard sampling or soft sampling
        self.tau = tau  # Temperature parameter for Gumbel-Softmax
        self.uniform = torch.distributions.Uniform(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, logits):
        """
        Perform the forward pass to sample spikes using the Gumbel-Softmax trick.
        
        Parameters:
        - logits (tensor): The input logits for Gumbel-Softmax sampling.
        
        Returns:
        - tensor: The sampled spikes tensor.
        """
        if logits.device != self.device:
            self.device = logits.device
            self.uniform = torch.distributions.Uniform(torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device))
        # Sample uniform noise
        unif = self.uniform.sample(logits.shape + (2,))
        # Compute Gumbel noise from the uniform noise
        gumbels = -torch.log(-torch.log(unif))
        # Apply softmax function to the logits and Gumbel noise
        y_soft = self.softmax(torch.stack([(logits + gumbels[..., 0]) / self.tau, (-logits + gumbels[..., 1]) / self.tau]))[0]
        if self.hard:
            # Use straight-through estimator for hard sampling
            y_hard = torch.where(y_soft > 0.5, 1.0, 0.0)
            ret = y_hard - y_soft.detach() + y_soft  # Detach y_soft to stop gradient flow
        else:
            # Use reparameterization trick for soft sampling
            ret = y_soft
        return ret





if __name__=='__main__':
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    import matplotlib.pyplot as plt
    
    # Define the input signal tensor
    input_signal = torch.rand((2, 100, 5))
    input_signal[:, 3] = 1.0  # Set a specific time step to a higher value to observe spike generation
    
    # Instantiate the ParaLIF model
    paralif = ParaLIF(
        n_neuron=5,
        spike_mode="D",
        recurrent=False,
        fire=True,
        recurrent_fire=True,
        spk_threshold=[1.0, 0.3, 1.0, 1.0, 0.5],
        learn_threshold=False,
        tau_mem=[2e-3, 2e-3, 2e-3, 1e-3, 5e-3],
        tau_syn=[1e-3, 1e-3, 2e-3, 1e-3, 5e-3],
        time_step=1e-3,
        learn_tau=True,
        device=None,
        surrogate_mode="atan",
        surrogate_scale=None,
        debug=True
    )
    
    # Perform forward pass with parallel computation
    spikes_par, mem_pot_par = paralif(input_signal)
    
    # Perform forward pass without parallel computation
    spikes_seq, mem_pot_seq = paralif(input_signal, parallel=False)
    
    # Print shapes of output tensors
    print("Spikes shape (parallel):", spikes_par.shape)
    print("Spikes shape (sequential):", spikes_seq.shape)
    print("Difference in spikes:", torch.mean((spikes_par - spikes_seq) ** 2).item())
    print("Difference in membrane potential:", torch.mean((mem_pot_par - mem_pot_seq) ** 2).item())
    
    # Plot the results for each neuron
    for i in range(5):
        plt.figure(figsize=(10, 6))
        plt.plot(input_signal[0, :, i].detach(), label='Input', color='blue')
        plt.plot(mem_pot_par[0, :, i].detach(), label='Membrane Potential (parallel)', color='orange')
        plt.plot(spikes_par[0, :, i].detach(), linestyle='--', label='Spikes (parallel)', color='red')
        plt.title(f'ParaLIF - Neuron {i}')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
        
        
