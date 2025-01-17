# -*- coding: utf-8 -*-
"""
Created on December 2025

@author: Arnaud Yarga

We compare LIF and ParaLIF by following the methodology of
[E. M. Izhikevich, “Which model to use for cortical spiking neurons?”
IEEE transactions on neural networks, vol. 15, no. 5, pp. 1063–1070, 2004],
which involves assessing how many of the 20 most important features of biological
neurons each model can replicate
"""

import torch
import numpy as np
from ParaLIF import LIF,ParaLIF
import matplotlib.pyplot as plt



def run(input_samples, inp_=1., tau_mem=1e-3, spk_threshold=1., spike_mode="LIF", prop=None):
    if spike_mode=="LIF":
        neuron = LIF(1, recurrent=False, fire=True, recurrent_fire=True, spk_threshold=spk_threshold, 
                 learn_threshold=False, tau_mem=tau_mem, tau_syn=None, debug=True)
    else:
        neuron = ParaLIF(1, spike_mode, recurrent=False, fire=True, recurrent_fire=True, spk_threshold=spk_threshold, 
                     learn_threshold=False, tau_mem=tau_mem, tau_syn=None, debug=True)
        spike_mode = "ParaLIF-"+spike_mode
    
    input_samples*=inp_
    nb_steps = input_samples.shape[1]
    spikes,mem_pot = neuron(input_samples)
    return input_samples, spikes, mem_pot, nb_steps, spike_mode, prop
    


all_outputs = []
outputs = []
nb_steps = 100
input_samples = torch.zeros(1,nb_steps,1)
input_samples[:,10:]= 1
for (inp_, tau_mem, spk_threshold, spike_mode, prop) in [(2., 0.0107, 1.0, "LIF", "A"), (2., 0.0107, 1.0, "T", "A"), (2, 0.0187, 0.1, "D", "B"), (1.5, 0.0074, 0.1, "D", "D")]:
    o = run(input_samples.clone(), inp_=inp_, tau_mem=tau_mem, spk_threshold=spk_threshold, spike_mode=spike_mode, prop=prop)
    outputs.append(o)
all_outputs.append(outputs)

outputs = []
input_samples = torch.zeros(1,nb_steps,1)
input_samples[0,10:-10,0]= torch.linspace(0,1,nb_steps-20)
for (inp_, tau_mem, spk_threshold, spike_mode, prop) in [(5.5, 0.0115, 1.0, "LIF", "G"),(5.5, 0.0011, 0.9, "T", "H"),(8.5, 0.0025, 0.1, "D", "H")]:
    o = run(input_samples.clone(), inp_=inp_, tau_mem=tau_mem, spk_threshold=spk_threshold, spike_mode=spike_mode, prop=prop)
    outputs.append(o)
all_outputs.append(outputs)

outputs = []
input_samples = torch.zeros(1,nb_steps,1)
input_samples[0,10:12,0]= 1
input_samples[0,20:22,0]= 1
input_samples[0,70:72,0]= 1
input_samples[0,90:92,0]= 1
for (inp_, tau_mem, spk_threshold, spike_mode, prop) in [(3.9, 0.009, 1.0, "LIF", "L"), (3.9, 0.009, 1.0, "T", "L")]:
    o = run(input_samples.clone(), inp_=inp_, tau_mem=tau_mem, spk_threshold=spk_threshold, spike_mode=spike_mode, prop=prop)
    outputs.append(o)
all_outputs.append(outputs)


outputs = []
input_samples = torch.zeros(1,nb_steps,1)
input_samples[0,10:12,0]= 1
for (inp_, tau_mem, spk_threshold, spike_mode, prop) in [(-1.0, 0.001, 0.3, "D", "M"),(-22.3, 0.001, 0.3, "D", "N")]:
    o = run(input_samples.clone(), inp_=inp_, tau_mem=tau_mem, spk_threshold=spk_threshold, spike_mode=spike_mode, prop=prop)
    outputs.append(o)
all_outputs.append(outputs)


outputs = []
input_samples = torch.zeros(1,nb_steps,1)
input_samples[0,10:12,0]= 1
input_samples[0,70:72,0]= -1
input_samples[0,80:82,0]= 1
for (inp_, tau_mem, spk_threshold, spike_mode, prop) in [(9.0, 0.009899999999999999, 0.9, "D", "O")]:
    o = run(input_samples.clone(), inp_=inp_, tau_mem=tau_mem, spk_threshold=spk_threshold, spike_mode=spike_mode, prop=prop)
    outputs.append(o)
all_outputs.append(outputs)

outputs = []
input_samples = torch.zeros(1,nb_steps,1)
input_samples[0,5:50,0]= torch.linspace(0,1,45)
input_samples[0,70:80,0]= torch.linspace(0,1/2,10)
for (inp_, tau_mem, spk_threshold, spike_mode, prop) in [(2.2, 0.005, 0.1, "D", "R")]:
    o = run(input_samples.clone(), inp_=inp_, tau_mem=tau_mem, spk_threshold=spk_threshold, spike_mode=spike_mode, prop=prop)
    outputs.append(o)
all_outputs.append(outputs)


def show_all(all_outputs,save=False):
    num_all = sum([len(o)+1 for o in all_outputs])
    plt.figure(figsize=(5.5,0.25*num_all))
    ax = plt.subplot(1,1,1)
    ax.set_title('Biophysical properties', fontdict={'fontsize': 9, 'fontweight': 'medium'})
    c = 0
    ypos = np.arange(num_all,0,-1)
    labels = []
    labels_prop = []
    for j,outputs in enumerate(all_outputs):
        input_samples, _, _, nb_steps, _, _ = outputs[0]
        plt.plot(input_samples[0,:,0]/input_samples.abs().max()/2+ypos[c], c='black', linestyle='-', alpha=0.99, linewidth=1.)
        c += 1
        labels.append("")
        labels_prop.append("")
        for i,(_, spikes, _, _, spike_mode, prop) in enumerate(outputs):
            spike_id = torch.where(spikes.squeeze()==1)[0]
            labels.append(spike_mode)
            labels_prop.append(prop)
            plt.scatter(spike_id, np.ones(len(spike_id))*ypos[c], label=f'spikes - {spike_mode}', zorder=2, marker='|', c='black', alpha=0.6,linewidths=1)
            c += 1
    plt.yticks(ypos, labels)
    plt.xlim(0, nb_steps)
    plt.ylim(0, num_all+1)
    plt.grid(linestyle='--',axis="y", linewidth=0.5)
    ax.set_axisbelow(True)
    plt.xlabel("Time (100 ms)", fontsize=8)
    plt.xticks([])
    ax.set_frame_on(False)
    ax.yaxis.set_tick_params(labelsize=8)
    ax1 = ax.twinx()
    ax1.set_yticks(ypos, labels_prop)
    plt.ylabel("Properties", fontsize=8)
    ax1.set_frame_on(False)
    ax.yaxis.set_ticks_position('none') 
    ax1.yaxis.set_ticks_position('none') 
    ax1.yaxis.set_tick_params(labelsize=8)
    plt.ylim(0, num_all+1)
    if save: plt.savefig("images/biophysical_properties.png")
    else: plt.show()
show_all(all_outputs, False)

"""
# Tonic Spiking, Phasic Spiking, Phasic Bursting

- (2.0, 0.0107, 1.0, "LIF")
- (2.0, 0.0107, 1.0, "T")
- (2.0, 0.0187, 0.1, "D")
- (2.0, 0.0074, 0.1, "D")

# Class 1 Excitability, Class 2 Excitability

- (5.5, 0.0115, 1.0, "LIF")
- (5.5, 0.0011, 0.9, "T")
- (8.5, 0.0025, 0.1, "D")

# Integration and Coincidence Detection

- (3.9, 0.009, 1.0, "LIF")
- (3.9, 0.009, 1.0, "T")

# Rebound Spike, Rebound Burst

- (- 1.0, 0.001, 0.3, "D")
- (- 22.3, 0.001, 0.3, "D")

# Threshold Variability

- (9.0, 0.009899999999999999, 0.9, "D")

# Accommodation

- (2.2, 0.005, 0.1, "D")
"""