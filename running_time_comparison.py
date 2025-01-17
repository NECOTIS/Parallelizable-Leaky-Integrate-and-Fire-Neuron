# -*- coding: utf-8 -*-
"""
Created on December 2025

@author: Arnaud Yarga
"""

import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from ParaLIF import LIF, ParaLIF

torch.manual_seed(0)

# Define a function to measure running time
def measure_time(model, input_tensor, parallel=True):
    start_time = time.time()
    if parallel:
        model(input_tensor)
    else:
        model(input_tensor, parallel=False)
    return time.time() - start_time


# Define a range of input lengths to test
input_lengths = np.linspace(10, 5000, 10, dtype=int)

# Initialize lists to store average times
lif_times = {}
paralif_parallel_times = {}
paralif_sequential_times = {}

# Number of trials
n_trials = 5
n_neuron = 128
batch_size = 1

for device in ["cpu", "cuda"]:
    lif_times[device] = []
    paralif_parallel_times[device] = []
    paralif_sequential_times[device] = []

    # Instantiate models
    lif = LIF(
        n_neuron=n_neuron,
        recurrent=False,
        fire=True,
        recurrent_fire=True,
        spk_threshold=1.,
        learn_threshold=False,
        tau_mem=1e-3,
        tau_syn=1e-3,
        time_step=1e-3,
        learn_tau=False,
        device=device,
        surrogate_mode="atan",
        surrogate_scale=2.,
        debug=True
    )

    paralif = ParaLIF(
        n_neuron=n_neuron,
        spike_mode="T",
        recurrent=False,
        fire=True,
        recurrent_fire=True,
        spk_threshold=1.,
        learn_threshold=False,
        tau_mem=1e-3,
        tau_syn=1e-3,
        time_step=1e-3,
        learn_tau=False,
        device=device,
        surrogate_mode="atan",
        surrogate_scale=2.,
        debug=True
    )
    # Perform timing measurements
    for length in input_lengths:
        input_signal = torch.rand((batch_size, length, n_neuron), device=device)

        # Measure average time for LIF
        lif_time = np.mean([measure_time(lif, input_signal, parallel=True) for _ in range(n_trials)]).item()
        lif_times[device].append(lif_time)

        # Measure average time for ParaLIF (parallel)
        paralif_parallel_time = np.mean([measure_time(paralif, input_signal, parallel=True) for _ in range(n_trials)]).item()
        paralif_parallel_times[device].append(paralif_parallel_time)

        # Measure average time for ParaLIF (sequential)
        paralif_sequential_time = np.mean([measure_time(paralif, input_signal, parallel=False) for _ in range(n_trials)]).item()
        paralif_sequential_times[device].append(paralif_sequential_time)

print("input_lengths:", input_lengths)
print("lif_times:", lif_times)
print("paralif_parallel_times:", paralif_parallel_times)
print("paralif_sequential_times:", paralif_sequential_times)

# Plotting the results
plt.figure(figsize=(10, 6))
for i,device in enumerate(["cpu", "cuda"]):
    plt.subplot(2, 1, i+1)
    plt.plot(input_lengths, lif_times[device], label='LIF', marker='o')
    plt.plot(input_lengths, paralif_parallel_times[device], label='ParaLIF Parallel', marker='o')
    plt.plot(input_lengths, paralif_sequential_times[device], label='ParaLIF Sequential', marker='o')
    plt.xlabel('Input Length')
    plt.ylabel('Average Running Time (seconds)')
    plt.title('Running Time Comparison of LIF and ParaLIF - '+device)
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()

























