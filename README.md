# Parallelizable Leaky-Integrate-and-Fire (ParaLIF) Neuron

This repository contains code for simulating ParaLIF neuron to accelerate training of spiking neural networks (SNN). ParaLIF is presented in `Yarga, S. Y. A., & Wood, S. U. (2024). Accelerating Spiking Neural Networks with Parallelizable Leaky Integrate-and-Fire Neurons`. The ParaLIF neuron is compared to Leaky-Integrate-and-Fire (LIF) neuron on the Spiking Heidelberg Digits (SHD), Neuromorphic-TIDIGITS, Neuromorphic-MNIST, DVSGesture and Yin-Yang datasets. This repository consists of a few key components:

- `datasets.py`: This module provides a simple interface for loading and accessing training and test datasets.

- `network.py`: This module contains the implementation of the neural network itself, including code for training and evaluating the network.

- `run.py`: This is the main entry point for running the simulation. It provides a simple command-line interface for specifying various options.

- `datasets` directory: This directory contains training and test datasets.

- `neurons` directory: This directory contains implementations for the two neurons types, extending the base class in `base.py`. The available models are:

	-  `lif.py`: The Leaky Integrate-and-Fire model  
	-  `paralif.py`: The Parallelizable Leaky-Integrate-and-Fire Neuron model. It can be simulated with various spiking functions.

- `outputs` directory: This directory contains outputs generated by the simulation.


## Usage
The `run.py` script can be run using various arguments. The following are available:

- `--seed`: Random seed for reproducibility.
- `--dataset`: The dataset to use for training, options include `heidelberg`, `ntidigits`, `nmnist`, `dvsgesture` and `yinyang` .
- `--neuron`: The neuron model to use for training, options include `LIF`, `ParaLIF-SB`, `ParaLIF-GS`, `ParaLIF-D`, `ParaLIF-T` and other variations.
- `--nb_epochs`: The number of training epochs.
- `--tau_mem`: The neuron membrane time constant.
- `--tau_syn`: The neuron synaptic current time constant.
- `--batch_size`: The batch size for training.
- `--hidden_size`: The number of neurons in the hidden layer.
- `--nb_layers`: The number of hidden layers.
- `--recurrent`: Whether to use recurrent architecture or not.
- `--reg_thr`: The spiking frequency regularization threshold.
- `--reg_thr_r`: The spiking frequency regularization threshold for recurrent spikes.
- `--loss_mode`: The mode for computing the loss, options include `last`, `max`, and `mean`.
- `--data_augmentation`: Whether to use data augmentation during training.
- `--shift`: The random shift factor for data augmentation.
- `--scale`: The random scale factor for data augmentation.
- `--window_size`: Define the input time resolution.
- `--dir`: The directory to save the results.
- `--save_model`: Whether to save the trained model.
- `--best_config`: Select the best configuration for the given dataset.

### Examples - Basic
To run the code in the basic mode, the following commands can be used.
```console
python run.py --seed 0 --dataset 'heidelberg' --best_config
python run.py --seed 0 --neuron 'ParaLIF-GS' --dataset 'yinyang'
python run.py --seed 0 --neuron 'ParaLIF-T' --dataset 'ntidigits' --recurrent
```




## Citation
If you find this work helpful, please consider citing it:

```
@article{yarga2024accelerating,
  title={Accelerating Spiking Neural Networks with Parallelizable Leaky Integrate-and-Fire Neurons},
  author={Yarga, Sidi Yaya Arnaud and Wood, Sean UN},
  year={2024}
}
```