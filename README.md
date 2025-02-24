# Parallelizable Leaky-Integrate-and-Fire (ParaLIF) Neuron

This repository contains code for simulating ParaLIF neuron to accelerate training of spiking neural networks (SNN). ParaLIF is presented in [`Yarga, S. Y. A., & Wood, S. U. (2024). Accelerating Spiking Neural Networks with Parallelizable Leaky Integrate-and-Fire Neurons`](https://doi.org/10.1088/2634-4386/adb7fe). The ParaLIF neuron is compared to Leaky-Integrate-and-Fire (LIF) neuron on the Spiking Heidelberg Digits (SHD), Neuromorphic-TIDIGITS, Neuromorphic-MNIST, DVSGesture, Yin-Yang and Sequencial Cifar10 datasets. 

---
## Acceleration
Across a range of neuromorphic datasets, ParaLIF demonstrated speeds up to 200 times faster than LIF neurons, consistently achieving superior accuracy on average with comparable sparsity.
This time comparison can be analyzed using the `running_time_comparison.py` script.

![ParaLIF acceleration](images/time_comparison.png)

---
## Repository Structure
This repository consists of a few key components:

- `ParaLIF` directory: This directory contains implementations for the two neurons types, extending the base class in `base.py`. The available models are:
	-  `lif.py`: The Leaky Integrate-and-Fire model  
	-  `paralif.py`: The Parallelizable Leaky-Integrate-and-Fire Neuron model. It can be simulated with various spiking functions.

- `examples` directory: This directory contains various example scripts and configurations for different network models and experiments.
    - `basic` directory: Example scripts for basic neural network setups.
        - `datasets.py`: This module provides a simple interface for loading and accessing training and test datasets.
        - `network.py`: This module contains the implementation of the neural network itself, including code for training and evaluating the network.
        - `run.py`: This is the main entry point for running the simulation. It provides a simple command-line interface for specifying various options.
        - `run_robustness.py`: This script evaluate generalization and robustness of ParaLIF and LIF on SHD dataset.
    - `plif_net` directory: An example of integrating ParaLIF into a network, presented by [Fang, W., Yu, Z., Chen, Y., Masquelier, T., Huang, T., & Tian, Y. (2021)], to classify NMNIST and DVSGesture datasets.
    - `seqcifar10` directory: An example of integrating ParaLIF into a network, presented by [Fang, W., Yu, Z., Zhou, Z., Chen, D., Chen, Y., Ma, Z., ... & Tian, Y. (2024)], to classify Sequencial Cifar10 dataset.
    - `SNN-delays` directory: An example of integrating ParaLIF into a network, presented by [Hammouamri, I., Khalfaoui-Hassani, I., & Masquelier, T. (2023)], to classify SHD and N-TIDIGITS datasets.

- `biological_features_exploration.py`:  A script designed to explore and analyze biological features of LIF and ParaLIF, by following the methodology of Izhikevich, E. M. (2004).

- `running_time_comparison.py`: This script compares the running times of ParaLIF and LIF.

- `datasets` directory: This directory contains training and test datasets.

- `requirements.txt`:  This file lists the Python dependencies required to run the project. Install them using `pip install -r requirements.txt`.


---

## Usage

1.  **Install Required Packages:** Install the necessary Python packages by running the following command:
    
    ```bash
    pip install -r requirements.txt
    
    ```
    
2.  **Create a Repository for Datasets:** Create a directory named `datasets` where the datasets will be stored:
    
    ```bash
    mkdir datasets
    
    ```
    
3.  **Download Datasets:** The **SHD** and **NTidigits** datasets need to be downloaded manually. You can obtain them from the following links:
    
    -   [Download SHD Dataset](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/)
    -   [Download NTidigits Dataset](https://docs.google.com/document/d/1Uxe7GsKKXcy6SlDUX4hoJVAC0-UkH-8kr5UXp0Ndi1M)
    
      The remaining datasets will be downloaded automatically.



4.  **Run the scripts:** The `examples/basic/run.py` script can be run using various arguments. The following are available:

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

---

### Examples - Basic
To run the code in the basic mode, the following commands can be used.
```console
python examples/basic/run.py --dataset 'heidelberg' --neuron 'ParaLIF-D' --best_config
python examples/basic/run.py --neuron 'ParaLIF-GS' --dataset 'yinyang'
python examples/basic/run.py --neuron 'ParaLIF-T' --dataset 'ntidigits' --recurrent
```


---

## Citation
If you find this work helpful, please consider citing it:

```
@article{yarga2024accelerating,
  title={Accelerating spiking neural networks with parallelizable leaky integrate-and-fire neurons},
  author={Yarga, Sidi Yaya Arnaud and Wood, Sean UN},
  journal={Neuromorphic Computing and Engineering},
  year={2024}
}
```









