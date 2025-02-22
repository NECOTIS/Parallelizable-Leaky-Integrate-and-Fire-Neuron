from utils import set_seed

import numpy as np

from torch.utils.data import DataLoader
#from torch.utils.data import random_split
from typing import Callable, Optional

#import torchvision.transforms as transforms
# ToDO : Pay attention to this bug : https://github.com/fangwei123456/spikingjelly/commit/5a03f159755414eb514c0024fd0c0a3634ec91ac
from spikingjelly.datasets.shd import SpikingHeidelbergDigits
#from spikingjelly.datasets.shd import SpikingSpeechCommands
from spikingjelly.datasets import pad_sequence_collate
import torch
import os
import h5py


class RNoise(object):
  
  def __init__(self, sig):
    self.sig = sig
        
  def __call__(self, sample):
    noise = np.abs(np.random.normal(0, self.sig, size=sample.shape).round())
    return sample + noise


class TimeNeurons_mask_aug(object):

  def __init__(self, config):
    self.config = config
  
  
  def __call__(self, x, y):
    # Sample shape: (time, neurons)
    for sample in x:
      # Time mask
      if np.random.uniform() < self.config.TN_mask_aug_proba:
        mask_size = np.random.randint(0, self.config.time_mask_size)
        ind = np.random.randint(0, sample.shape[0] - self.config.time_mask_size)
        sample[ind:ind+mask_size, :] = 0

      # Neuron mask
      if np.random.uniform() < self.config.TN_mask_aug_proba:
        mask_size = np.random.randint(0, self.config.neuron_mask_size)
        ind = np.random.randint(0, sample.shape[1] - self.config.neuron_mask_size)
        sample[:, ind:ind+mask_size] = 0

    return x, y


class CutMix(object):
  """
  Apply Spectrogram-CutMix augmentaiton which only cuts patch across time axis unlike 
  typical Computer-Vision CutMix. Applies CutMix to one batch and its shifted version.
    
  """

  def __init__(self, config):
    self.config = config
  
  
  def __call__(self, x, y):
    
    # x shape: (batch, time, neurons)
    # Go to L-1, no need to augment last sample in batch (for ease of coding)

    for i in range(x.shape[0]-1):
      # other sample to cut from
      j = i+1
      
      if np.random.uniform() < self.config.cutmix_aug_proba:
        lam = np.random.uniform()
        cut_size = int(lam * x[j].shape[0])

        ind = np.random.randint(0, x[i].shape[0] - cut_size)

        x[i][ind:ind+cut_size, :] = x[j][ind:ind+cut_size, :]

        y[i] = (1-lam) * y[i] + lam * y[j]

    return x, y



class Augs(object):

  def __init__(self, config):
    self.config = config
    self.augs = [TimeNeurons_mask_aug(config), CutMix(config)]
  
  def __call__(self, x, y):
    for aug in self.augs:
      x, y = aug(x, y)
    
    return x, y



def SHD_dataloaders(config):
  set_seed(config.seed)

  train_dataset = BinnedSpikingHeidelbergDigits(config.datasets_path, config.n_bins, train=True, data_type='frame', duration=config.time_step)
  test_dataset= BinnedSpikingHeidelbergDigits(config.datasets_path, config.n_bins, train=False, data_type='frame', duration=config.time_step)

  #train_dataset, valid_dataset = random_split(train_dataset, [0.8, 0.2])

  train_loader = DataLoader(train_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, shuffle=True, num_workers=4)
  #valid_loader = DataLoader(valid_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size)
  test_loader = DataLoader(test_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, num_workers=4)

  return train_loader, test_loader


def NTidigits_dataloaders(config):
  set_seed(config.seed)

  train_dataset = BinnedNTidigits(config.datasets_path, config.n_bins, train=True, data_type='frame', duration=config.time_step)
  test_dataset= BinnedNTidigits(config.datasets_path, config.n_bins, train=False, data_type='frame', duration=config.time_step)

  #train_dataset, valid_dataset = random_split(train_dataset, [0.8, 0.2])

  train_loader = DataLoader(train_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, shuffle=True, num_workers=4)
  #valid_loader = DataLoader(valid_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size)
  test_loader = DataLoader(test_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, num_workers=4)

  return train_loader, test_loader




class BinnedSpikingHeidelbergDigits(SpikingHeidelbergDigits):
    def __init__(
            self,
            root: str,
            n_bins: int,
            train: bool = None,
            data_type: str = 'event',
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            custom_integrate_function: Callable = None,
            custom_integrated_frames_dir_name: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        """
        The Spiking Heidelberg Digits (SHD) dataset, which is proposed by `The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks <https://doi.org/10.1109/TNNLS.2020.3044364>`_.

        Refer to :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` for more details about params information.

        .. admonition:: Note
            :class: note

            Events in this dataset are in the format of ``(x, t)`` rather than ``(x, y, t, p)``. Thus, this dataset is not inherited from :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` directly. But their procedures are similar.

        :class:`spikingjelly.datasets.shd.custom_integrate_function_example` is an example of ``custom_integrate_function``, which is similar to the cunstom function for DVS Gesture in the ``Neuromorphic Datasets Processing`` tutorial.
        """
        super().__init__(root, train, data_type, frames_number, split_by, duration, custom_integrate_function, custom_integrated_frames_dir_name, transform, target_transform)
        self.n_bins = n_bins

    def __getitem__(self, i: int):
        if self.data_type == 'event':
            events = {'t': self.h5_file['spikes']['times'][i], 'x': self.h5_file['spikes']['units'][i]}
            label = self.h5_file['labels'][i]
            if self.transform is not None:
                events = self.transform(events)
            if self.target_transform is not None:
                label = self.target_transform(label)

            return events, label

        elif self.data_type == 'frame':
            frames = np.load(self.frames_path[i], allow_pickle=True)['frames'].astype(np.float32)
            label = self.frames_label[i]

            binned_len = frames.shape[1]//self.n_bins
            binned_frames = np.zeros((frames.shape[0], binned_len))
            for i in range(binned_len):
                binned_frames[:,i] = frames[:, self.n_bins*i : self.n_bins*(i+1)].sum(axis=1)

            if self.transform is not None:
                binned_frames = self.transform(binned_frames)
            if self.target_transform is not None:
                label = self.target_transform(label)

            return binned_frames, label

class BinnedNTidigits():
    def __init__(
            self,
            root: str,
            n_bins: int,
            train: bool = None,
            data_type: str = 'event',
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            custom_integrate_function: Callable = None,
            custom_integrated_frames_dir_name: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        """
        The Spiking Heidelberg Digits (SHD) dataset, which is proposed by `The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks <https://doi.org/10.1109/TNNLS.2020.3044364>`_.

        Refer to :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` for more details about params information.

        .. admonition:: Note
            :class: note

            Events in this dataset are in the format of ``(x, t)`` rather than ``(x, y, t, p)``. Thus, this dataset is not inherited from :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` directly. But their procedures are similar.

        :class:`spikingjelly.datasets.shd.custom_integrate_function_example` is an example of ``custom_integrate_function``, which is similar to the cunstom function for DVS Gesture in the ``Neuromorphic Datasets Processing`` tutorial.
        """
        super().__init__()
        self.root = root
        self.n_bins = n_bins
        self.train = train
        self.data_type = data_type
        self.frames_number = frames_number
        self.split_by = split_by
        self.duration = duration if (duration is not None and data_type!='event') else 1
        self.custom_integrate_function = custom_integrate_function
        self.custom_integrated_frames_dir_name = custom_integrated_frames_dir_name
        self.transform = transform
        self.target_transform = target_transform
        
        window_size = 1e-3
        self.transform = transform
        self.nb_features, self.nb_class = (64,11)
        self.spikes_times,self.spikes_units,self.targets = self.read_ntidigits()
        # Get the maximum duration of the spikes data
        spikes_times_digitized = [np.array(t/window_size/self.duration, dtype=int) for t in self.spikes_times]
        # Convert the digitized spike times and units to sparse tensors
        self.data = [self.to_sparse_tensor(spikes_t, spikes_u) for (spikes_t, spikes_u) in zip(spikes_times_digitized,self.spikes_units)]
        
        
    def read_ntidigits(self,):
        path = f'{self.root}/n-tidigits.hdf5'
        assert os.path.exists(path), f"ntidigits dataset not found at '{path}'. It is available for download at https://docs.google.com/document/d/1Uxe7GsKKXcy6SlDUX4hoJVAC0-UkH-8kr5UXp0Ndi1M"
        prename = "train" if self.train else "test"
        f = h5py.File(path, 'r')
        keys = [l.decode("utf-8") for l in f[prename+'_labels'] if len(l.decode("utf-8").split('-')[-1])==1]
        spikes_times = [f[prename+'_timestamps'][k][()] for k in keys]
        spikes_units = [f[prename+'_addresses'][k][()] for k in keys]
        f.close()
        targets = [k.split('-')[-1].replace("z", "0").replace("o", "10") for k in keys]
        targets = np.array(targets, dtype=int)
        return spikes_times,spikes_units,targets
    
    def to_sparse_tensor(self, spikes_times, spikes_units):
        v = torch.ones(len(spikes_times))
        shape = [spikes_times.max()+1, self.nb_features]
        t = torch.sparse_coo_tensor(torch.tensor([spikes_times.tolist(), spikes_units.tolist()]), v, shape, dtype=torch.float32)
        return t

    def __getitem__(self, i: int):
        if self.data_type == 'event':
            events = {'t': self.spikes_times[i], 'x': self.spikes_units[i]}
            label = self.targets[i]
            if self.transform is not None:
                events = self.transform(events)
            if self.target_transform is not None:
                label = self.target_transform(label)
            return events, label

        elif self.data_type == 'frame':
            frames = self.data[i].to_dense().numpy()
            label = self.targets[i]
            binned_len = frames.shape[1]//self.n_bins
            binned_frames = np.zeros((frames.shape[0], binned_len))
            for i in range(binned_len):
                binned_frames[:,i] = frames[:, self.n_bins*i : self.n_bins*(i+1)].sum(axis=1)
            if self.transform is not None:
                binned_frames = self.transform(binned_frames)
            if self.target_transform is not None:
                label = self.target_transform(label)
            return binned_frames, label
    
    def __len__(self):
        return len(self.targets)

