# -*- coding: utf-8 -*-
"""
Created on February 2023

@author: Arnaud Yarga
"""
import os
import h5py
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import tonic.datasets as tonicDatasets
import tonic.transforms as tonicTransforms


class Dataset():
    """
    """
    def __init__(self, window_size=1e-3, device=None, augment=False, shift=0., scale=0.):
        self.window_size = window_size
        self.device = device
        self.augment = augment
        self.shift = shift
        self.scale = scale
    
    def create_dataset(self, dataset):
        assert dataset in ["heidelberg", "ntidigits", "nmnist", "dvsgesture", "yinyang"], f"Dataset '{dataset}' not supported"
        if dataset in ["heidelberg", "ntidigits"]:
            self.collate_fn_train, self.collate_fn_test = self.collate_fn, self.collate_fn
            train_set, test_set, nb_features, nb_class = self.get_shd_ntidigits_datasets(dataset)
        elif dataset in ["nmnist", "dvsgesture"]:
            self.collate_fn_train, self.collate_fn_test = self.collate_fn, self.collate_fn
            train_set, test_set, nb_features, nb_class = self.get_nmnist_DVSGesture_dataset(dataset)
        elif dataset == "yinyang":
            train_set, test_set, nb_features, nb_class = self.get_yinyang_dataset()
            self.collate_fn_train, self.collate_fn_test = None, None
            
        return train_set, test_set, nb_features, nb_class, self.collate_fn_train, self.collate_fn_test
            
            
    def get_shd_ntidigits_datasets(self, dataset):
        if self.augment:
            self.affine_transfomer = T.RandomAffine(degrees=0, translate=(self.shift, 0.), scale=(1-self.scale,1+self.scale))
            transform_function = lambda samples : self.affine_transfomer(samples.unsqueeze(0)).squeeze(0)
        else:
            transform_function = None
        train_set = Dataset_shd_ntidigits(dataset, True, window_size=self.window_size, device=self.device, transform=transform_function)
        test_set = Dataset_shd_ntidigits(dataset, False, window_size=self.window_size, device=self.device)
        return train_set, test_set, test_set.nb_features, test_set.nb_class
    
    def get_nmnist_DVSGesture_dataset(self, dataset):
        if self.augment:
            self.affine_transfomer_dvs = T.RandomAffine(degrees=0, translate=(self.shift, self.shift), scale=(1-self.scale,1+self.scale))
            transform_function = self.affine_transfomer_dvs
        else:
            transform_function = None
        
        train_set = Dataset_nmnist_DVSGesture(dataset, True, window_size=self.window_size, device=self.device, transform=transform_function)
        test_set = Dataset_nmnist_DVSGesture(dataset, False, window_size=self.window_size, device=self.device)
        return train_set, test_set, test_set.nb_features, test_set.nb_class

    def get_yinyang_dataset(self):
        train_set = YinYangDataset(size=5000, seed=42, duration=self.window_size, device=self.device)
        test_set = YinYangDataset(size=1000, seed=40, duration=self.window_size, device=self.device)
        return train_set, test_set, test_set.nb_features, test_set.nb_class
    
    
    def collate_fn(self, samples):
        max_d = max([st[0].shape[0] for st in samples])
        spike_train_batch = []
        labels_batch = []
    
        for (spike_train, label) in samples:
            # pad spike trains if needed
            pad = (0, 0, max_d-spike_train.shape[0], 0)
            if len(spike_train.shape)==4: pad = (0,0,0,0)+pad
            spike_train_batch.append(F.pad(spike_train, pad, "constant", 0))
            labels_batch.append(label)
    
        return torch.stack(spike_train_batch).to(self.device), torch.tensor(labels_batch, device=self.device, dtype=torch.long)
    



class Dataset_shd_ntidigits(torch.utils.data.Dataset):
    def __init__(self, dataset, is_train, window_size=1e-3, device=None, transform=None):
        super(Dataset_shd_ntidigits, self).__init__()
        self.is_train = is_train
        self.device = device
        self.window_size = window_size
        self.transform = transform
        self.nb_features, self.nb_class = (700,20) if dataset=="heidelberg" else (64,11)

        if dataset=="heidelberg":
            spikes_times,spikes_units,self.targets = self.read_shd()
        elif dataset=="ntidigits":
            spikes_times,spikes_units,self.targets = self.read_ntidigits()
        
        # Get the maximum duration of the spikes data
        spikes_times_digitized = [np.array(t/self.window_size, dtype=int) for t in spikes_times]
        # Convert the digitized spike times and units to sparse tensors
        self.data = [self.to_sparse_tensor(spikes_t, spikes_u) for (spikes_t, spikes_u) in zip(spikes_times_digitized,spikes_units)]


    def read_shd(self,):
        path = f"../../datasets/shd_{'train' if self.is_train else 'test'}.h5"
        assert os.path.exists(path), f"shd dataset not found at '{path}'. It is available for download at https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/"
        
        f = h5py.File(path, 'r')
        spikes_times = [k for k in f['spikes']['times']]
        spikes_units = [k.astype(np.int32) for k in f['spikes']['units']]
        targets = [k for k in f['labels']]
        f.close()

        return spikes_times,spikes_units,targets


    def read_ntidigits(self,):
        path = '../../datasets/n-tidigits.hdf5'
        assert os.path.exists(path), f"ntidigits dataset not found at '{path}'. It is available for download at https://docs.google.com/document/d/1Uxe7GsKKXcy6SlDUX4hoJVAC0-UkH-8kr5UXp0Ndi1M"
        prename = "train" if self.is_train else "test"
        
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
        t = torch.sparse_coo_tensor(torch.tensor([spikes_times.tolist(), spikes_units.tolist()]), v, shape, dtype=torch.float32, device=self.device)
        return t
        
    def __getitem__(self, index):
        if self.transform: return self.transform(self.data[index].to_dense()), self.targets[index]
        return self.data[index].to_dense(), self.targets[index]
    
    def __len__(self):
        return len(self.targets)






class Dataset_nmnist_DVSGesture(torch.utils.data.Dataset):
    def __init__(self, dataset, is_train, window_size=1e-3, device=None, transform=None):
        super(Dataset_nmnist_DVSGesture, self).__init__()
        path = '../../datasets'
        self.is_train = is_train
        self.device = device
        self.window_size = window_size
        self.transform = transform
        self.polarity = 0 if dataset=="nmnist" else 2
        new_size = (64,64) if dataset=="dvsgesture" else None
        #new_size = None
        self.nb_class = 10
        dt = self.window_size * 1e6
        
        dataset_name = "NMNIST" if dataset=="nmnist" else "DVSGesture"
        fname = os.path.join(path, f"{dataset_name}_{'train' if is_train else 'test'}_TW_{window_size}.pt")
        if os.path.isfile(fname):
            dataset = torch.load(fname)
            self.data = dataset['data']
            self.targets = dataset['targets']
            sensor_size = self.data[0].shape[2:]
        else:
            tonic_dataset = getattr(tonicDatasets, dataset_name)(save_to=path, train=is_train)
            sensor_size = tonic_dataset.sensor_size
            if dataset_name == "DVSGesture":
                # Remove the 'other' class from the dataset
                other_class_ids = np.where(np.array(tonic_dataset.targets)==10)
                tonic_dataset.targets = np.delete(tonic_dataset.targets, other_class_ids).tolist()
                tonic_dataset.data = np.delete(tonic_dataset.data, other_class_ids).tolist()
            self.frame_transform = tonicTransforms.ToFrame(sensor_size=sensor_size, time_window=dt, overlap=0.)
            self.targets = tonic_dataset.targets
            self.tonic_dataset = tonic_dataset
            self.data = {}
            
        self.nb_features = (int(self.polarity==2)+1, sensor_size[0], sensor_size[1]) if new_size is None else (int(self.polarity==2)+1, new_size[0], new_size[1])
        self.resize = T.Resize(size=new_size, interpolation=T.InterpolationMode.NEAREST) if new_size else None

    def to_sparse_tensor(self, sample):
        frames = torch.tensor(self.frame_transform(sample), dtype=torch.float32)
        if self.polarity in [0,1]: frames = frames[:,self.polarity:self.polarity+1,...]
        i = torch.where(frames!=0)
        v = frames[i]
        sparse = torch.sparse_coo_tensor(torch.stack(i), v, frames.shape, dtype=torch.float32, device=None)
        return sparse
        
    def __getitem__(self, index):
        if index not in self.data.keys():
            sample,_ = self.tonic_dataset[index]
            self.data[index] = self.to_sparse_tensor(sample)
        frames = self.data[index].to_dense()
        if self.resize: frames = self.resize(frames)
        if self.transform: return self.transform(frames), self.targets[index]
        return frames, self.targets[index]
    
    def __len__(self):
        return len(self.targets)





class YinYangDataset(torch.utils.data.Dataset):
    def __init__(self, r_small=0.1, r_big=0.5, size=1000, seed=42, mode='rate', duration=1000, device=None):
        super(YinYangDataset, self).__init__()
        # using a numpy RNG to allow compatibility to other deep learning frameworks
        self.nb_features, self.nb_class = 4,3
        self.rng = np.random.RandomState(seed)
        self.r_small = r_small
        self.r_big = r_big
        self.__vals = []
        self.__cs = []
        self.class_names = ['yin', 'yang', 'dot']
        for i in range(size):
            # keep num of class instances balanced by using rejection sampling
            # choose class for this sample
            goal_class = self.rng.randint(3)
            x, y, c = self.get_sample(goal=goal_class)
            # add mirrod axis values
            x_flipped = 1. - x
            y_flipped = 1. - y
            val = np.array([x, y, x_flipped, y_flipped])
            self.__vals.append(val)
            self.__cs.append(c)
        self.mode = mode
        self.duration = int(duration)
        self.device = device
        self.__cs = torch.tensor(self.__cs, device=self.device)
        
        self.__vals = np.array(self.__vals, dtype=np.float32)
        inputs = torch.tensor(self.__vals, dtype=torch.float32, device=self.device)
        self.inputs_spk = inputs[:,None,:].expand(-1, self.duration, -1)
        self.inputs_spk = self.rate_encoding(self.inputs_spk)

    def get_sample(self, goal=None):
        # sample until goal is satisfied
        found_sample_yet = False
        while not found_sample_yet:
            # sample x,y coordinates
            x, y = self.rng.rand(2) * 2. * self.r_big
            # check if within yin-yang circle
            if np.sqrt((x - self.r_big)**2 + (y - self.r_big)**2) > self.r_big:
                continue
            # check if they have the same class as the goal for this sample
            c = self.which_class(x, y)
            if goal is None or c == goal:
                found_sample_yet = True
                break
        return x, y, c

    def which_class(self, x, y):
        # equations inspired by
        # https://link.springer.com/content/pdf/10.1007/11564126_19.pdf
        d_right = self.dist_to_right_dot(x, y)
        d_left = self.dist_to_left_dot(x, y)
        criterion1 = d_right <= self.r_small
        criterion2 = d_left > self.r_small and d_left <= 0.5 * self.r_big
        criterion3 = y > self.r_big and d_right > 0.5 * self.r_big
        is_yin = criterion1 or criterion2 or criterion3
        is_circles = d_right < self.r_small or d_left < self.r_small
        if is_circles:
            return 2
        return int(is_yin)

    def dist_to_right_dot(self, x, y):
        return np.sqrt((x - 1.5 * self.r_big)**2 + (y - self.r_big)**2)

    def dist_to_left_dot(self, x, y):
        return np.sqrt((x - 0.5 * self.r_big)**2 + (y - self.r_big)**2)

    def __getitem__(self, index):
        if self.mode=='rate':
            #sample = self.rate_encoding(self.inputs_spk[index])
            sample = self.inputs_spk[index]
        else:
            sample = self.__vals[index].copy()
        return sample, self.__cs[index]

    def __len__(self):
        return len(self.__cs)
    
    def rate_encoding(self, sample):
        return torch.bernoulli(sample)






