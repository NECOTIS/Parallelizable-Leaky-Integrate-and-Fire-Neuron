# from https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron/tree/main/codes
import torch
import torch.nn.functional as F
from torchvision.transforms import RandomAffine
from spikingjelly.activation_based import functional
#from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import argparse
import time
import sys
import json
import random
import gc
"""
torch.backends.cudnn.benchmark = True
_seed_ = 2020
torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)
"""
import models


import tonic.datasets as tonicDatasets
import tonic.transforms as transforms
class Dataset_NMNIST_DVSGesture(torch.utils.data.Dataset):
    def __init__(self, path, dataset_name, is_train, frames_number=20, ratio=None, transform=None, preload=True):
        super(Dataset_NMNIST_DVSGesture, self).__init__()
        self.is_train = is_train
        self.transform = transform
        self.nb_class = 10
        
        fname = os.path.join(path, f"{dataset_name}_{'train' if is_train else 'test'}_T_{frames_number}.pt")
        if preload and os.path.isfile(fname):
            dataset = torch.load(fname)
            self.data = dataset['data']
            self.targets = dataset['targets']
            if ratio:
                self.data, self.targets = self.select_subset(self.data, self.targets, ratio)
            print("dataset succesfully preloaded")
        else:
            os.makedirs(path, exist_ok=True)
            tonic_dataset = getattr(tonicDatasets, dataset_name)(save_to=path, train=is_train)
            if dataset_name == "DVSGesture":
                # Remove the 'other' class from the dataset
                other_class_ids = np.where(np.array(tonic_dataset.targets)==10)
                tonic_dataset.targets = np.delete(tonic_dataset.targets, other_class_ids).tolist()
                tonic_dataset.data = np.delete(tonic_dataset.data, other_class_ids).tolist()
            sensor_size = tonic_dataset.sensor_size
            self.frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=frames_number)
            self.targets = tonic_dataset.targets
            self.tonic_dataset = tonic_dataset
            if ratio:
                self.tonic_dataset.data, self.targets = self.select_subset(self.tonic_dataset.data, self.targets, ratio)
            #data = {index:to_sparse_tensor(torch.tensor(frame_transform(sample), dtype=torch.float32)) for index,(sample,_) in enumerate(tonic_dataset)}
            self.data = {}

    def to_sparse_tensor(self, frames):
        i = torch.where(frames!=0)
        v = frames[i]
        sparse = torch.sparse_coo_tensor(torch.stack(i), v, frames.shape, dtype=torch.float32)
        return sparse
    
    def select_subset(self, data, targets, ratio):
        indexes = np.arange(len(targets)).tolist()
        subset_size = int(len(indexes) * ratio)
        random.seed(0)
        selected_indexes = random.sample(indexes, subset_size)
        
        targets_subset = [targets[ind] for ind in selected_indexes]
        if isinstance(data, dict):
            data_subset = {new_key: data[key] for new_key,key in enumerate(selected_indexes)}
        elif isinstance(data, list):
            data_subset = [data[ind] for ind in selected_indexes]
        else:
            raise ValueError("Input must be either a list or a dictionary.")
        
        return data_subset, targets_subset

        
    def __getitem__(self, index):
        if index in self.data.keys():
            frames = self.data[index].to_dense()
        else:
            sample,_ = self.tonic_dataset[index]
            frames = torch.tensor(self.frame_transform(sample), dtype=torch.float32)
            self.data[index] = self.to_sparse_tensor(frames)
        if self.transform: frames = self.transform(frames)
        return frames, self.targets[index]
            
    def __len__(self):
        return len(self.targets)




def get_configs(args):
    params = {"NMNIST":[10, 2, 64, 1e-3, 100], "DVS128Gesture":[20, 5, 3, 1e-3, 250]}
    neu = "ParaLIF-T"
    T, number_layer, batch_size, learning_rate, max_epoch = params[args.dataset_name]
    args.dir = f"{args.dataset_name}/final/{args.neu}/"
    args.log_dir_prefix = f"outputs/{args.dataset_name}/final/logs"
    args.max_epoch = max_epoch
    args.neu = neu
    args.tau_mem = 0.
    args.tau_syn = None
    args.learn_tau = True
    args.use_max_pool = True
    args.T = T
    args.number_layer = number_layer
    args.batch_size = batch_size
    args.learning_rate = learning_rate
    args.data_ratio = None
    args.data_aug = True
    args.shift = 0.15
    args.scale = 0.
    args.save_model = True
    return args


def save_results(train_loss_hist, train_acc_hist, test_acc_hist, speed_per_epoch_hist, args, model, subdir=""):
    """
    This function creates a dictionary of results from the training and testing and save it
    as a json file. 
    If the 'save_model' parameter is set to True, the trained model is also saved. 
    """
    args.device = str(args.device)
    outputs = {
       'loss_hist':train_loss_hist, 
       'train_accuracies':train_acc_hist, 
       'test_accuracies':test_acc_hist, 
       'speed_per_epoch':speed_per_epoch_hist, 
       'PARAMS': vars(args)
      }

    output_dir = f"outputs/{args.dir}"
    filename = output_dir+f"results_{args.neu}.json"
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(outputs, f)

    if args.save_model: 
        modelname = output_dir+f"model/model_{args.neu}.pt"
        os.makedirs(os.path.dirname(modelname), exist_ok=True)
        torch.save(model, modelname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # init_tau, batch_size, learning_rate, T_max, log_dir, neu
    parser.add_argument('-init_tau', type=float, default=2.)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-learning_rate', type=float, default=1e-4)
    parser.add_argument('-T_max', type=int, default=64)
    parser.add_argument('-neu', type=str, default="PLIF")
    #parser.add_argument('-alpha_learnable', action='store_true', default=False)
    parser.add_argument('-use_max_pool', action='store_true', default=False)
    parser.add_argument('-device', type=str)
    parser.add_argument('-dataset_name', type=str)
    parser.add_argument('-log_dir_prefix', type=str)
    parser.add_argument('-T', type=int)
    parser.add_argument('-channels', type=int, default=128)
    parser.add_argument('-number_layer', type=int)
    #parser.add_argument('-split_by', type=str)
    #parser.add_argument('-normalization', type=str)
    parser.add_argument('-max_epoch', type=int, default=1024)
    parser.add_argument('-detach_reset', action='store_true', default=False)

    parser.add_argument('-tau_mem', default=1e-2, type=float, help='tau_mem')
    parser.add_argument('-tau_syn', default=1e-4, type=float, help='tau_syn')
    parser.add_argument('-data_ratio', type=float)
    #parser.add_argument('-dropout', type=float, default=0)
    parser.add_argument('-data_aug', action='store_true', default=False)
    parser.add_argument('-shift', type=float, default=0)
    parser.add_argument('-scale', type=float, default=0)
    parser.add_argument('-save_model', action='store_true', default=False)
    parser.add_argument('-recurrent', action='store_true', default=False)
    parser.add_argument('-learn_tau', action='store_true', default=False)
    parser.add_argument('-best_config', action='store_true', default=False)

    args = parser.parse_args()
    argv = ' '.join(sys.argv)
    if args.best_config: args = get_configs(args)

    print(args)
    init_tau = args.init_tau
    tau_mem = args.tau_mem
    tau_syn = args.tau_syn
    learn_tau = args.learn_tau
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    T_max = args.T_max
    neu = args.neu
    #alpha_learnable = args.alpha_learnable
    use_max_pool = args.use_max_pool
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    dataset_name = args.dataset_name
    dataset_dir = '../../datasets/'
    log_dir_prefix = args.log_dir_prefix
    T = args.T
    max_epoch = args.max_epoch
    detach_reset = args.detach_reset

    number_layer = args.number_layer
    channels = args.channels
    data_ratio = args.data_ratio
    recurrent = args.recurrent

    dir_name = f'tau_mem_{tau_mem}_tau_syn_{tau_syn}_neu_{neu}_c_{channels}_n_{number_layer}'
    log_dir = os.path.join(log_dir_prefix, dir_name)

    pt_dir = os.path.join(log_dir_prefix, 'pt_' + dir_name)
    print(log_dir, pt_dir)
    if not os.path.exists(pt_dir):
        os.makedirs(pt_dir, exist_ok=True)
    class_num = 10
    
    data_aug_fn = RandomAffine(degrees=0, translate=(args.shift, args.shift), scale=(1-args.scale,1+args.scale)) if args.data_aug else None
    name = "DVSGesture" if dataset_name == 'DVS128Gesture' else dataset_name

    train_data_loader = torch.utils.data.DataLoader(
        dataset=Dataset_NMNIST_DVSGesture(dataset_dir, name, True, frames_number=T, ratio=data_ratio, transform=data_aug_fn),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=Dataset_NMNIST_DVSGesture(dataset_dir, name, False, frames_number=T, ratio=data_ratio),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True)
    

    check_point_path = None
    check_point_max_path = os.path.join(pt_dir, 'check_point_max.pt')


    check_point = None
    if check_point_path and os.path.exists(check_point_path):
        check_point = torch.load(check_point_path, map_location=device)
        net = check_point['net']
        print(net.train_times, net.max_test_accuracy)
    else:
        if dataset_name == 'NMNIST':
            net = models.NMNISTNet(T=T, init_tau=init_tau, neu=neu, use_max_pool=use_max_pool,
                                   detach_reset=detach_reset, channels=channels,
                                   number_layer=number_layer, tau_mem=tau_mem, tau_syn=tau_syn, learn_tau=learn_tau).to(device)
        elif dataset_name == 'DVS128Gesture':
            net = models.DVS128GestureNet(T=T, init_tau=init_tau, neu=neu, use_max_pool=use_max_pool,
                                          detach_reset=detach_reset, channels=channels,
                                          number_layer=number_layer, tau_mem=tau_mem, tau_syn=tau_syn, learn_tau=learn_tau).to(device)

    print(net, sum(p.numel() for p in net.parameters() if p.requires_grad))
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)

    if check_point is not None:
        optimizer.load_state_dict(check_point['optimizer'])
        scheduler.load_state_dict(check_point['scheduler'])
        log_data_list = check_point['log_data_list']
        del check_point
    else:
        log_data_list = []
    """
    if log_data_list.__len__() > 0 and not os.path.exists(log_dir):
        rewrite_tb = True
    else:
        rewrite_tb = False
    writer = SummaryWriter(log_dir)
    if rewrite_tb:
        for i in range(log_data_list.__len__()):
            for item in log_data_list[i]:
                writer.add_scalar(item[0], item[1], item[2])
    """
    
    train_loss_hist, train_acc_hist, test_acc_hist = {}, {}, {}
    speed_per_epoch_hist = []
    if net.epoch != 0:
        net.epoch += 1
    ckpt_time = time.time()
    for net.epoch in range(net.epoch, max_epoch):
        start_time = time.time()

        log_data_list.append([])
        print(f'log_dir={log_dir}, max_test_accuracy={net.max_test_accuracy}, train_times={net.train_times}, epoch={net.epoch}')
        print(args)
        print(argv)

        net.train()
        loss_, acc_ = [],[]
        for img, label in train_data_loader:
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            
            out_spikes_counter = net(img)
            out_spikes_counter_frequency = out_spikes_counter / net.T
            loss = F.mse_loss(out_spikes_counter_frequency, F.one_hot(label, class_num).float())
            loss.backward()
            optimizer.step()
            functional.reset_net(net)
            accuracy = (out_spikes_counter_frequency.argmax(dim=1) == label).float().mean().item()
            log_data_list[-1].append(('train_accuracy', accuracy, net.train_times))
            log_data_list[-1].append(('train_loss', loss.item(), net.train_times))
            loss_.append(loss.item())
            acc_.append(accuracy)
            net.train_times += 1
            torch.cuda.empty_cache()
        gc.collect()
        scheduler.step()
        train_loss_hist[net.epoch] = np.mean(loss_)
        train_acc_hist[net.epoch] = np.mean(acc_)
        print('train accuracy = ', train_acc_hist[net.epoch])
        
        if (net.epoch+1)%10==0:
            net.eval()
            with torch.no_grad():
                test_sum = 0
                correct_sum = 0
                for img, label in test_data_loader:
                    img = img.to(device)
                    label = label.to(device)
                    out_spikes_counter = net(img)
                    correct_sum += (out_spikes_counter.argmax(dim=1) == label).float().sum().item()
                    test_sum += label.numel()
                    functional.reset_net(net)
                test_accuracy = correct_sum / test_sum
                print('test_accuracy', test_accuracy)
                log_data_list[-1].append(('test_accuracy', test_accuracy, net.epoch))
                test_acc_hist[net.epoch] = test_accuracy
                if neu=='PLIF':
                    plif_idx = 0
                    for m in net.modules():
                        if isinstance(m, models.PLIFNode):
                            log_data_list[-1].append(('w' + str(plif_idx), m.w.item(), net.train_times))
                            plif_idx += 1

                #print('Writing....')
                #for item in log_data_list[-1]:
                #    writer.add_scalar(item[0], item[1], item[2])
                    
                if net.max_test_accuracy < test_accuracy:
                    net.max_test_accuracy = test_accuracy
                    args.max_test_accuracy = test_accuracy

                #print('Written.')
    
    
                speed_per_epoch = time.time() - start_time
                speed_per_epoch_hist.append(speed_per_epoch)
                print('speed per epoch', speed_per_epoch)
                args.epoch = net.epoch
                save_results(train_loss_hist, train_acc_hist, test_acc_hist, speed_per_epoch_hist, args, net, subdir=f"{net.epoch+1}/")

    #writer.close()
