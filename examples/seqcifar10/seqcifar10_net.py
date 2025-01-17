# -*- coding: utf-8 -*-
# From https://github.com/fangwei123456/Parallel-Spiking-Neuron/tree/main/sequential_cifar
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
#_seed_ = 2020
import random
#random.seed(2020)

import torchvision
from spikingjelly.activation_based import neuron, functional, surrogate, layer
import os
import time
import argparse
from torch.cuda import amp
from datetime import datetime
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode
import json
import sys
sys.path.append('../ParaLIF')
from ParaLIF import ParaLIF
from torchvision.datasets import CIFAR10





class ClassificationPresetTrain:
    def __init__(
        self,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        random_erase_prob=0.0,
    ):
        trans = []
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                trans.append(autoaugment.RandAugment(interpolation=interpolation))
            elif auto_augment_policy == "ta_wide":
                trans.append(autoaugment.TrivialAugmentWide(interpolation=interpolation))
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(autoaugment.AutoAugment(policy=aa_policy, interpolation=interpolation))
        trans.extend(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)

from torch import Tensor
from typing import Tuple
class RandomMixup(torch.nn.Module):
    """Randomly apply Mixup to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        assert num_classes > 0, "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on mixup paper, page 3.
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s
class RandomCutmix(torch.nn.Module):
    """Randomly apply Cutmix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        assert num_classes > 0, "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        W, H = torchvision.transforms.functional.get_image_size(batch)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s
    
    
class ParaLIFWrapper(nn.Module):
    def __init__(self, neuron):
        super().__init__()
        self.neuron = neuron
    def forward(self,inputs):
        if len(inputs.shape)==4:
            W, N, C, H = inputs.shape
            return self.neuron(inputs.permute(1,3,0,2).reshape(N*H,W,C)).reshape(N,H,W,C).permute(2,0,3,1)
        elif len(inputs.shape)==3:
            return self.neuron(inputs.permute(1,0,2)).permute(1,0,2)
        else:
            return self.neuron(inputs.unsqueeze(0)).squeeze(0)


def create_neuron(neu: str, **kwargs):
    if neu=="lif":
        return neuron.LIFNode(tau=2., detach_reset=True, surrogate_function=kwargs['surrogate_function'], v_reset=None, step_mode='m')
    elif neu.split('-')[0]=="ParaLIF":
        return ParaLIFWrapper(ParaLIF(kwargs["features"], neu.split('-')[-1], recurrent=kwargs["recurrent"], fire=True, recurrent_fire=kwargs["recurrent_fire"], 
                                      learn_threshold=True, tau_mem=kwargs['tau_mem'], tau_syn=kwargs['tau_syn'], learn_tau=False, surrogate_mode=kwargs["surrogate_mode"], 
                                      surrogate_scale=kwargs["surrogate_scale"], device=kwargs["device"]))
    


# 输入是 [W, N, C, H] = [32, N, 3, 32]
class CIFAR10Net(nn.Module):
    def __init__(self, channels, neu: str, T: int, class_num: int, P:int=-1, exp_init:bool=False,
                 device=None, tau_mem=1e-3, tau_syn=1e-3, surrogate_mode="atan", surrogate_scale=2., recurrent=False, recurrent_fire=True,
                 n_conv=3, n_down=2):
        super().__init__()
        conv = []
        down_T = T//(2**n_down)
        for i in range(n_down):
            for j in range(n_conv):
                if conv.__len__() == 0:
                    in_channels = 3
                else:
                    in_channels = channels
                conv.append(layer.Conv1d(in_channels, channels, kernel_size=3, padding=1, bias=False))
                conv.append(layer.BatchNorm1d(channels))
                conv.append(create_neuron(neu, T=T, features=channels, surrogate_function=surrogate.ATan(), channels=channels, P=P, exp_init=exp_init,
                                          device=device, recurrent=recurrent, tau_mem=tau_mem, tau_syn=tau_syn, surrogate_mode=surrogate_mode, surrogate_scale=surrogate_scale, recurrent_fire=recurrent_fire))

            conv.append(layer.AvgPool1d(2))

        self.conv = nn.Sequential(*conv)


        self.fc = nn.Sequential(
            layer.Flatten(),
            layer.Linear(channels * down_T, channels * down_T // 4),
            create_neuron(neu, T=T, features=channels * down_T // 4, surrogate_function=surrogate.ATan(), P=P, exp_init=exp_init,
                          device=device, recurrent=recurrent, tau_mem=tau_mem, tau_syn=tau_syn, surrogate_mode=surrogate_mode, surrogate_scale=surrogate_scale, recurrent_fire=recurrent_fire),
            layer.Linear(channels * down_T // 4, class_num),
        )

        functional.set_step_mode(self, 'm')

    def forward(self, x_seq: torch.Tensor):
        # [N, C, H, W] -> [W, N, C, H]
        x_seq = x_seq.permute(3, 0, 1, 2)
        x_seq = self.fc(self.conv(x_seq))  # [W, N, C]
        return x_seq.mean(0)




def save_results(train_loss_hist, train_acc_hist, test_acc_hist, args, model):
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
       'PARAMS': vars(args)
      }

    output_dir = f"outputs/{args.dir}"
    filename = output_dir+f"results_{args.neu}.json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(outputs, f)
    
    if args.save_model: 
        modelname = output_dir+f"model_{args.neu}.pt"
        torch.save(model.state_dict(), modelname)



def get_configs(args):
    neu = "ParaLIF-GS"
    recurrent,recurrent_fire,desc = [(False, False, "feedforward"), (True, True, "recurrent"), (True, False, "recurrent_relu")][0]
    args.dir = f"seq_cifar10/final/{desc}/{neu}/"
    args.epochs = 300
    args.neu = neu
    args.tau_mem = 5e-5
    args.tau_syn = 1e-3
    args.recurrent = recurrent
    args.recurrent_fire = recurrent_fire
    args.lr = 1e-1
    args.momentum = 0.9
    args.opt = "sgd"
    args.save_model = True
    return args


def main():

    parser = argparse.ArgumentParser(description='Classify Sequential CIFAR10/100')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=128, type=int, help='batch size')
    parser.add_argument('-epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', type=str, help='root dir of CIFAR10/100 dataset')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-opt', type=str, default='sgd', help='use which optimizer. SDG or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-channels', default=128, type=int, help='channels of CSNN')
    parser.add_argument('-neu', type=str, default='lif', help='use which neuron')
    parser.add_argument('-class-num', type=int, default=10)

    parser.add_argument('-P', type=int, default=None, help='the order of the masked/sliding PSN')
    parser.add_argument('-exp-init', action='store_true', help='use the exp init method to initialize the weight of SPSN')

    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-save_model', action='store_true')
    parser.add_argument('-tau_mem', default=1e-2, type=float, help='tau_mem')
    parser.add_argument('-tau_syn', default=1e-4, type=float, help='tau_syn')
    parser.add_argument('-surrogate_mode', type=str, default="atan")
    parser.add_argument('-surrogate_scale', type=float, default=2.)
    parser.add_argument('-recurrent', action='store_true')
    parser.add_argument('-recurrent_fire', action='store_true')
    parser.add_argument('-n_conv', type=int, default=3)
    parser.add_argument('-n_down', type=int, default=2)
    parser.add_argument('-best_config', action='store_true', default=False)


    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.dir = f"seq_cifar10_reprod/{args.neu}/"
    if args.best_config:
        args = get_configs(args)
        
    print(args)

    mixup_transforms = []
    mixup_transforms.append(RandomMixup(args.class_num, p=1.0, alpha=0.2))
    mixup_transforms.append(RandomCutmix(args.class_num, p=1.0, alpha=1.))
    mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
    collate_fn = lambda batch: mixupcutmix(*default_collate(batch))  # noqa: E731

    if args.class_num == 10:
        transform_train = ClassificationPresetTrain(mean=(0.4914, 0.4822, 0.4465),
                                                      std=(0.2023, 0.1994, 0.2010), interpolation=InterpolationMode('bilinear'),
                                                      auto_augment_policy='ta_wide',
                                                      random_erase_prob=0.1)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


    if args.class_num == 10:
        train_set = CIFAR10(root="../../datasets/", train=True, transform=transform_train, download=True)
        test_set = CIFAR10(root="../../datasets/", train=False, transform=transform_test, download=True)
        
        if args.debug:
            train_set.data = train_set.data[:2*args.b]
            train_set.targets = train_set.targets[:2*args.b]
            test_set.data = test_set.data[:2*args.b]
            test_set.targets = test_set.targets[:2*args.b]


    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.b,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        #num_workers=4,
        #pin_memory=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=False,
        drop_last=False,
        #num_workers=4,
        #pin_memory=True
    )

    net = CIFAR10Net(channels=args.channels, neu=args.neu, T=32, class_num=args.class_num, P=args.P, exp_init=args.exp_init,
                     device=args.device, tau_mem=args.tau_mem, tau_syn=args.tau_syn, surrogate_mode=args.surrogate_mode, 
                     surrogate_scale=args.surrogate_scale, recurrent=args.recurrent, n_conv=args.n_conv, n_down=args.n_down, recurrent_fire=args.recurrent_fire)
    net.to(args.device)
    print(net, sum(p.numel() for p in net.parameters() if p.requires_grad))


    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adamw':
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.)
    else:
        raise NotImplementedError(args.opt)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']
        print(max_test_acc)


    train_loss_hist = []
    train_acc_hist = []
    test_acc_hist = []
    for epoch in range(start_epoch, args.epochs):
        #start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0

        for batch_index, (img, label) in enumerate(train_data_loader):
            optimizer.zero_grad()
            img = img.to(args.device, non_blocking=True)
            label = label.to(args.device, non_blocking=True)



            with torch.cuda.amp.autocast(enabled=scaler is not None):
                y = net(img)
                loss = F.cross_entropy(y, label, label_smoothing=0.1)


            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_samples += label.shape[0]
            train_loss += loss.item() * label.shape[0]
            train_acc += (y.argmax(1) == label.argmax(1)).float().sum().item()

            functional.reset_net(net)


        train_loss /= train_samples
        train_acc /= train_samples

        lr_scheduler.step()
        
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        print(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}')

        if (epoch+1)%5==0:
            net.eval()
            test_acc = 0
            test_samples = 0
            with torch.no_grad():
                for img, label in test_data_loader:
                    img = img.to(args.device)
                    label = label.to(args.device)
                    y = net(img)
                    test_samples += label.numel()
                    test_acc += (y.argmax(1) == label).float().sum().item()
                    functional.reset_net(net)
            test_acc /= test_samples
            test_acc_hist.append(test_acc)
            if test_acc > max_test_acc:
                max_test_acc = test_acc
                args.max_test_acc = max_test_acc
            print(f'epoch = {epoch}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
            save_results(train_loss_hist, train_acc_hist, test_acc_hist, args, net)
                
    print("train_loss_hist", train_loss_hist, "train_acc_hist", train_acc_hist,"test_acc_hist", test_acc_hist)
    save_results(train_loss_hist, train_acc_hist, test_acc_hist, args, net)
    

if __name__ == '__main__':
    main()