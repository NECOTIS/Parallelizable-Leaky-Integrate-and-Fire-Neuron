# from https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron/tree/main/codes

import sys
sys.path.append('../ParaLIF')

import torch
import torch.nn as nn

from spikingjelly.activation_based import surrogate, layer
from spikingjelly.activation_based.neuron import LIFNode
from spikingjelly.activation_based.neuron import ParametricLIFNode as PLIFNode

from ParaLIF import ParaLIF


class ParaLIFWrapper(nn.Module):
    def __init__(self, neuron):
        super().__init__()
        self.neuron = neuron
    def forward(self,inputs):
        if len(inputs.shape)==5:
            T, N, C, W, H = inputs.shape
            return self.neuron(inputs.permute(1,3,4,0,2).reshape(N*W*H,T,C), parallel=True).reshape(N,W,H,T,C).permute(3,0,4,1,2) # features = C
        elif len(inputs.shape)==4:
            T, N, W, H = inputs.shape
            return self.neuron(inputs.permute(1,2,0,3).reshape(N*W,T,H), parallel=True).reshape(N,W,T,H).permute(2,0,1,3) # features = H
        elif len(inputs.shape)==3:
            return self.neuron(inputs.permute(1,0,2), parallel=True).permute(1,0,2)
        else:
            return self.neuron(inputs.unsqueeze(0), parallel=True).squeeze(0)


def create_neuron(neu: str, **kwargs):
    if neu=="LIF":
        return LIFNode(tau=kwargs["init_tau"], surrogate_function=surrogate.ATan(), detach_reset=kwargs["detach_reset"], step_mode='m')
    elif neu.split('-')[0]=="PLIF":
        return PLIFNode(init_tau=kwargs["init_tau"], surrogate_function=surrogate.ATan(), detach_reset=kwargs["detach_reset"], step_mode='m')
    elif neu.split('-')[0]=="ParaLIF":
        spk_threshold = 1.
        threshold_multiple = False
        if "D" in neu.split('-')[-1]:
            spk_threshold = torch.FloatTensor(kwargs["features"]).uniform_(1e-3, 1e-1).tolist() if threshold_multiple else 0.01
        elif "T" in neu.split('-')[-1]:
            spk_threshold = torch.FloatTensor(kwargs["features"]).uniform_(0.5, 1.5).tolist() if threshold_multiple else 1.
        return ParaLIFWrapper(ParaLIF(kwargs["features"], neu.split('-')[-1], recurrent=kwargs.get("recurrent", False), fire=kwargs.get("fire", True), spk_threshold=spk_threshold, 
                             learn_threshold=False, tau_mem=kwargs['tau_mem'], tau_syn=kwargs['tau_syn'], learn_tau=kwargs['learn_tau'], surrogate_mode="atan"))
    


def create_conv_sequential(in_channels, out_channels, number_layer, init_tau, neu, use_max_pool, detach_reset, tau_mem, tau_syn, learn_tau):
    conv = [
        layer.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False, step_mode='m'),
        layer.BatchNorm2d(out_channels, step_mode='m'),
        create_neuron(neu, init_tau=init_tau, features=out_channels, detach_reset=detach_reset, tau_mem=tau_mem, tau_syn=tau_syn, learn_tau=learn_tau),
        layer.MaxPool2d(2, 2, step_mode='m') if use_max_pool else layer.AvgPool2d(2, 2, step_mode='m')
    ]

    for i in range(number_layer - 1):
        conv.extend([
            layer.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(out_channels, step_mode='m'),
            create_neuron(neu, init_tau=init_tau, features=out_channels, detach_reset=detach_reset, tau_mem=tau_mem, tau_syn=tau_syn, learn_tau=learn_tau),
            layer.MaxPool2d(2, 2, step_mode='m') if use_max_pool else layer.AvgPool2d(2, 2, step_mode='m')
        ])
    return nn.Sequential(*conv)


def create_2fc(channels, h, w, dpp, class_num, init_tau, neu, detach_reset, tau_mem, tau_syn, learn_tau):
    return nn.Sequential(
        layer.Flatten(step_mode='m'),
        layer.Dropout(dpp, step_mode='m'),
        layer.Linear(channels * h * w, channels * h * w // 4, bias=False, step_mode='m'),
        create_neuron(neu, init_tau=init_tau, features=channels * h * w // 4, detach_reset=detach_reset, tau_mem=tau_mem, tau_syn=tau_syn, learn_tau=learn_tau),
        layer.Dropout(dpp, step_mode='m'),
        layer.Linear(channels * h * w // 4, class_num * 10, bias=False, step_mode='m'),
        create_neuron(neu, init_tau=init_tau, features=class_num * 10, detach_reset=detach_reset, tau_mem=tau_mem, tau_syn=tau_syn, learn_tau=learn_tau),
    )


class NeuromorphicNet(nn.Module):
    def __init__(self, T, init_tau, neu, use_max_pool, detach_reset, tau_mem, tau_syn):
        super().__init__()
        self.T = T
        self.init_tau = init_tau
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.neu = neu
        self.use_max_pool = use_max_pool
        self.detach_reset = detach_reset

        self.train_times = 0
        self.max_test_accuracy = 0
        self.epoch = 0
        self.conv = None
        self.fc = None
        self.boost = layer.AvgPool1d(10, 10, step_mode='m')

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]
        out_spikes_counter = self.boost(self.fc(self.conv(x)).unsqueeze(2)).squeeze(2).sum(0)
        return out_spikes_counter


class NMNISTNet(NeuromorphicNet):
    def __init__(self, T, init_tau, neu, use_max_pool, detach_reset, channels, number_layer, tau_mem, tau_syn, learn_tau):
        super().__init__(T=T, init_tau=init_tau, neu=neu, use_max_pool=use_max_pool, detach_reset=detach_reset, tau_mem=tau_mem, tau_syn=tau_syn)
        w = 34
        h = 34
        self.conv = create_conv_sequential(2, channels, number_layer=number_layer, init_tau=init_tau, neu=neu, 
                                           use_max_pool=use_max_pool, detach_reset=detach_reset, tau_mem=tau_mem, tau_syn=tau_syn, learn_tau=learn_tau)
        self.fc = create_2fc(channels=channels, w=w >> number_layer, h=h >>number_layer, dpp=0.5, class_num=10, 
                             init_tau=init_tau, neu=neu, detach_reset=detach_reset, tau_mem=tau_mem, tau_syn=tau_syn, learn_tau=learn_tau)


class DVS128GestureNet(NeuromorphicNet):
    def __init__(self, T, init_tau, neu, use_max_pool, detach_reset, channels, number_layer, tau_mem, tau_syn, learn_tau):
        super().__init__(T=T, init_tau=init_tau, neu=neu, use_max_pool=use_max_pool, detach_reset=detach_reset, tau_mem=tau_mem, tau_syn=tau_syn)
        w = 128
        h = 128
        self.conv = create_conv_sequential(2, channels, number_layer=number_layer, init_tau=init_tau, neu=neu,
                                           use_max_pool=use_max_pool, detach_reset=detach_reset, tau_mem=tau_mem, tau_syn=tau_syn, learn_tau=learn_tau)
        self.fc = create_2fc(channels=channels, w=w >> number_layer, h=h >> number_layer, dpp=0.5, class_num=10,
                             init_tau=init_tau, neu=neu, detach_reset=detach_reset, tau_mem=tau_mem, tau_syn=tau_syn, learn_tau=learn_tau)

