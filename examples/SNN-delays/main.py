#import sys
#sys.path.append('./examples/SNN-delays/')

from data_sets import SHD_dataloaders, NTidigits_dataloaders #, SSC_dataloaders, GSC_dataloaders

#from config import Config
from best_config_SHD import Config as Config_shd
from best_config_NTIDIGITS import Config as Config_ntidigits
from snn_delays import SnnDelays
import torch
#from snn import SNN
import utils
import argparse

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="shd", choices=["shd", "ntidigits"])
    parser.add_argument("--neuron", type=str, default="paralif", choices=["lif", "paralif"])
    args = parser.parse_args()
    
    config = Config_ntidigits() if args.dataset=="ntidigits" else Config_shd()
    config.seed = None
    config.output_dir = f"snn_delays/{config.dataset}/{config.spiking_neuron_type}/"
    config.save_model_path = f'{config.output_dir}model_REPL.pt'
        
    return config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n=====> Device = {device} \n\n")

config = get_config()


model = SnnDelays(config).to(device)

if config.model_type == 'snn_delays_lr0':
    model.round_pos()

print(dict((name, getattr(config, name)) for name in dir(config) if not name.startswith('__')))
print(f"===> Dataset    = {config.dataset}")
print(f"===> Model type = {config.model_type}")
print(f"===> Model size = {utils.count_parameters(model)}\n\n")
print(model)


if config.dataset == 'shd':
    train_loader, valid_loader = SHD_dataloaders(config)
    test_loader = None
elif config.dataset == 'ntidigits':
    train_loader, valid_loader = NTidigits_dataloaders(config)
    test_loader = None
else:
    raise Exception(f'dataset {config.dataset} not implemented')


model.train_model(train_loader, valid_loader, test_loader, device)