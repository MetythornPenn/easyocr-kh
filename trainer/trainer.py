import os
import torch.backends.cudnn as cudnn
import yaml
from train import train
from utils import AttrDict
import pandas as pd

cudnn.benchmark = True
cudnn.deterministic = False

import torch

def check_cuda():
    print('\n======================================================\n')
    if torch.cuda.is_available():
        print("CUDA is available. You can use GPU acceleration.")
        print("CUDA version:", torch.version.cuda)
        print("PyTorch version:", torch.__version__)
        print("Device name:", torch.cuda.get_device_name(0))  # Get the name of the GPU device
        print("Number of CUDA devices:", torch.cuda.device_count())
    else:
        print("CUDA is not available. GPU acceleration is not supported.")
    print('\n======================================================\n')
    

check_cuda()


def get_config(file_path):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    if opt.lang_char == 'None':
        characters = ''
        for data in opt['select_data'].split('-'):
            csv_path = os.path.join(opt['train_data'], data, 'labels.csv')
            df = pd.read_csv(csv_path, sep='^([^,]+),', engine='python', usecols=['filename', 'words'], keep_default_na=False)
            all_char = ''.join(df['words'])
            characters += ''.join(set(all_char))
        characters = sorted(set(characters))
        opt.character= ''.join(characters)
    else:
        opt.character = opt.number + opt.symbol + opt.lang_char
    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
    return opt


opt = get_config("config_files/en_filtered_config.yaml")
train(opt, amp=False)