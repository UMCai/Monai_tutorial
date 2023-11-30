import logging
import numpy as np
import os
from pathlib import Path
import sys
import tempfile
import torch

from monai.apps import MedNISTDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.engines import SupervisedTrainer
from monai.handlers import StatsHandler
from monai.inferers import SimpleInferer
from monai.networks import eval_mode
from monai.networks.nets import densenet121
from monai.transforms import (LoadImageD, 
                              EnsureChannelFirstD, 
                              ScaleIntensityD, 
                              Compose)

print_config()

def get_my_monai_dir(user_name = 'Shizh'):
    import os
    path = rf'C:\Users\{user_name}\Data\Monai_data_dir'
    if os.path.exists(path):
        return os.path.normpath(path)

cfg = {
    'cache_num_workers': 2,
    'split_ratio':[0.5,0.25],

}
assert get_my_monai_dir(user_name = 'p70089067') is not None, 'check data path'    
data_dir = get_my_monai_dir(user_name = 'p70089067')

transform = Compose(
    [
        LoadImageD(keys="image", image_only=True),
        EnsureChannelFirstD(keys="image"),
        ScaleIntensityD(keys="image"),
    ]
)

print(f'train, val, test split ratio: {cfg["split_ratio"][0]}, {cfg["split_ratio"][1]}, {1-sum(cfg["split_ratio"])}')
dataset = MedNISTDataset(root_dir=data_dir, 
                         transform=transform, 
                         section="training", 
                         download=False, 
                         num_workers=cfg['cache_num_workers'],
                         cache_rate=1,
                         val_frac=cfg['split_ratio'][1],
                         test_frac=1-sum(cfg['split_ratio']))
print('num of data for training:', len(dataset))

