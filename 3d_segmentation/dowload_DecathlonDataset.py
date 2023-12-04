from monai.apps import DecathlonDataset

import sys
sys.path.append('./')
from utils import get_my_monai_dir

root_dir = get_my_monai_dir('p70089067')
print(root_dir)

tasks = ["Task01_BrainTumour", "Task02_Heart", "Task03_Liver", "Task04_Hippocampus", 
         "Task05_Prostate", "Task06_Lung", "Task07_Pancreas", "Task08_HepaticVessel", 
         "Task09_Spleen", "Task10_Colon"]
for task in tasks:
    ds1 = DecathlonDataset(
        root_dir=root_dir,
        task=task,
        transform=None,
        section="training",
        download=True,
        cache_rate=0.0,
        num_workers=4,
    )
    print(f'dowloaded {task}')