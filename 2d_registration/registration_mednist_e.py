from monai.utils import set_determinism, first
from monai.transforms import (
    EnsureChannelFirstD,
    Compose,
    LoadImageD,
    RandRotateD,
    RandZoomD,
    ScaleIntensityRanged,
)
from monai.data import DataLoader, Dataset, CacheDataset
from monai.config import print_config, USE_COMPILED
from monai.networks.nets import GlobalNet
from monai.networks.blocks import Warp
from monai.apps import MedNISTDataset
import numpy as np
import torch
from torch.nn import MSELoss
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('../')
from utils import get_my_monai_dir
#print_config()
set_determinism(42)


if __name__ == '__main__':


    cfg = {
        'user_name': 'p70089067',
        'split_ratio': [0.5,0.25],
        'cache_num_workers': 2,
        'is_plot':False, 
        'num_workers': 2,
        'batch_size': 128,
        'max_epochs': 150,
        'is_val_check': True,
        'lr': 1.0e-5,
    }

    root_dir = get_my_monai_dir(user_name=cfg['user_name'])
    print(root_dir)

    print(f'train, val, test split ratio: {cfg["split_ratio"][0]}, {cfg["split_ratio"][1]}, {1-sum(cfg["split_ratio"])}')
    train_data = MedNISTDataset(root_dir=root_dir, 
                             transform=None, 
                             section="training", 
                             download=False, 
                             num_workers=cfg['cache_num_workers'],
                             cache_rate=1,
                             val_frac=cfg['split_ratio'][1],
                             test_frac=1-sum(cfg['split_ratio']))
    print(len(train_data))


    # currently, both fixed_hand and moving_hand share the same images 
    # (no transformation being made)
    training_datadict = [
        {"fixed_hand": item["image"], 
         "moving_hand": item["image"]}
        for item in train_data.data
        if item["label"] == 4  # label 4 is for xray hands
    ]
    print("\n first training items: ", training_datadict[:3])


    # this is a key component !
    train_transforms = Compose(
            [
            LoadImageD(keys=["fixed_hand", "moving_hand"]),
            EnsureChannelFirstD(keys=["fixed_hand", "moving_hand"]),
            ScaleIntensityRanged(
                keys=["fixed_hand", "moving_hand"],
                a_min=0.0,
                a_max=255.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            RandRotateD(keys=["moving_hand"], 
                        range_x=np.pi / 4, 
                        prob=1.0, 
                        keep_size=True, 
                        mode="bicubic"
                        ),
            RandZoomD(keys=["moving_hand"], 
                      min_zoom=0.9, 
                      max_zoom=1.1, 
                      prob=1.0, 
                      mode="bicubic", 
                      align_corners=False
                      ),
            ]
        )

    train_ds = CacheDataset(data=training_datadict[:1000], 
                            transform=train_transforms, 
                            cache_rate=1.0, 
                            num_workers=cfg['cache_num_workers'])
    train_loader = DataLoader(train_ds, 
                              batch_size=cfg['batch_size'], 
                              num_workers=cfg['num_workers'],
                              shuffle = True)
    val_ds = CacheDataset(data=training_datadict[2000:2500], 
                          transform=train_transforms, 
                          cache_rate=1.0, 
                          num_workers=cfg['num_workers'])
    val_loader = DataLoader(val_ds, 
                            batch_size=cfg['batch_size'], 
                            num_workers=cfg['num_workers'],
                            shuffle = False)
    
    check_data = first(train_loader)
    fixed_image = check_data["fixed_hand"][0][0]
    moving_image = check_data["moving_hand"][0][0]

    if cfg['is_plot']:
        print(f"moving_image shape: {moving_image.shape}")
        print(f"fixed_image shape: {fixed_image.shape}")
        plt.figure("check", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title("moving_image")
        plt.imshow(moving_image, cmap="gray")
        plt.subplot(1, 2, 2)
        plt.title("fixed_image")
        plt.imshow(fixed_image, cmap="gray")
        plt.show()


    device = torch.device("cuda:0")
    model = GlobalNet(
        image_size=(64, 64), 
        spatial_dims=2, 
        in_channels=2, 
        num_channel_initial=16, 
        depth=3  # moving and fixed
    ).to(device)
    image_loss = MSELoss()
    #if USE_COMPILED:
    if False:
        warp_layer = Warp(3, "border").to(device)
    else:
        warp_layer = Warp("bilinear", "border").to(device)
    optimizer = torch.optim.Adam(model.parameters(), cfg['lr'])

    max_epochs = cfg['max_epochs']
    epoch_loss_values = []

    print('start training!')
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss, step = 0, 0
        for batch_data in train_loader:
            step += 1
            optimizer.zero_grad()

            moving = batch_data["moving_hand"].to(device)
            fixed = batch_data["fixed_hand"].to(device)
            ddf = model(torch.cat((moving, fixed), dim=1))
            pred_image = warp_layer(moving, ddf)
            loss = image_loss(pred_image, fixed)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # print(f"{step}/{len(train_ds) // train_loader.batch_size}, "
            #       f"train_loss: {loss.item():.4f}")

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    print('training finished!')
    if cfg['is_plot']:
        plt.figure('train loss', (9,6))
        plt.plot(epoch_loss_values)
        plt.show()

    if cfg['is_val_check']:
        for batch_data in val_loader:
            moving = batch_data["moving_hand"].to(device)
            fixed = batch_data["fixed_hand"].to(device)
            ddf = model(torch.cat((moving, fixed), dim=1))
            pred_image = warp_layer(moving, ddf)
            break

        fixed_image = fixed.detach().cpu().numpy()[:, 0]
        moving_image = moving.detach().cpu().numpy()[:, 0]
        pred_image = pred_image.detach().cpu().numpy()[:, 0]
        batch_size = 5
        plt.subplots(batch_size, 3, figsize=(8, 10))
        for b in range(batch_size):
            # moving image
            plt.subplot(batch_size, 3, b * 3 + 1)
            plt.axis("off")
            plt.title("moving image")
            plt.imshow(moving_image[b], cmap="gray")
            # fixed image
            plt.subplot(batch_size, 3, b * 3 + 2)
            plt.axis("off")
            plt.title("fixed image")
            plt.imshow(fixed_image[b], cmap="gray")
            # warped moving
            plt.subplot(batch_size, 3, b * 3 + 3)
            plt.axis("off")
            plt.title("predicted image")
            plt.imshow(pred_image[b], cmap="gray")
        plt.axis("off")
        plt.show()