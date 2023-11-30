import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import PIL
import torch
import numpy as np
from sklearn.metrics import classification_report

from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)
from monai.utils import set_determinism
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
#print_config()

### load all the functions
def get_my_monai_dir(user_name = 'Shizh'):
    import os
    path = rf'C:\Users\{user_name}\Data\Monai_data_dir'
    if os.path.exists(path):
        return os.path.normpath(path)

def lists_shuffle(list1:list, list2:list, is_seed = True):
    import random
    if is_seed:
        random.seed(0)
    temp = list(zip(list1, list2))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    return list(res1), list(res2)

class MedNISTDataset(Dataset):
    def __init__(self, X, y, transforms):
        self.X = X
        self.y = y
        self.transforms = transforms
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.transforms(self.X[index]), self.y[index]

if __name__ == '__main__':

    # Use cfg to guide the code, all the parameters can be stored here!
    cfg = {
        'is_seed': True,
        'data_dir': 'MedNIST',
        'split_ratio':[0.1,0.1], # for train, val, rest is for test
        'num_workers': 2,
        'batch_size': 32,
        'device': 'cuda',
        'max_epochs': 2,
        'val_interval': 1,
        'checkpoint': '.',
        'loss_function': 'CrossEntropyLoss()'
    }
    
    
    # set the random seed
    if cfg['is_seed']:
        set_determinism(seed=0)
    
    
    # get monai mednist data path
    assert get_my_monai_dir() is not None, 'check data path'    
    data_dir = os.path.join(get_my_monai_dir(), cfg['data_dir'])
    #print(data_dir)
    
    ### arrange data file 
    '''
    Here the class name is the foler name, and inside each folder, there are many images.
    The goal is to store all the image pathes, along with their class name. 
    '''
    # get all the class name
    class_name = sorted(os.listdir(data_dir))
    class_name.pop(-1)
    print(class_name)
    # and num_class
    num_class = len(class_name)
    # get all the image file path from each folder 
    img_files = []
    img_labels = []
    for i,name in enumerate(class_name):
        class_dir = os.path.join(data_dir, name)
        for img in os.listdir(class_dir):
            img_dir = os.path.join(class_dir,img)
            img_files.append(img_dir)
            img_labels.append(i)
    
    assert len(img_files) == len(img_labels) == 58954, 'wrong'
    
    # random split the dataset into 10% test, 10% val and 80% train
    num_total = len(img_files)
    num_train, num_val = int(num_total*cfg['split_ratio'][0]), int(num_total*cfg['split_ratio'][1])
    num_test = int(num_total - num_train - num_val)
    img_files, img_labels = lists_shuffle(img_files,img_labels)
    train_x, train_y = img_files[:num_train], img_labels[:num_train]
    val_x, val_y = img_files[:num_val], img_labels[:num_val]
    test_x, test_y = img_files[:num_test], img_labels[:num_test]
    print(f"Training count: {len(train_x)}, Validation count: " f"{len(val_x)}, Test count: {len(test_x)}")
    
    # get dataset and loader 
    train_transforms = Compose(
        [
            LoadImage(image_only = True),
            EnsureChannelFirst(),
            ScaleIntensity(),
            RandRotate(range_x = np.pi / 12, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.8, max_zoom=1.2, prob=1.0),
        ]
    )
    val_transforms = Compose(
        [
            LoadImage(image_only = True),
            EnsureChannelFirst(),
            ScaleIntensity()
        ]
    )
    
    train_ds = MedNISTDataset(train_x, train_y, train_transforms)
    val_ds = MedNISTDataset(val_x, val_y, val_transforms)
    test_ds = MedNISTDataset(test_x, test_y, val_transforms)
    
    train_loader = DataLoader(train_ds, 
                              num_workers = cfg['num_workers'], 
                              batch_size = cfg['batch_size'],
                              shuffle = True)
    val_loader = DataLoader(val_ds, 
                              num_workers = cfg['num_workers'], 
                              batch_size = cfg['batch_size'],
                              shuffle = False)
    test_loader = DataLoader(test_ds, 
                              num_workers = cfg['num_workers'], 
                              batch_size = cfg['batch_size'],
                              shuffle = False)    
    
    device = torch.device(cfg['device'])
    print(device)
    model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=num_class).to(device)
    # loss_function = CrossEntropyLoss()
    loss_function = eval(cfg['loss_function'])
    print(loss_function)
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    max_epochs = cfg['max_epochs']
    val_interval = cfg['val_interval']
    auc_metric = ROCAUCMetric()
    
    ### train loop
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    
    y_pred_trans = Compose([Activations(softmax=True)])
    y_trans = Compose([AsDiscrete(to_onehot=num_class)])
    
    print('start training!')
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
            epoch_len = len(train_ds) // train_loader.batch_size
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels = (
                        val_data[0].to(device),
                        val_data[1].to(device),
                    )
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)
                
                # this decollate_batch is to make the evaluation faster
                # be casreful, this step is after cat all the related y_pred and y
                # y_pred [num_val, 6] -> softmax
                # y [num_val] -> one_hot
                y_onehot = [y_trans(i) for i in decollate_batch(y, detach=False)]
                y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
                auc_metric(y_pred_act, y_onehot)
                result = auc_metric.aggregate()
                auc_metric.reset()
                # free up the space
                del y_pred_act, y_onehot
                metric_values.append(result)
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                if result > best_metric:
                    best_metric = result
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(cfg['checkpoint'], "best_metric_model.pth"))
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current AUC: {result:.4f}"
                    f" current accuracy: {acc_metric:.4f}"
                    f" best AUC: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
    
    print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")