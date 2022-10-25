import torch
from torchvision import transforms
import torchvision
from torchvision.datasets import ImageFolder
from my_utils import *
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchdata.datapipes.map import SequenceWrapper


import numpy as np
from PIL import Image
from logger_conf import logger

class DatasetType:
    limited_ct = "limited-ct"

class DatasetLoader(torch.utils.data.Dataset):
    def __init__(self, 
        dataset = 'mnist', 
        size=(32,32), 
        c = 1,
        missing_cone = 'vertical',
        cond = False,
        ):
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            
        ])

        self.c = c
        self.dataset = dataset
        if self.dataset == 'mnist':
            self.img_dataset = torchvision.datasets.MNIST('dataset/MNIST', train=True,
                                                    download=True)
        
        elif self.dataset == 'celeba-hq':
            celeba_path = '/raid/Amir/Projects/datasets/celeba_hq/celeba_hq_256/'
            self.img_dataset = ImageFolder(celeba_path, self.transform)
        elif self.dataset == DatasetType.limited_ct :
            self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.Grayscale(),
            transforms.ToTensor(),
            ])
            x_limited_ct_path = '/raid/Amir/Projects/datasets/CT_dataset/images/gt_train'
            self.img_dataset = ImageFolder(x_limited_ct_path, self.transform)
        if cond:
            if self.dataset == DatasetType.limited_ct:
                if missing_cone == "vertical":
                    y_folder = "/raid/Amir/Projects/datasets/CT_dataset/images/fbp_train_vertical_snr_40"
                else:
                    y_folder = "/raid/Amir/Projects/datasets/CT_dataset/images/fbp_train_horizontal_snr_40"
                self.img_dataset = ImageFolder(y_folder, self.transform)

            

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, item):
        img = self.img_dataset[item][0]
        if self.dataset == 'celeba-hq':
            img = transforms.ToPILImage()(img)
        elif self.dataset == 'limited-ct':
            img = transforms.ToPILImage()(img)
        
        
        img = self.transform(img)
        return img
    
  

def load_dataset(dataset_type = "mnist", test_pct=0.1, img_size=(32,32), c=1, cond=False):
    
    dataset = DatasetLoader(dataset = dataset_type ,size = img_size, c = c, cond = cond)
    if dataset_type == DatasetType.limited_ct:
        y_dataset = DatasetLoader(
            dataset=dataset_type,
            size = img_size,
            c=c,
            cond=True,
        )
        dataset = SequenceWrapper(dataset)
        y_dataset = SequenceWrapper(y_dataset)
        dataset = dataset.zip(y_dataset)
    test_size = int(len(dataset) * test_pct)
    train_size = int(len(dataset) - test_size)
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    logger.info("Dataset size: {dataset_size}".format(dataset_size=len(dataset)))
    logger.info("Train dataset size: {train_size}".format(train_size=len(train_ds)))
    logger.info("Test dataset size: {test_size}".format(test_size=len(test_ds)))
    return train_ds, test_ds

