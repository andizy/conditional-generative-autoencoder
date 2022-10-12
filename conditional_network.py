import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class CondNetNF(nn.Module):

    def __init__(self, latent_dim=256, in_res=64, c=3):
        super().__init__()
        self.in_res = in_res
        self.c = c
        prev_ch = c
        CNNs = []
        num_layers = [128,128,128,256]
        for i in range(len(num_layers)):
            CNNs.append(nn.Conv2d(prev_ch, num_layers[i] ,3,
                                  padding = 'same'))
            prev_ch = num_layers[i]
        if in_res == 64:
            num_layers = [256]
            for i in range(len(num_layers)):
                CNNs.append(nn.Conv2d(prev_ch, num_layers[i] ,3,
                                      padding = 'same'))
                prev_ch = num_layers[i]
        
        if in_res == 256:
            num_layers = [256,256,256]
            for i in range(len(num_layers)):
                CNNs.append(nn.Conv2d(prev_ch, num_layers[i] ,3,
                                      padding = 'same'))
                prev_ch = num_layers[i]

        self.CNNs = nn.ModuleList(CNNs)
        self.maxpool = nn.MaxPool2d(2, 2)
        
        feature_dim = 2 * 2 * 256
        mlps = []
        mlps.append(nn.Linear(feature_dim, 2*latent_dim))
        mlps.append(nn.Linear(2*latent_dim , latent_dim))

        self.mlps = nn.ModuleList(mlps)
       

    def forward(self, x):

        for i in range(len(self.CNNs)):
            x = self.CNNs[i](x)
            x = torch.sigmoid(x)
            x = self.maxpool(x)
        x = torch.flatten(x, 1)
        for i in range(len(self.mlps)-1):
            x = self.mlps[i](x)
            x = F.sigmoid(x)
        x = self.mlps[-1](x)
        
        return x



class CondNetAutoEncoder(nn.Module):

    def __init__(self, in_res=64, out_res=64, c=3, out_c=3):
        super().__init__()
        self.in_res = in_res
        self.c = c
        prev_ch = c
        CNNs = []
        STATIC_C = 64
        
        for _ in range(int(np.log2(in_res//out_res)-1)):
            CNNs.append(nn.Conv2d(prev_ch, STATIC_C ,3,
                                  padding = 'same'))
            prev_ch = STATIC_C
        
        CNNs.append(nn.Conv2d(prev_ch, out_c,3,
                                  padding = 'same'))
        
        self.CNNs = nn.ModuleList(CNNs)
        self.maxpool = nn.MaxPool2d(2, 2)
        
       

    def forward(self, x):

        for i in range(len(self.CNNs)):
            x = self.CNNs[i](x)
            x = torch.relu(x)
            x = self.maxpool(x)
        return x

