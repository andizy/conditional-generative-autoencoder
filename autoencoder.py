"""
Autoencoder
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
from conditional_network import CondNetAutoEncoder
from logger_conf import logger




class Autoencoder(nn.Module):

    def __init__(self, encoder=None, decoder=None):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder




class Encoder(nn.Module):

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
            x = F.relu(x)
            x = self.maxpool(x)
        x = torch.flatten(x, 1)
        for i in range(len(self.mlps)-1):
            x = self.mlps[i](x)
            x = F.relu(x)
        
        x = self.mlps[-1](x)
        
        return x


class Decoder(nn.Module):

    def __init__(self, latent_dim=256, in_res=64, c=3, device=None):

        super().__init__()
        self.device = device
        self.in_res = in_res
        self.c = c
        prev_ch = 256
        t_CNNs = []
        CNNs = []
        if in_res == 256:
            num_layers = [32,64,128,128,128,128,self.c]
            for i in range(len(num_layers)):
                t_CNNs.append(nn.ConvTranspose2d(prev_ch, 128 ,3,
                    stride=2,padding = 1, output_padding=1))
                CNNs.append(nn.Conv2d(128, num_layers[i] ,3,
                    padding = 'same'))
                prev_ch = num_layers[i]

        elif in_res == 64:
            num_layers = [128,128,128,128,self.c]
            for i in range(len(num_layers)):
                t_CNNs.append(nn.ConvTranspose2d(prev_ch, 128 ,3,
                    stride=2,padding = 1, output_padding=1))
                CNNs.append(nn.Conv2d(128, num_layers[i] ,3,
                    padding = 'same'))
                prev_ch = num_layers[i]
        

        self.t_CNNs = nn.ModuleList(t_CNNs)
        self.CNNs = nn.ModuleList(CNNs)
        
        self.feature_dim = 2 * 2 * 256
        mlps = []
        # mlps.append(nn.Linear(latent_dim, 2*latent_dim))
        mlps.append(nn.Linear(latent_dim , self.feature_dim))

        self.mlps = nn.ModuleList(mlps)
       
    def forward(self, x, y):
        for i in range(len(self.mlps)):
            x = self.mlps[i](x)
            x = F.relu(x)
        
        x = x.view(-1, 256, 2, 2)
        
        for i in range(len(self.t_CNNs)-1):
            x = self.t_CNNs[i](x)
            x = F.relu(x)
            x = self.CNNs[i](x)
            x = F.relu(x)

        x = self.t_CNNs[-1](x)
        x = F.relu(x)
        x = self.CNNs[-1](x)

        return x


class CondEncoder(nn.Module):

    def __init__(self, latent_dim=256, in_res=64, c=3):
        super().__init__()
        self.in_res = in_res
        self.c = c
        prev_ch = c
        cond_nets = [] 
        CNNs = []
        num_layers = [128,128,128,256]
        for i in range(len(num_layers)):
            CNNs.append(nn.Conv2d(prev_ch, num_layers[i] ,3,
                                  padding = 'same'))
            prev_ch = num_layers[i]
            cond_nets.append(CondNetAutoEncoder(in_res= self.in_res, c=self.c, out_res= self.in_res//(2**(i+1)),out_c=num_layers[i]))
        if in_res == 64:
            num_layers = [256]
            for i in range(len(num_layers)):
                CNNs.append(nn.Conv2d(prev_ch, num_layers[i] ,3,
                                      padding = 'same'))
                prev_ch = num_layers[i]
                cond_nets.append(CondNetAutoEncoder(in_res= self.in_res, c=self.c, out_res= self.in_res//(2**(4 + i+1)),out_c=num_layers[i]))
        
        if in_res == 256:
            num_layers = [256,256,256]
            for i in range(len(num_layers)):
                CNNs.append(nn.Conv2d(prev_ch, num_layers[i] ,3,
                                      padding = 'same'))
                cond_nets.append(CondNetAutoEncoder(in_res= self.in_res, c=self.c, out_res= self.in_res//(2**(4 + i+1)),out_c=num_layers[i]))
                prev_ch = num_layers[i]
                # print(f" Encoder channels ")
        
        self.CNNs = nn.ModuleList(CNNs)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.cond_nets = nn.ModuleList(cond_nets)
        
        feature_dim = 2 * 2 * 256
        mlps = []
        mlps.append(nn.Linear(feature_dim, 2*latent_dim))
        mlps.append(nn.Linear(2*latent_dim , latent_dim))

        self.mlps = nn.ModuleList(mlps)
       

    def forward(self, x, y):
        """ Y is noisy sample that we are using as condition variable"""
        for i in range(len(self.CNNs)):
            x = self.CNNs[i](x)
            x = F.relu(x)
            x = self.maxpool(x)
            ### Convert y in the same level of dimensionality as the Xdimensionality
            y_hat = self.cond_nets[i](y)
            x  = x + y_hat
        x = torch.flatten(x, 1)
        for i in range(len(self.mlps)-1):
            x = self.mlps[i](x)
            x = F.relu(x)
        
        x = self.mlps[-1](x)
        
        return x


class CondDecoder(nn.Module):

    def __init__(self, latent_dim=256, in_res=64, c=3):
        super().__init__()
        self.in_res = in_res
        self.c = c
        prev_ch = 256
        t_CNNs = []
        CNNs = []
        cond_nets = []
        self.cond = CondNetAutoEncoder
        if in_res == 256:
            num_layers = [32,64,128,128,128,128,self.c]
            for i in range(len(num_layers)):
                t_CNNs.append(nn.ConvTranspose2d(prev_ch, 128 ,3,
                    stride=2,padding = 1, output_padding=1))
                CNNs.append(nn.Conv2d(128, num_layers[i] ,3,
                    padding = 'same'))
                calculatet_i  = len(num_layers) - i
                cond_nets.append(
                    CondNetAutoEncoder(
                        in_res= self.in_res, 
                        c=self.c, 
                        out_res= self.in_res//(2**(calculatet_i-1)),
                        out_c=num_layers[i]
                        ))
                prev_ch = num_layers[i]

        elif in_res == 64:
            num_layers = [128,128,128,128,self.c]
            for i in range(len(num_layers)):
                t_CNNs.append(nn.ConvTranspose2d(prev_ch, 128 ,3,
                    stride=2,padding = 1, output_padding=1))
                CNNs.append(nn.Conv2d(128, num_layers[i] ,3,
                    padding = 'same'))
                prev_ch = num_layers[i]
                calculatet_i  = len(num_layers) - i
                cond_nets.append(
                    CondNetAutoEncoder(
                        in_res= self.in_res, 
                        c=self.c, 
                        out_res= self.in_res//(2**(calculatet_i-1)),
                        out_c=num_layers[i]
                        ))
        
        self.t_CNNs = nn.ModuleList(t_CNNs)
        self.CNNs = nn.ModuleList(CNNs)
        self.cond_nets = nn.ModuleList(cond_nets)
        self.feature_dim = 2 * 2 * 256
        mlps = []
        mlps.append(nn.Linear(latent_dim , self.feature_dim))

        self.mlps = nn.ModuleList(mlps)
       
    def forward(self, x, y):
        for i in range(len(self.mlps)):
            x = self.mlps[i](x)
            x = F.relu(x)
        
        x = x.view(-1, 256, 2, 2)
        
        for i in range(len(self.t_CNNs)-1):
            x = self.t_CNNs[i](x)
            x = F.relu(x)
            x = self.CNNs[i](x)
            x = F.relu(x)
            #convert the y in the dimensionality as x

            y_hat = self.cond_nets[i](y)
            x = x + y_hat
        x = self.t_CNNs[-1](x)
        x = F.relu(x)
        x = self.CNNs[-1](x)
        #convert the y in the dimensionality as x
        return x