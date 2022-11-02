import torch
from datasets import DatasetType


def dataset_mean_std(Xtr):
    """Return the mean of the input dataset"""
    
    mean = 0.0
    var = 0.0
    for x in Xtr:
        if DatasetType.limited_ct:
            x = x[0]
        batch_samples = x.size(0) # batch size (the last batch can have smaller size!)
        x = x.view(batch_samples, x.size(1), -1)
        mean += x.mean(2).sum(0)
        
    mean = mean / len(Xtr.dataset)
    for x in Xtr:
        if DatasetType.limited_ct:
            x = x[0]
        batch_samples = x.size(0)
        x = x.view(batch_samples, x.size(1), -1)
        var += ((x - mean.unsqueeze(1))**2).sum([0,2])
    std = torch.sqrt(var / (len(Xtr.dataset)*x.size(2)))
    return mean, std
        
    