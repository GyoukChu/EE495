import torch

def Optimizer(parameters, lr, weight_decay):
    optim_fn=torch.optim.AdamW(parameters, lr = lr, betas=(0.9, 0.95), weight_decay = weight_decay)
    print('Initialised AdamW optimizer')
    return optim_fn