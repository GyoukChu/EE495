import math
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchaudio
import os

import DPTDiffSeg

os.environ['CUDA_VISIBLE_DEVICES']='6'

model = DPTDiffSeg._DPTDiffSeg(t_len=512, f_len=128, batch_first=True).cuda()
src = torch.rand(16, 128, 512).cuda() # shape (B, T, F)
output = model(src).to(device='cpu').detach()
print(output)
'''
    Masking example
masked_img = transform(input_image)
masked_img[masked_img!=0]=1 # masking 된 부분은 0으로, masking 안 된 부분은 1로
'''