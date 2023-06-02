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
src = torch.rand(32, 128, 512).cuda() # shape (B, F, T)
output = model(src).to(device='cpu').detach()
