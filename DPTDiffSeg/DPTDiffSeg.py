import math
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchaudio
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

class DPTBlock(nn.Module):
    def __init__(self, t_len, f_len, batch_first):
        super(DPTBlock, self).__init__()
        self.t_len = t_len
        self.f_len = f_len
        self.batch_first = batch_first
        # 512=args.frames, 128=args.n_mels
        self.encoder_layer_time = nn.TransformerEncoderLayer(d_model=self.t_len, nhead=8, dim_feedforward=self.t_len*4, 
                                                             batch_first=self.batch_first, activation='gelu')
        self.transformer_encoder_time = nn.TransformerEncoder(self.encoder_layer_time, num_layers=1)
        self.encoder_layer_freq = nn.TransformerEncoderLayer(d_model=self.f_len, nhead=8, dim_feedforward=self.f_len*4, 
                                                             batch_first=self.batch_first, activation='gelu')
        self.transformer_encoder_freq = nn.TransformerEncoder(self.encoder_layer_freq, num_layers=1)

    # example of shape
    def forward(self, x):
        out = self.transformer_encoder_time(x) # out shape: (B, T, F) / (T, B, F)
        if self.batch_first:
            out = torch.transpose(out, dim0=1, dim1=2) # out shape: (B, F, T)
        else:
            out = torch.transpose(out, dim0=0, dim1=2) # out shape: (F, B, T)
        out = self.transformer_encoder_freq(out) # out shape: (B, F, T) / (F, B, T)
        if self.batch_first:
            out = torch.transpose(out, dim0=1, dim1=2) # out shape: (B, T, F)
        else:
            out = torch.transpose(out, dim0=0, dim1=2) # out shape: (T, B, F)
        return out

''' 
Trials: conv2d vs. avg. pooling
'''
class SSDPT(nn.Module):
    def __init__(self, t_len, f_len, batch_first=True):
        super(SSDPT, self).__init__()
        self.t_len = t_len
        self.f_len = f_len
        self.batch_first = batch_first

        self.DPT1 = DPTBlock(t_len=self.t_len, f_len=self.f_len, batch_first=batch_first)
        self.DPT2 = DPTBlock(t_len=self.t_len, f_len=self.f_len, batch_first=batch_first)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=2, padding=2)
        self.DPT3 = DPTBlock(t_len=self.t_len//2, f_len=self.f_len//2, batch_first=batch_first)
        self.DPT4 = DPTBlock(t_len=self.t_len//2, f_len=self.f_len//2, batch_first=batch_first)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=2, padding=2)
    

    def forward(self, x):
        out = self.DPT1(x)
        out = self.DPT2(out)
        if self.batch_first:
            out = self.conv1(out.unsqueeze(1))
        else:
            out = torch.transpose(out, dim0=0, dim1=1) # out shape: (B, T, F)
            out = self.conv1(out.unsqueeze(1)) # out shape: (B, 1, T, F)
            out = torch.transpose(out, dim0=0, dim1=2) # out shape: (T, 1, B, F) 
        out = self.DPT3(out.squeeze(1))
        out = self.DPT4(out)
        if self.batch_first:
            out = self.conv2(out.unsqueeze(1))
        else:
            out = torch.transpose(out, dim0=0, dim1=1) # out shape: (B, T, F)
            out = self.conv2(out.unsqueeze(1)) # out shape: (B, 1, T, F)
            out = torch.transpose(out, dim0=0, dim1=2) # out shape: (T, 1, B, F) 
        return out.squeeze(1)

''' FINAL ARCHITECTURE '''

class DPTDiffSeg(nn.Module):
    def __init__(self, t_len, f_len, batch_first=True):
        super(DPTDiffSeg, self).__init__()
        self.t_len = t_len
        self.f_len = f_len
        self.batch_first = batch_first

        # 1. Pass through SSDPT
        self.SSDPT = SSDPT(t_len=self.t_len, f_len=self.f_len, batch_first=self.batch_first)

        # 2. Latent Diffusion -> Loss1, unsqueeze(1) Needed.
        self.model = Unet(
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            channels = 1
        )
        self.diffusion = GaussianDiffusion(
            self.model,
            image_size = (self.f_len//4, self.t_len//4), # note that h, w must be divisible by 2
            timesteps = 1000,   # number of steps
            sampling_timesteps = 1000//5, # number of sampling steps -> DDIM if s_t < t
            objective = 'pred_v', # See : https://arxiv.org/abs/2202.00512
            beta_schedule = 'cosine'
        )

        # 3. Decoder -> Loss2
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.InstanceNorm2d(1),
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.InstanceNorm2d(1),
        )

        self.loss2 = nn.SmoothL1Loss()

    def forward(self, x):
        latent = self.SSDPT(x) # (B, T, F) -> (B, T//4, F//4)
        
        # Loss1
        loss1 = self.diffusion(latent.unsqueeze(1))
        sampled = self.diffusion.sample(batch_size=x.shape[0]) # (B, 1, T//4, F//4)

        # Loss2
        out = self.decoder(sampled) # (B, 1, T, F) <- (B, 1, T//4, F//4)
        loss2 = self.loss2(x.unsqueeze(1), out)

        return loss1 + loss2

def _DPTDiffSeg(f_len, t_len, batch_first):
    return DPTDiffSeg(f_len=f_len, t_len=t_len, batch_first=batch_first)