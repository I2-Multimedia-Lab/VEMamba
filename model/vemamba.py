from model.videomamba.mamba import Mamba3D
from model.moco import MoCo, Encoder
from distutils.version import LooseVersion

import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import repeat,rearrange
import time

import numpy as np

class PixelShuffle_Iso(nn.Module):
    """
    Anisotropic pixel shuffle along height only.
    Upscale factor r: expands height by r, keeps width the same.
    Input: (b, c, d, h, w) where c must equal r * c_out
    Output: (b, c_out, d, h * r, w)
    The channel arrangement expected: channels are interleaved so that
    channel indices 0, r, 2r, ... go to output-row-offset 0;
    channel indices 1, r+1, 2r+1, ... go to output-row-offset 1; etc.
    """
    def __init__(self, r=2):
        super().__init__()
        assert isinstance(r, int) and r >= 1
        self.r = r

    def forward(self, x):
        # x: b, c, d, h, w
        b, c, d, h, w = x.shape
        r = self.r
        assert c % r == 0, f'channels ({c}) must be divisible by r ({r})'
        c_out = c // r
        h_out = h * r

        # allocate output
        y = x.new_zeros((b, c_out, d, h_out, w))

        # for offset k in [0..r-1], take channels k, k+r, k+2r,... and place them at rows k, k+r, ...
        for k in range(r):
            ch = x[:, k::r, :, :, :]            # shape (b, c_out, d, h, w)
            y[:, :, :, k::r, :] = ch            # place into every r-th row starting at k

        return y


class Upsample(nn.Sequential):
    """Upsample module for video SR.

    Args:
        scale (int): Scale factor. Supported: 2^n, 3, 10 (and now any combination if extended).
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        assert LooseVersion(torch.__version__) >= LooseVersion('1.8.1'), \
            'PyTorch version >= 1.8.1 to support 5D PixelShuffle.'

        class Transpose_Dim12(nn.Module):
            """ Transpose Dim1 and Dim2 of a tensor (swap channel and depth)."""
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.transpose(1, 2)

        m = []

        # scale = 2^n (use anisotropic 2x steps)
        if (scale & (scale - 1)) == 0 and scale != 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv3d(num_feat, 2 * num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
                m.append(PixelShuffle_Iso(r=2))
                m.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            m.append(nn.Conv3d(num_feat, num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))

        # scale = 3 (isotropic 3x using native PixelShuffle on width+height)
        elif scale == 3:
            m.append(nn.Conv3d(num_feat, 9 * num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
            m.append(Transpose_Dim12())
            m.append(nn.PixelShuffle(3))   # this upsamples H and W both by 3
            m.append(Transpose_Dim12())
            m.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            m.append(nn.Conv3d(num_feat, num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))

        # scale = 10 (anisotropic: height x10, width x1). factorize 10 = 5 * 2
        elif scale == 10:
            # first: conv -> channels 5 * num_feat, then PixelShuffle_Iso(5) => height x5
            m.append(nn.Conv3d(num_feat, 5 * num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
            m.append(PixelShuffle_Iso(r=5))    # only upsamples height by 5
            m.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

            # second: conv -> channels 2 * num_feat, then PixelShuffle_Iso(2) => height x2 (total x10)
            m.append(nn.Conv3d(num_feat, 2 * num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
            m.append(PixelShuffle_Iso(r=2))
            m.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

            # final smoothing conv
            m.append(nn.Conv3d(num_feat, num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))

        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n, 3, and 10.')

        super(Upsample, self).__init__(*m)




class VDIM(nn.Module):
    def __init__(self, dim,resolution):
        super().__init__()

        self.kernel = nn.Sequential(
            nn.Linear(128, dim*2, bias=False),
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.resolution = resolution

    def forward(self, x, cdp):
        """
        Input: x: (B*F, H*W, C), cdp(B, 256)
        Output: x: (B*F, H*W, C)
        """       
        B = cdp.shape[0]
        F = x.shape[0]//B 
        x = self.norm(x)
        cdp=self.kernel(cdp).unsqueeze(1).unsqueeze(1)# .view(-1,1,C*2)

        cdp1,cdp2=cdp.chunk(2, dim=3)

        x = rearrange(x,"(b f) (h w) c -> b f (h w) c",b=B,f=F,h=self.resolution[0],w=self.resolution[1])
        x = x*cdp1+cdp2  
        x = rearrange(x,"b f (h w) c -> (b f) (h w) c",b=B,f=F,h=self.resolution[0],w=self.resolution[1])
        return x
    

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., input_resolution=(64,64)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.input_resolution = input_resolution
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        # self.conv = nn.Conv3d(in_features, in_features, kernel_size=(3, 3, 3), padding=(1, 1, 1),groups=in_features,bias=True)

    def forward(self, x):
        # B = 2
        # F = x.shape[0]//B
        # C = x.shape[2]
        # x = rearrange(x, "(b f) (h w) c -> b c f h w",b=B, f=F, h=self.input_resolution[0], w=self.input_resolution[1],c = C).contiguous()
        # x = self.conv(x)
        # x = rearrange(x, "b c f h w -> (b f) (h w) c",b=B, f=F, h=self.input_resolution[0], w=self.input_resolution[1],c = C).contiguous()
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        
        
        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution

        flops += 2 * H * W * self.in_features * self.hidden_features
        flops += H * W * self.hidden_features

        return flops
    

class ResDWC3D(nn.Module):
    """Depthwise 3D conv with a frozen center impulse (like your 2D version)."""
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        # depthwise conv3d: in_channels=dim, out_channels=dim, groups=dim
        self.conv = nn.Conv3d(dim, dim, kernel_size, stride=1,
                              padding=kernel_size // 2, groups=dim, bias=True)
        # create a constant kernel with center = 1 and others = 0
        k = kernel_size
        a = torch.zeros(k**3)
        center_index = (k**3) // 2
        a[center_index] = 1.0
        # shape (1,1,k,k,k) so it can broadcast when added to conv.weight (dim,1,k,k,k)
        self.conv_constant = nn.Parameter(a.reshape(1, 1, k, k, k))
        self.conv_constant.requires_grad = False

    def forward(self, x):
        # x shape: (B, C, D, H, W)
        # add fixed impulse to conv weights (broadcasting over out_channels)
        weight = self.conv.weight + self.conv_constant
        return F.conv3d(x, weight, self.conv.bias, stride=1,
                        padding=self.kernel_size // 2, groups=self.dim)

class ConvFFN3D(nn.Module):
    """
    3D version of ConvFFN: uses 1x1x1 convs for "fc1/fc2" and a depthwise 3D conv (ResDWC3D).
    Expected input shapes:
      - (B, C, D, H, W)  (recommended), or
      - you can reshape from tokens (see Mlp3D wrapper).
    """
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., ffnconv=True, kernel_size=3):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        # pointwise 1x1x1 convs act like fc applied channel-wise
        self.fc1 = nn.Conv3d(in_features, hidden_features, kernel_size=1, bias=True)
        self.act1 = act_layer()
        self.drop1 = nn.Dropout(drop)
        if ffnconv:
            self.conv = ResDWC3D(hidden_features, kernel_size=kernel_size)
        else:
            self.conv = nn.Identity()
        self.fc2 = nn.Conv3d(hidden_features, out_features, kernel_size=1, bias=True)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        # x : (B, C, D, H, W)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.conv(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

    
class RVMB(nn.Module):
    def __init__(self,
                hidden_dim: int = 32,
                # drop_path: float = 0,
                attn_drop_rate: float = 0,
                d_state: int = 16,
                ssm_ratio: float = 1.5,
                input_resolution= (64, 64),
                mlp_ratio=1.5,
                 ):
        super().__init__()
        self.ln_1 = VDIM(hidden_dim,input_resolution)
        # self.ln_1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.mamba3d = Mamba3D(d_model=hidden_dim, d_state=d_state,expand=ssm_ratio,dropout=attn_drop_rate)
        self.skip_scale1= nn.Parameter(torch.ones(hidden_dim))
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.conv_blk = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim,input_resolution=input_resolution)

        self.ln_2 = VDIM(hidden_dim)
        # self.ln_2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


        self.hidden_dim = hidden_dim
        self.input_resolution = input_resolution

    def forward(self, input, cdp):
        # input: B*F,h*w,C    cdp: B, 256
        B = cdp.shape[0]
        F = input.shape[0]// B

        # x = self.ln_1(input, cdp)
        x = self.ln_1(input)
        x = input*self.skip_scale1+self.mamba3d(x, video_length=F, height=self.input_resolution[0], weight=self.input_resolution[1])

        x = x*self.skip_scale2+self.conv_blk(self.ln_2(x))
        return x 

class RVMG(nn.Module):
    def __init__(self, 
                 dim:int=32,
                 depth:int=4,
                 input_resolution:tuple=(64,64),
                 d_state:int = 16,
                 attn_drop_rate: float = 0,
                 ssm_ratio: float = 1.5,
                 mlp_ratio=1.5
                 ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.input_resolution = input_resolution

        self.conv = nn.Conv3d(dim, dim, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                RVMB(hidden_dim=dim, 
                     d_state=d_state, 
                     ssm_ratio=ssm_ratio, 
                     input_resolution=input_resolution,
                     attn_drop_rate=attn_drop_rate, 
                     mlp_ratio=mlp_ratio)
            )
    def forward(self, x,cdp):
        B = cdp.shape[0]
        F = x.shape[0]//B
        C = x.shape[2]

        x_blk = x

        for blk in self.blocks:
            x = blk(x, cdp)
        x = rearrange(x, "(b f) (h w) c -> b c f h w",b=B, f=F, h=self.input_resolution[0], w=self.input_resolution[1],c = C).contiguous()
        x = self.conv(x)
        x = rearrange(x, "b c f h w -> (b f) (h w) c",b=B, f=F, h=self.input_resolution[0], w=self.input_resolution[1],c = C).contiguous()
        x = x + x_blk
        return x

        

class VEMamba(nn.Module):
    def __init__(self, 
                 dim:int=32,
                 depths:tuple=(4,4,4,4),
                 input_resolution:tuple=(32,128),
                 d_state:int = 16,
                 upscales:int = 4,
                 attn_drop_rate: float = 0,
                 ssm_ratio: float = 2.0,
                 mlp_ratio=2.0):
        super().__init__()
        self.upscale = upscales
        self.dim = dim
        self.input_resolution = input_resolution
        self.apprx_scale= int(2**np.around(np.log2(upscales)))

        self.upsample = Upsample(upscales,dim)



        self.first_conv = nn.Sequential(
            nn.Conv3d(1, dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            # nn.Conv3d(dim, dim, kernel_size=(3, 3, 3), padding=(1, 1, 1),stride=(1,2,2)),
        )

        self.body = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = RVMG(dim=dim,
                               depth=depths[i_layer],
                               input_resolution=input_resolution,
                               d_state=d_state,
                               attn_drop_rate=attn_drop_rate,
                               ssm_ratio=ssm_ratio,
                               mlp_ratio=mlp_ratio
                               )
            self.body.append(layer)

        self.last_conv = nn.Conv3d(dim, 1, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        # self.conv = nn.Conv3d(dim, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1))


    
    def forward(self, x, cdp=None):
        # x: B C F H W  cdp: B 128
        B,C,F,H,W = x.shape

        x = self.first_conv(x)
        x_blk = x

        x = rearrange(x, "b c f h w -> (b f) (h w) c",b=B,f=F,h=self.input_resolution[0],w=self.input_resolution[1],c=self.dim).contiguous()
        for layer in self.body:
            x = layer(x, cdp)
        x = rearrange(x, "(b f) (h w) c -> b c f h w",b=B,f=F,h=self.input_resolution[0],w=self.input_resolution[1],c=self.dim).contiguous()
        
        x = x + x_blk

        x = self.upsample(x)

        x = self.last_conv(x)
       
        return x
