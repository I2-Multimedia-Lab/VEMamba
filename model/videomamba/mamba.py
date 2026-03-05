import torch
import torch.nn as nn
import math
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import repeat,rearrange
from model.videomamba.get_restore_seq import *
# from get_restore_seq import *

import torch.nn.functional as F


class WeightedFusion(nn.Module):
    def __init__(self, C):
        super().__init__()
        # MLP: 4C -> hidden_dim -> 4
        hidden_dim = C*2
        self.mlp = nn.Sequential(
            nn.Linear(4 * C, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, x1, x2, x3, x4):
        """
        x1, x2, x3, x4: (B, L, C)
        return: (B, L, C)
        """
        B, L, C = x1.shape
        
        
        x_cat = torch.cat([x1, x2, x3, x4], dim=-1)  # (B, L, 4C)
        
        weights = self.mlp(x_cat)  
        weights = F.softmax(weights, dim=-1)  # (B, L, 4)
        
        x_stack = torch.stack([x1, x2, x3, x4], dim=-2)  # (B, L, 4, C)
        weights = weights.unsqueeze(-1)  # (B, L, 4, 1)
        out = (x_stack * weights).sum(dim=-2)  # (B, L, C)
        
        return out
class Mamba3D(nn.Module):
    def __init__(
            self,
            d_model,
            # directions=[0,1,2,3,4,5,6,7],
            d_state=16,
            d_conv=3,
            expand=1.5,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            scan_mode="continue",
            **kwargs,
            
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        
        assert self.d_inner % 2 == 0, "d_inner must be divisible by 2 for two-stream processing"
        self.d_inner_half = self.d_inner // 2

        
        self.directions_s1 = [2, 3, 6, 7]  # HWF, WHF and reverse
        self.directions_s2 = [0, 1, 4, 5]  # FHW, FWH and reverse
        self.num_directions_s1 = len(self.directions_s1)
        self.num_directions_s2 = len(self.directions_s2)

       
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        
        self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=(d_conv, d_conv, d_conv),
            padding=(d_conv // 2, d_conv // 2, d_conv // 2),
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        
        x_proj_weight_s1 = [nn.Linear(self.d_inner_half, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs).weight for _ in range(self.num_directions_s1)]
        self.x_proj_weight_s1 = nn.Parameter(torch.stack(x_proj_weight_s1, dim=0))

        dt_projs_s1 = [self.dt_init(self.dt_rank, self.d_inner_half, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs) for _ in range(self.num_directions_s1)]
        self.dt_projs_weight_s1 = nn.Parameter(torch.stack([dt_proj.weight for dt_proj in dt_projs_s1], dim=0))
        self.dt_projs_bias_s1 = nn.Parameter(torch.stack([dt_proj.bias for dt_proj in dt_projs_s1], dim=0))

        self.A_logs_s1 = self.A_log_init(self.d_state, self.d_inner_half, copies=self.num_directions_s1, merge=True)
        self.Ds_s1 = self.D_init(self.d_inner_half, copies=self.num_directions_s1, merge=True)
        
        self.fusion1 = WeightedFusion(C=self.d_inner_half)

       
        x_proj_weight_s2 = [nn.Linear(self.d_inner_half, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs).weight for _ in range(self.num_directions_s2)]
        self.x_proj_weight_s2 = nn.Parameter(torch.stack(x_proj_weight_s2, dim=0))

        dt_projs_s2 = [self.dt_init(self.dt_rank, self.d_inner_half, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs) for _ in range(self.num_directions_s2)]
        self.dt_projs_weight_s2 = nn.Parameter(torch.stack([dt_proj.weight for dt_proj in dt_projs_s2], dim=0))
        self.dt_projs_bias_s2 = nn.Parameter(torch.stack([dt_proj.bias for dt_proj in dt_projs_s2], dim=0))
        
        self.A_logs_s2 = self.A_log_init(self.d_state, self.d_inner_half, copies=self.num_directions_s2, merge=True)
        self.Ds_s2 = self.D_init(self.d_inner_half, copies=self.num_directions_s2, merge=True)

        self.fusion2 = WeightedFusion(C=self.d_inner_half)

        
        self.selective_scan = selective_scan_fn
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        if scan_mode == "continue":
            self.seq_dict = {
                0: lambda x: Continue_FHW(x), 1: lambda x: Continue_FWH(x),
                2: lambda x: Continue_HWF(x), 3: lambda x: Continue_WHF(x),
                4: lambda x: Continue_FHW(x).flip([1]), 5: lambda x: Continue_FWH(x).flip([1]),
                6: lambda x: Continue_HWF(x).flip([1]), 7: lambda x: Continue_WHF(x).flip([1]),
            }
            self.restore_dict = {
                0: lambda x,f,h,w: Restore_FHW(x,f,h,w), 1: lambda x,f,h,w: Restore_FWH(x,f,h,w),
                2: lambda x,f,h,w: Restore_HWF(x,f,h,w), 3: lambda x,f,h,w: Restore_WHF(x,f,h,w),
                4: lambda x,f,h,w: Restore_FHW(x.flip([1]),f,h,w), 5: lambda x,f,h,w: Restore_FWH(x.flip([1]),f,h,w),
                6: lambda x,f,h,w: Restore_HWF(x.flip([1]),f,h,w), 7: lambda x,f,h,w: Restore_WHF(x.flip([1]),f,h,w),
            }
        else: # einops a little slow
            self.seq_dict = {
                0: lambda x: rearrange(x, "b c f h w -> b (f h w) c"), 1: lambda x: rearrange(x, "b c f h w -> b (f w h) c"),
                2: lambda x: rearrange(x, "b c f h w -> b (h w f) c"), 3: lambda x: rearrange(x, "b c f h w -> b (w h f) c"),
                4: lambda x: rearrange(x, "b c f h w -> b (f h w) c").flip([1]), 5: lambda x: rearrange(x, "b c f h w -> b (f w h) c").flip([1]),
                6: lambda x: rearrange(x, "b c f h w -> b (h w f) c").flip([1]), 7: lambda x: rearrange(x, "b c f h w -> b (w h f) c").flip([1]),
            }
            self.restore_dict = {
                0: lambda x,f,h,w: rearrange(x, "b (f h w) c -> (b f) (h w) c", f=f,h=h,w=w), 1: lambda x,f,h,w: rearrange(x, "b (f w h) c -> (b f) (h w) c", f=f,h=h,w=w),
                2: lambda x,f,h,w: rearrange(x, "b (h w f) c -> (b f) (h w) c", f=f,h=h,w=w), 3: lambda x,f,h,w: rearrange(x, "b (w h f) c -> (b f) (h w) c", f=f,h=h,w=w),
                4: lambda x,f,h,w: rearrange(x.flip([1]), "b (f h w) c -> (b f) (h w) c", f=f,h=h,w=w), 5: lambda x,f,h,w: rearrange(x.flip([1]), "b (f w h) c -> (b f) (h w) c", f=f,h=h,w=w),
                6: lambda x,f,h,w: rearrange(x.flip([1]), "b (h w f) c -> (b f) (h w) c", f=f,h=h,w=w), 7: lambda x,f,h,w: rearrange(x.flip([1]), "b (w h f) c -> (b f) (h w) c", f=f,h=h,w=w),
            }

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, directions: list, x_proj_weight, dt_projs_weight, dt_projs_bias, A_logs, Ds ,fusion):
       
        B, C, F, H, W = x.shape
        L = F * H * W
        K = len(directions)

        xs = torch.stack([self.seq_dict[i](x) for i in directions],
                         dim=1).view(B, K, L, C).permute(0, 1, 3, 2).contiguous()

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), dt_projs_weight)

        if self.training:
            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            Bs = Bs.view(B, K, -1, L)
            Cs = Cs.view(B, K, -1, L)
        else:
            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            Bs = Bs.view(B, K, -1, L)
            Cs = Cs.view(B, K, -1, L)

        Ds = Ds.float().view(-1)
        As = -torch.exp(A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, C, L).permute(0, 1, 3, 2)

        restored_y = [self.restore_dict[i](out_y[:, j], F, H, W) for i, j in zip(directions, range(K))]
        final_y = fusion(*restored_y)
        # final_y = sum(restored_y)
        return final_y
    def forward(self, x: torch.Tensor, video_length, height, weight):
        
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        
        x = rearrange(x, "(b f) (h w) c -> b c f h w", f=video_length, h=height, w=weight).contiguous()
        x = self.act(self.conv3d(x))  # x 形状: [b, d_inner, f, h, w]

       
        x1, x2 = x.chunk(2, dim=1)  # x1, x2 形状: [b, d_inner_half, f, h, w]

       
        y1 = self.forward_core(x1, self.directions_s1, self.x_proj_weight_s1,
                               self.dt_projs_weight_s1, self.dt_projs_bias_s1,
                               self.A_logs_s1, self.Ds_s1,self.fusion1)
        
        y2 = self.forward_core(x2, self.directions_s2, self.x_proj_weight_s2,
                               self.dt_projs_weight_s2, self.dt_projs_bias_s2,
                               self.A_logs_s2, self.Ds_s2,self.fusion2)

       
        y = torch.cat([y1, y2], dim=-1) # y 形状: [(b f), (h w), d_inner]

        
        y = y.to(x.dtype)
        y = self.out_norm(y)
        y = y * F.silu(z)  
        out = self.out_proj(y)

        if self.dropout is not None:
            out = self.dropout(out)
        return out
if __name__ == "__main__":
    model = Mamba3D(d_model=64).cuda()
    inputs = torch.randn((5*10,10*10,64)).cuda()
    out = model(inputs,video_length=5,height=10,weight = 10)
    print(out.shape)

