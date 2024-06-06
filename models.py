import torch
import torch.nn as nn
import numpy as np
import math

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class TimeSeriesPatchEmbed(nn.Module):
    def __init__(self, seq_length, patch_size, in_channels, hidden_size, stride):
        super().__init__()
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.stride = stride
        self.num_patches = (self.seq_length - self.patch_size) // self.stride + 1
        self.hidden_size = hidden_size

        self.proj = nn.Linear(self.patch_size * in_channels, self.hidden_size)
    
    def forward(self, x):
        B, L, C = x.shape
        x = x.unfold(dimension=1, size=self.patch_size, step=self.stride)  # (B, num_patches, patch_size, C)
        x = x.permute(0, 2, 1, 3).reshape(B, self.num_patches, -1)
        x = self.proj(x)  # (B, num_patches, hidden_size)
        return x

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        )

    def forward(self, x, t_emb):
        shift_msa, scale_msa, shift_mlp, scale_mlp = self.adaLN_modulation(t_emb).chunk(4, dim=1)
        x = x + self.attn(modulate(self.norm1(x), shift_msa, scale_msa), x, x)[0]
        x = x + self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, seq_length, out_channels, patch_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.out_channels = out_channels
    
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x.view(-1, self.seq_length, self.out_channels) 

class DiT(nn.Module):
    def __init__(
        self,
        seq_length,
        in_channels,
        patch_size,
        stride,
        hidden_size=512,
        depth=14,
        num_heads=8,
        mlp_ratio=4.0,
        learn_sigma=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.seq_length = seq_length

        self.x_embedder = TimeSeriesPatchEmbed(seq_length, patch_size, in_channels, hidden_size, stride)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.num_patches = self.x_embedder.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, seq_length, self.out_channels, patch_size)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x, t):
        # print("norma", x.shape)
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D)
        # print("afternorma", x.shape)
        t_emb = self.t_embedder(t)               # (N, D)
        for block in self.blocks:
            x = block(x, t_emb)                  # (N, T, D)
        x = self.final_layer(x, t_emb)           # (N, T, out_channels)
        x = x.view(-1, self.num_patches, self.out_channels)
        B, T, C = x.shape
        target_shape = (B, self.seq_length, self.out_channels)
        # print("target shape", target_shape)
        if T < target_shape[1]:
            # print("here")
            padding = target_shape[1] - T
            x = torch.cat([x, torch.zeros(B, padding, C, device=x.device)], dim=1)
        elif T > target_shape[1]:
            x = x[:, :target_shape[1], :]

        # print(f"Adjusted model output shape: {x.shape}")
        #x = x.view(-1, self.num_patches, self.out_channels)  # Reshape according to out_channels
        return x

def DiT_XL_2(seq_length, in_channels, **kwargs):
    return DiT(seq_length=seq_length, in_channels=in_channels, hidden_size=1152, patch_size=2, stride=2, depth=28, num_heads=16, **kwargs)

def DiT_XL_4(seq_length, in_channels, **kwargs):
    return DiT(seq_length=seq_length, in_channels=in_channels, hidden_size=512, patch_size=4, depth=14, num_heads=8, **kwargs)

def DiT_XL_8(seq_length, in_channels, **kwargs):
    return DiT(seq_length=seq_length, in_channels=in_channels, hidden_size=1152, patch_size=8, depth=28, num_heads=16, **kwargs)

def DiT_L_2(seq_length, in_channels, **kwargs):
    return DiT(seq_length=seq_length, in_channels=in_channels, hidden_size=1024, patch_size=2, depth=24, num_heads=16, **kwargs)

def DiT_L_4(seq_length, in_channels, **kwargs):
    return DiT(seq_length=seq_length, in_channels=in_channels, hidden_size=1024, patch_size=4, depth=24, num_heads=16, **kwargs)

def DiT_L_8(seq_length, in_channels, **kwargs):
    return DiT(seq_length=seq_length, in_channels=in_channels, hidden_size=1024, patch_size=8, depth=24, num_heads=16, **kwargs)

def DiT_B_2(seq_length, in_channels, **kwargs):
    return DiT(seq_length=seq_length, in_channels=in_channels, hidden_size=768, patch_size=2, depth=12, num_heads=12, **kwargs)

def DiT_B_4(seq_length, in_channels, **kwargs):
    return DiT(seq_length=seq_length, in_channels=in_channels, hidden_size=768, patch_size=4, depth=12, num_heads=12, **kwargs)

def DiT_B_8(seq_length, in_channels, **kwargs):
    return DiT(seq_length=seq_length, in_channels=in_channels, hidden_size=768, patch_size=8, depth=12, num_heads=12, **kwargs)

def DiT_S_2(seq_length, in_channels, **kwargs):
    return DiT(seq_length=seq_length, in_channels=in_channels, hidden_size=384, patch_size=2, depth=12, num_heads=6, **kwargs)

def DiT_S_4(seq_length, in_channels, **kwargs):
    return DiT(seq_length=seq_length, in_channels=in_channels, hidden_size=384, patch_size=4, depth=12, num_heads=6, **kwargs)

def DiT_S_8(seq_length, in_channels, **kwargs):
    return DiT(seq_length=seq_length, in_channels=in_channels, hidden_size=384, patch_size=8, depth=12, num_heads=6, **kwargs)

DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
