import numpy as np 


import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from torchvision import transforms

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., quantization = False):
        super().__init__()

        self.quantization = quantization
        if self.quantization:
            self.quant = torch.ao.quantization.QuantStub()
            self.dequant = torch.ao.quantization.DeQuantStub()

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        if self.quantization:
            x = self.quant(x)
        x = self.net(x)
        if self.quantization:
            x = self.dequant(x)
        return x 

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., quantization = False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.norm = nn.LayerNorm(dim)
        
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), 
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        
        
    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)

       

        q, k, v = [rearrange(t, 'b n (h d) -> b h n d', h = self.heads) for t in qkv]
        # q, k, v = map(_rearrange, qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        attention = self.attend(dots)
        attention = self.dropout(attention)
        
        out = torch.matmul(attention, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., quantized = False):
        super().__init__()
        
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
        # self.quantized = quantized
        # if self.quantized:
        #     self.quant = torch.ao.quantization.QuantStub()
        #     self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        # if self.quantized:
        #     x = self.quant(x)
        for attention, ff in self.layers:
            x = attention(x) + x
            x = ff(x) + x
        
        x = self.norm(x)
        # if self.quantized:
        #     x = self.dequant(x)
        return x 

class PlantVIT(nn.Module):
    def __init__(self, image_shape = (3, 256, 256), patch_shape = (8, 8), num_classes = 2, hidden_dim = 8, depth=6, heads = 2, mlp_dim = 4, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0., quantized = False):
        # Super constructor
        super().__init__()

        self.heads    = heads
        self.dim_head = dim_head
        
        # Attributes
        self.image_shape  = image_shape # ( C , H , W )
        self.hidden_dim = hidden_dim
        channels, height, width = image_shape
        
        self.patch_shape = patch_shape
        patch_height, patch_width = patch_shape 
        num_patches = (height // patch_height) * (width // patch_width)
        patch_dim = channels * patch_height * patch_width
        
        
#         assert height % patch_dim == 0, "Input shape not entirely divisible by number of patches"
#         assert width % patch_dim == 0, "Input shape not entirely divisible by number of patches"
        self.patch_area = int(patch_height * patch_width)
        
#         self.input_dim = int(chw[0] * self.patch_size[0] * self.patch_size[1])
#         self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
        
        # 2) learnable classification
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.dropout     = nn.Dropout(emb_dropout)
        
        self.transformer = Transformer(hidden_dim, depth, heads, dim_head, mlp_dim, dropout)
        self.quantized = quantized
        if self.quantized:
            self.quant = torch.ao.quantization.QuantStub()
            self.dequant = torch.ao.quantization.DeQuantStub()
        
        self.to_patch_embedding = nn.Sequential(
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(hidden_dim, num_classes)
        

    def forward(self, images):
        patches = self.to_patches(images)
        if self.quantized:
            patches = self.quant(patches)
        x = self.to_patch_embedding(patches)#self.patchify(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)

        x = self.mlp_head(x)
        if self.quantized:
            x = self.dequant(x)
        return x
    
    def to_patches(self, images):
        n, c, h, w = images.shape
        transposed = torch.transpose(images, 2, 3)
        transposed = torch.transpose(images, 1, 3)
        #end result should be n, h, w, c

        patches = torch.reshape(transposed, (n, h*w // self.patch_area, self.patch_area * c))


        assert h == w, "Patchify method is implemented for square images only"


        # patches = torch.zeros(n, h*w // self.patch_area, self.patch_area * c)

        # for idx, image in enumerate(images):
        #     for i in range(self.patch_shape[0]):
        #         for j in range(self.patch_shape[0]):
        #                 patch = image[idx, i * num_patches: (i + 1) * num_patches, j * num_patches: (j + 1) * num_patches, :]
        #                 patches[idx, c * (i * self.patch_shape[0] + j)] = patch.flatten()
        return patches

    
# class MSA(nn.Module):
#     def __init__(self, d, n_heads=2):
#         super(MSA, self).__init__()
#         self.d = d
#         self.n_heads = n_heads

#         assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

#         d_head = int(d / n_heads)
#         self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
#         self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
#         self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
#         self.d_head = d_head
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, sequences):
#         # Sequences has shape (N, seq_length, token_dim)
#         # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
#         # And come back to    (N, seq_length, item_dim)  (through concatenation)
#         result = []
#         for sequence in sequences:
#             seq_result = []
#             for head in range(self.n_heads):
#                 q_mapping = self.q_mappings[head]
#                 k_mapping = self.k_mappings[head]
#                 v_mapping = self.v_mappings[head]

#                 seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
#                 q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

#                 attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
#                 seq_result.append(attention @ v)
#             result.append(torch.hstack(seq_result))
#         return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

# class EncoderBlock(nn.Module):
#     def __init__(self, hidden_d, n_heads, mlp_ratio=4):
#         super(EncoderBlock, self).__init__()
#         self.hidden_d = hidden_d
#         self.n_heads = n_heads

#         self.norm1 = nn.LayerNorm(hidden_d)
#         self.mhsa = MSA(hidden_d, n_heads)
#         self.norm2 = nn.LayerNorm(hidden_d)
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_d, mlp_ratio * hidden_d),
#             nn.GELU(),
#             nn.Linear(mlp_ratio * hidden_d, hidden_d)
#         )

#     def forward(self, x):
#         out = x + self.mhsa(self.norm1(x))
#         out = out + self.mlp(self.norm2(out))
#         return out
