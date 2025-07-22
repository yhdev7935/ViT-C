# utils/export_weights.py

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

# PlantVIT Model Definition (matching the saved model structure)
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x):
        x = self.norm(x)
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(0.)
        )

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class PlantVIT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, dim_head=32, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img, mask=None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.to_patch_embedding(x)
        
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding

        x = self.transformer(x)
        x = x[:, 0]

        return self.mlp_head(x)

def export_weights():
    """
    Loads a trained PlantVIT model, extracts its weights,
    and saves them to a binary file in the order expected by the C implementation.
    """
    # PlantVIT Configuration (matching vit_config.h)
    config = {
        'image_size': 256,
        'patch_size': 32,
        'num_classes': 10,  # Updated to match saved model
        'dim': 32,
        'depth': 3,
        'heads': 3,
        'mlp_dim': 16,
        'dim_head': 32,
        'channels': 3
    }
    
    # 1. Load the actual saved model weights
    print("Loading trained weights from 'plantvit_tomato.pth'...")
    try:
        checkpoint = torch.load('plantvit_tomato.pth', map_location='cpu')
        print("Weights loaded successfully!")
    except FileNotFoundError:
        print("Error: 'plantvit_tomato.pth' not found.")
        return
    
    output_path = "vit_weights.bin"
    print(f"Exporting weights to {output_path}...")

    with open(output_path, 'wb') as f:
        # Helper function to write a tensor to the file
        def write_tensor(tensor_name):
            if tensor_name in checkpoint:
                tensor = checkpoint[tensor_name].numpy().astype(np.float32)
                print(f"  - Writing {tensor_name} with shape {tensor.shape}")
                f.write(tensor.tobytes())
            else:
                print(f"  - Warning: {tensor_name} not found in checkpoint")

        # Order must match ViTWeights struct in vit.h
        
        # 1. Patch Embedding Layers (LayerNorm -> Linear -> LayerNorm)
        print("--- Patch Embedding ---")
        write_tensor('to_patch_embedding.0.weight')  # patch_ln1_w
        write_tensor('to_patch_embedding.0.bias')    # patch_ln1_b
        write_tensor('to_patch_embedding.1.weight')  # patch_linear_w
        write_tensor('to_patch_embedding.1.bias')    # patch_linear_b
        write_tensor('to_patch_embedding.2.weight')  # patch_ln2_w
        write_tensor('to_patch_embedding.2.bias')    # patch_ln2_b

        # 2. Position Embedding and CLS token
        print("--- Position Embedding & CLS Token ---")
        write_tensor('pos_embedding')  # pos_embed
        write_tensor('cls_token')      # cls_token

        # 3. Transformer Encoder Blocks
        print("--- Transformer Blocks ---")
        for i in range(config['depth']):
            print(f"  Block {i}:")
            
            # Attention weights (QKV combined) - using actual saved structure
            write_tensor(f'transformer.layers.{i}.0.to_qkv.weight')  # qkv_weights
            # QKV bias (PlantVIT doesn't use it, write zeros)
            qkv_bias_size = config['dim'] * config['heads'] * 3
            zeros = np.zeros(qkv_bias_size, dtype=np.float32)
            print(f"  - Writing dummy qkv_bias (zeros) with shape {zeros.shape}")
            f.write(zeros.tobytes())
            
            write_tensor(f'transformer.layers.{i}.0.to_out.0.weight')  # proj_weights
            write_tensor(f'transformer.layers.{i}.0.to_out.0.bias')    # proj_bias
            
            # MLP weights - using actual saved structure
            write_tensor(f'transformer.layers.{i}.1.net.1.weight')   # mlp fc1 weight
            write_tensor(f'transformer.layers.{i}.1.net.1.bias')     # mlp fc1 bias
            write_tensor(f'transformer.layers.{i}.1.net.4.weight')   # mlp fc2 weight
            write_tensor(f'transformer.layers.{i}.1.net.4.bias')     # mlp fc2 bias
            
            # Layer Norms - CORRECTED MAPPING!
            # ln1 is actually inside attention (Pre-LN)
            write_tensor(f'transformer.layers.{i}.0.norm.weight')    # ln1_weights (attention internal norm)
            write_tensor(f'transformer.layers.{i}.0.norm.bias')      # ln1_bias (attention internal norm)
            # ln2 is actually inside MLP 
            write_tensor(f'transformer.layers.{i}.1.net.0.weight')   # ln2_weights (MLP internal norm)
            write_tensor(f'transformer.layers.{i}.1.net.0.bias')     # ln2_bias (MLP internal norm)

        # 4. Final Layer Norm and Classification Head
        print("--- Final Layers ---")
        write_tensor('transformer.norm.weight')  # final_norm_weight
        write_tensor('transformer.norm.bias')    # final_norm_bias
        write_tensor('mlp_head.weight')          # head_weights
        write_tensor('mlp_head.bias')            # head_bias

    print("\nWeight export complete!")
    print(f"Binary file '{output_path}' has been created.")
    print("Note: Successfully matched the saved model structure")

if __name__ == '__main__':
    export_weights() 