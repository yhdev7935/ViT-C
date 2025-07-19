# utils/export_weights.py

import torch
import timm
import struct
import numpy as np

def export_weights():
    """
    Loads a pre-trained ViT model from timm, extracts its weights,
    and saves them to a binary file in the order expected by the C implementation.
    """
    # 1. Load pre-trained model
    print("Loading pre-trained ViT model 'vit_base_patch16_224'...")
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.eval()
    
    state_dict = model.state_dict()
    output_path = "vit_weights.bin"

    print(f"Exporting weights to {output_path}...")

    with open(output_path, 'wb') as f:
        # Helper function to write a tensor to the file
        def write_tensor(tensor_name):
            tensor = state_dict[tensor_name].numpy().astype(np.float32)
            print(f"  - Writing {tensor_name} with shape {tensor.shape}")
            f.write(tensor.tobytes())

        # 1. Patch + Position Embedding
        write_tensor('cls_token')
        write_tensor('pos_embed')
        # Patch embedding weights need to be reshaped for the C linear layer
        patch_embed_w = state_dict['patch_embed.proj.weight'].numpy().reshape(-1)
        print(f"  - Writing patch_embed.proj.weight with shape {state_dict['patch_embed.proj.weight'].shape}")
        f.write(patch_embed_w.astype(np.float32).tobytes())
        write_tensor('patch_embed.proj.bias')

        # 2. Transformer Encoder Blocks
        for i in range(model.blocks.num_layers):
            print(f"--- Processing Block {i} ---")
            prefix = f'blocks.{i}.'
            
            # Layer Norm 1
            write_tensor(f'{prefix}norm1.weight')
            write_tensor(f'{prefix}norm1.bias')
            
            # Attention
            # Combine Q, K, V weights and biases
            q_w = state_dict[f'{prefix}attn.qkv.weight_q']
            k_w = state_dict[f'{prefix}attn.qkv.weight_k']
            v_w = state_dict[f'{prefix}attn.qkv.weight_v']
            qkv_w = torch.cat([q_w, k_w, v_w], dim=0).numpy().astype(np.float32)
            print(f"  - Writing combined QKV weights with shape {qkv_w.shape}")
            f.write(qkv_w.tobytes())
            
            q_b = state_dict[f'{prefix}attn.qkv.bias_q']
            k_b = state_dict[f'{prefix}attn.qkv.bias_k']
            v_b = state_dict[f'{prefix}attn.qkv.bias_v']
            qkv_b = torch.cat([q_b, k_b, v_b], dim=0).numpy().astype(np.float32)
            print(f"  - Writing combined QKV biases with shape {qkv_b.shape}")
            f.write(qkv_b.tobytes())
            
            write_tensor(f'{prefix}attn.proj.weight')
            write_tensor(f'{prefix}attn.proj.bias')

            # Layer Norm 2
            write_tensor(f'{prefix}norm2.weight')
            write_tensor(f'{prefix}norm2.bias')
            
            # MLP
            write_tensor(f'{prefix}mlp.fc1.weight')
            write_tensor(f'{prefix}mlp.fc1.bias')
            write_tensor(f'{prefix}mlp.fc2.weight')
            write_tensor(f'{prefix}mlp.fc2.bias')

        # 3. Final Layer Norm and Head
        write_tensor('norm.weight')
        write_tensor('norm.bias')
        write_tensor('head.weight')
        write_tensor('head.bias')

    print("\nWeight export complete!")
    print(f"Binary file '{output_path}' has been created in the root directory.")

if __name__ == '__main__':
    export_weights() 