# utils/export_weights.py

import sys
import os
sys.path.append('pv_pytorch')

import torch
import torch.nn as nn
import numpy as np
from vit import PlantVIT

def export_weights():
    """
    Loads a trained PlantVIT model, extracts its weights,
    and saves them to a binary file in the order expected by the C implementation.
    """
    
    # 1. Load the actual saved model weights
    print("Loading trained weights from 'plantvit_tomato.pth'...")
    try:
        checkpoint = torch.load('plantvit_tomato.pth', map_location='cpu')
        print("Weights loaded successfully!")
    except FileNotFoundError:
        print("Error: 'plantvit_tomato.pth' not found.")
        return
    
    # PlantVIT model configuration (matches saved model)
    config = {
        'image_shape': (3, 256, 256),
        'patch_shape': (32, 32),
        'num_classes': 10,
        'hidden_dim': 32,
        'depth': 3,
        'heads': 3,
        'mlp_dim': 16,
        'dim_head': 32,
        'dropout': 0.0,
        'emb_dropout': 0.0
    }
    
    # Create model (matches saved structure)
    model = PlantVIT(
        image_shape=config['image_shape'],
        patch_shape=config['patch_shape'],
        num_classes=config['num_classes'],
        hidden_dim=config['hidden_dim'],
        depth=config['depth'],
        heads=config['heads'],
        mlp_dim=config['mlp_dim'],
        dim_head=config['dim_head'],
        dropout=config['dropout'],
        emb_dropout=config['emb_dropout']
    )
    
    # Load weights
    model.load_state_dict(checkpoint)
    model.eval()
    
    output_path = "vit_weights.bin"
    print(f"Exporting weights to {output_path}...")

    with open(output_path, 'wb') as f:
        # Helper function to write a tensor to the file
        def write_tensor(tensor, name):
            if tensor is not None:
                tensor_np = tensor.detach().numpy().astype(np.float32)
                print(f"  - Writing {name} with shape {tensor_np.shape}")
                f.write(tensor_np.tobytes())
            else:
                print(f"  - Warning: {name} is None")

        print("=== Analyzing model structure ===")
        print("Model parameters:")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.shape}")

        # Order must match ViTWeights struct in vit.h
        
        # 1. Patch Embedding Layers (LayerNorm -> Linear -> LayerNorm)
        print("\n--- Patch Embedding ---")
        write_tensor(model.to_patch_embedding[0].weight, 'patch_ln1_w')
        write_tensor(model.to_patch_embedding[0].bias, 'patch_ln1_b')
        write_tensor(model.to_patch_embedding[1].weight, 'patch_linear_w')
        write_tensor(model.to_patch_embedding[1].bias, 'patch_linear_b')
        write_tensor(model.to_patch_embedding[2].weight, 'patch_ln2_w')
        write_tensor(model.to_patch_embedding[2].bias, 'patch_ln2_b')

        # 2. Position Embedding and CLS token
        print("--- Position Embedding & CLS Token ---")
        write_tensor(model.pos_embedding, 'pos_embed')
        write_tensor(model.cls_token, 'cls_token')

        # 3. Transformer Encoder Blocks
        print("--- Transformer Blocks ---")
        for i in range(config['depth']):
            print(f"  Block {i}:")
            
            layer = model.transformer.layers[i]
            attention = layer[0]
            feedforward = layer[1]
            
            # Attention weights (QKV combined)
            write_tensor(attention.to_qkv.weight, f'qkv_weights_{i}')
            
            # QKV bias (pv_pytorch PlantVIT doesn't use bias for QKV, write zeros)
            # Match C expectation: 3 * NUM_HEADS * HEAD_DIM = 3 * 3 * 32 = 288
            qkv_bias_size = 3 * config['heads'] * config['dim_head']  # 3 * 3 * 32 = 288
            zeros = np.zeros(qkv_bias_size, dtype=np.float32)
            print(f"  - Writing dummy qkv_bias (zeros) with shape {zeros.shape}")
            f.write(zeros.tobytes())
            
            write_tensor(attention.to_out[0].weight, f'proj_weights_{i}')
            write_tensor(attention.to_out[0].bias, f'proj_bias_{i}')
            
            # MLP weights
            write_tensor(feedforward.net[1].weight, f'mlp_fc1_weights_{i}')
            write_tensor(feedforward.net[1].bias, f'mlp_fc1_bias_{i}')
            write_tensor(feedforward.net[4].weight, f'mlp_fc2_weights_{i}')
            write_tensor(feedforward.net[4].bias, f'mlp_fc2_bias_{i}')
            
            # Layer Norms
            write_tensor(attention.norm.weight, f'attention_norm_weights_{i}')
            write_tensor(attention.norm.bias, f'attention_norm_bias_{i}')
            write_tensor(feedforward.net[0].weight, f'mlp_norm_weights_{i}')
            write_tensor(feedforward.net[0].bias, f'mlp_norm_bias_{i}')

        # 4. Final Layer Norm and Classification Head
        print("--- Final Layers ---")
        write_tensor(model.transformer.norm.weight, 'final_norm_weight')
        write_tensor(model.transformer.norm.bias, 'final_norm_bias')
        write_tensor(model.mlp_head.weight, 'head_weights')
        write_tensor(model.mlp_head.bias, 'head_bias')

    print("\nWeight export complete!")
    print(f"Binary file '{output_path}' has been created.")
    print("Note: Using pv_pytorch/vit.py PlantVIT structure")

if __name__ == '__main__':
    export_weights() 