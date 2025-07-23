#!/usr/bin/env python3
import sys
import os
sys.path.append('pv_pytorch')

import torch
import numpy as np
from PIL import Image
from vit import PlantVIT

def load_and_preprocess_image_python(image_path):
    """
    Python에서 C와 동일한 전처리를 수행
    """
    print(f"Loading image: {image_path}")
    
    try:
        # PIL로 이미지 로드 (RGB 형태)
        image = Image.open(image_path).convert('RGB')
        print(f"Original image: {image.size[0]}x{image.size[1]} with 3 channels")
        
        # 256x256으로 리사이즈
        image_resized = image.resize((256, 256), Image.BILINEAR)
        print(f"Resized to: 256x256")
        
        # PIL Image -> numpy array (RGB -> BGR 변환)
        image_array = np.array(image_resized, dtype=np.float32)
        
        # RGB -> BGR 변환 (C 코드와 일치시키기 위해)
        image_bgr = image_array[:, :, [2, 1, 0]]  # RGB -> BGR
        
        # [0,255] -> [0,1] 정규화
        image_normalized = image_bgr / 255.0
        
        # HWC -> CHW 변환
        image_chw = np.transpose(image_normalized, (2, 0, 1))  # HWC -> CHW
        
        # Tensor로 변환하고 배치 차원 추가
        image_tensor = torch.from_numpy(image_chw).unsqueeze(0)  # (1, C, H, W)
        
        print(f"Final tensor shape: {image_tensor.shape}")
        return image_tensor
        
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def debug_to_patches_detailed(model, image_tensor):
    """
    Python to_patches 함수의 상세한 동작 분석
    """
    print(f"\n=== 🔍 DETAILED PATCH EXTRACTION ANALYSIS ===")
    
    n, c, h, w = image_tensor.shape
    print(f"Original image shape: {image_tensor.shape} (n={n}, c={c}, h={h}, w={w})")
    
    # Step 1: First transpose (2, 3) - swap h and w
    transposed1 = torch.transpose(image_tensor, 2, 3)  # (n, c, w, h)
    print(f"After first transpose (2,3): {transposed1.shape}")
    
    # Step 2: Second transpose (1, 3) - swap c and h
    transposed2 = torch.transpose(transposed1, 1, 3)  # (n, h, w, c)
    print(f"After second transpose (1,3): {transposed2.shape}")
    
    # Check first few pixels in transposed2
    print(f"transposed2[0, 0, 0, :] (first pixel all channels): {transposed2[0, 0, 0, :].tolist()}")
    print(f"transposed2[0, 0, 1, :] (second pixel all channels): {transposed2[0, 0, 1, :].tolist()}")
    
    # Step 3: Reshape
    patches = torch.reshape(transposed2, (n, h*w // model.patch_area, model.patch_area * c))
    print(f"After reshape: {patches.shape}")
    print(f"patch_area = {model.patch_area}")
    print(f"h*w // patch_area = {h*w // model.patch_area}")
    print(f"patch_area * c = {model.patch_area * c}")
    
    # Analyze patch order
    print(f"\n--- PATCH ORDER ANALYSIS ---")
    patch_height, patch_width = model.patch_shape
    patches_per_row = h // patch_height
    patches_per_col = w // patch_width
    print(f"patch_height={patch_height}, patch_width={patch_width}")
    print(f"patches_per_row={patches_per_row}, patches_per_col={patches_per_col}")
    
    # Check how the first patch is formed
    print(f"\n--- FIRST PATCH FORMATION ---")
    first_patch_2d = transposed2[0, :patch_height, :patch_width, :]  # (32, 32, 3)
    first_patch_flattened = first_patch_2d.reshape(-1)  # (3072,)
    
    print(f"First patch from 2D extraction shape: {first_patch_2d.shape}")
    print(f"First patch flattened shape: {first_patch_flattened.shape}")
    print(f"First 10 values from manual extraction: {first_patch_flattened[:10].tolist()}")
    print(f"First 10 values from patches[0,0]: {patches[0, 0, :10].tolist()}")
    print(f"Are they equal? {torch.allclose(first_patch_flattened, patches[0, 0])}")
    
    # Check patch statistics
    print(f"\n--- PATCH STATISTICS ---")
    for i in range(min(5, patches.shape[1])):
        patch_mean = patches[0, i].mean().item()
        patch_std = patches[0, i].std().item()
        print(f"Patch {i}: mean={patch_mean:.6f}, std={patch_std:.6f}")
    
    return patches

def debug_step_by_step_inference(model, image_tensor, image_name):
    """
    단계별 디버깅을 위한 PlantVIT 추론
    """
    print(f"\n=== 🐍 PYTHON STEP-BY-STEP DEBUG: {image_name} ===")
    
    with torch.no_grad():
        # === STEP 1: 패치 추출 (상세 분석) ===
        print("\n--- STEP 1: Patch Extraction ---")
        patches = debug_to_patches_detailed(model, image_tensor)
        print(f"Patches shape: {patches.shape}")
        print(f"First patch first 10 values: {patches[0, 0, :10].tolist()}")
        print(f"First patch stats: mean={patches[0, 0].mean():.6f}, std={patches[0, 0].std():.6f}")
        
        # === STEP 2: 패치 임베딩 ===
        print("\n--- STEP 2: Patch Embedding ---")
        x = model.to_patch_embedding(patches)
        b, n, _ = x.shape
        print(f"Patch embedding shape: {x.shape}")
        print(f"First token first 10 values: {x[0, 0, :10].tolist()}")
        print(f"First token stats: mean={x[0, 0].mean():.6f}, std={x[0, 0].std():.6f}")
        
        # === STEP 3: CLS 토큰 추가 ===
        print("\n--- STEP 3: Add CLS Token ---")
        cls_tokens = torch.repeat_interleave(model.cls_token, b, dim=0)
        x = torch.cat((cls_tokens, x), dim=1)
        print(f"After CLS token shape: {x.shape}")
        print(f"CLS token first 10 values: {x[0, 0, :10].tolist()}")
        print(f"CLS token stats: mean={x[0, 0].mean():.6f}, std={x[0, 0].std():.6f}")
        
        # === STEP 4: Position Embedding 추가 ===
        print("\n--- STEP 4: Add Position Embedding ---")
        x += model.pos_embedding[:, :(n + 1)]
        x = model.dropout(x)
        print(f"After position embedding shape: {x.shape}")
        print(f"CLS token after pos_embed first 10 values: {x[0, 0, :10].tolist()}")
        print(f"CLS token after pos_embed stats: mean={x[0, 0].mean():.6f}, std={x[0, 0].std():.6f}")
        print(f"First patch token after pos_embed first 10 values: {x[0, 1, :10].tolist()}")
        
        # === STEP 5: Transformer 블록들 ===
        print(f"\n--- STEP 5: Transformer Blocks ---")
        for i, (attention, ff) in enumerate(model.transformer.layers):
            print(f"\n  Block {i}:")
            
            # Attention
            x_before_attn = x.clone()
            attn_out = attention(x)
            x = attn_out + x
            print(f"    After attention: CLS first 10 = {x[0, 0, :10].tolist()}")
            print(f"    After attention: CLS stats = mean={x[0, 0].mean():.6f}, std={x[0, 0].std():.6f}")
            
            # FeedForward
            x_before_ff = x.clone()
            ff_out = ff(x)
            x = ff_out + x
            print(f"    After feedforward: CLS first 10 = {x[0, 0, :10].tolist()}")
            print(f"    After feedforward: CLS stats = mean={x[0, 0].mean():.6f}, std={x[0, 0].std():.6f}")
        
        # === STEP 6: Final Layer Norm ===
        print(f"\n--- STEP 6: Final Layer Norm ---")
        x = model.transformer.norm(x)
        print(f"After final norm: CLS first 10 = {x[0, 0, :10].tolist()}")
        print(f"After final norm: CLS stats = mean={x[0, 0].mean():.6f}, std={x[0, 0].std():.6f}")
        
        # === STEP 7: 최종 분류 헤드 ===
        print(f"\n--- STEP 7: Classification Head ---")
        cls_output = x[:, 0]  # CLS token만 사용
        logits = model.mlp_head(cls_output)
        logits = logits.squeeze(0)  # 배치 차원 제거
        
        print(f"Final logits: {logits.tolist()}")
        print(f"Logits stats: mean={logits.mean():.6f}, std={logits.std():.6f}")
        
        return logits

def test_python_inference():
    """
    Python으로 PlantVIT 추론 테스트 (디버깅 모드)
    """
    print("=== Python PlantVIT Inference Test (DEBUG MODE) ===")
    
    # PlantVIT 모델 설정 (C와 동일)
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
    
    # 모델 생성
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
    
    # 학습된 가중치 로드
    try:
        print("Loading trained weights from 'plantvit_tomato.pth'...")
        checkpoint = torch.load('plantvit_tomato.pth', map_location='cpu')
        model.load_state_dict(checkpoint)
        print("Weights loaded successfully!")
    except FileNotFoundError:
        print("Error: 'plantvit_tomato.pth' not found.")
        return
    
    model.eval()
    
    # Monkey patch to add detailed attention debugging
    def debug_attention_forward(self, x):
        print(f"        === PYTHON ATTENTION DEBUG ===")
        print(f"        Input CLS first 10: {x[0, 0, :10].tolist()}")
        
        # Apply normalization
        normed = self.norm(x)
        print(f"        After LayerNorm CLS first 10: {normed[0, 0, :10].tolist()}")
        
        # Get Q, K, V
        qkv = self.to_qkv(normed)
        print(f"        QKV CLS first 10: {qkv[0, 0, :10].tolist()}")
        
        # Split into Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)
        print(f"        Q CLS first 10: {q[0, 0, :10].tolist()}")
        print(f"        K CLS first 10: {k[0, 0, :10].tolist()}")
        print(f"        V CLS first 10: {v[0, 0, :10].tolist()}")
        
        # Reshape for multi-head attention
        b, n, _ = q.shape
        dim_head = q.shape[-1] // self.heads  # Calculate dim_head
        q = q.view(b, n, self.heads, dim_head).transpose(1, 2)
        k = k.view(b, n, self.heads, dim_head).transpose(1, 2)
        v = v.view(b, n, self.heads, dim_head).transpose(1, 2)
        
        print(f"        Q shape after head split: {q.shape}")
        print(f"        Q head 0 CLS first 10: {q[0, 0, 0, :10].tolist()}")
        print(f"        Q head 1 CLS first 10: {q[0, 1, 0, :10].tolist()}")
        print(f"        Q head 2 CLS first 10: {q[0, 2, 0, :10].tolist()}")
        
        # Compute attention scores
        scale = self.scale
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        print(f"        Attention scores shape: {scores.shape}")
        print(f"        Scale factor: {scale}")
        print(f"        First attention score [CLS,CLS] for head 0: {scores[0, 0, 0, 0].item()}")
        
        # Apply softmax
        attn = torch.softmax(scores, dim=-1)
        print(f"        After softmax [CLS,CLS] for head 0: {attn[0, 0, 0, 0].item()}")
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        print(f"        After attention*V head 0 CLS first 10: {out[0, 0, 0, :10].tolist()}")
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        print(f"        Multi-head concat CLS first 10: {out[0, 0, :10].tolist()}")
        
        # Apply output projection
        out = self.to_out[0](out)
        print(f"        Final projection CLS first 10: {out[0, 0, :10].tolist()}")
        
        return out

    # Apply the monkey patch
    import types
    original_forward = None

    for i, layer in enumerate(model.transformer.layers):
        attention_layer = layer[0]
        if i == 0:  # Only for the first block for now
            attention_layer.forward = types.MethodType(debug_attention_forward, attention_layer)

    # 질병 이름 (C와 동일한 순서)
    disease_names = [
        "Bacterial_spot",           # 0
        "Early_blight",             # 1  
        "Healthy",                  # 2
        "Late_blight",              # 3
        "Leaf_mold",                # 4
        "Septoria_leaf_spot",       # 5
        "Spider_mites",             # 6
        "Target_spot",              # 7
        "Tomato_mosaic_virus",      # 8
        "Yellow_leaf_curl_virus"    # 9
    ]
    
    # 테스트할 이미지들
    test_images = ["test_tomato.jpg", "test_tomato_2.jpg"]
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found, skipping...")
            continue
            
        print(f"\n{'='*70}")
        print(f"Testing: {image_path}")
        print(f"{'='*70}")
        
        # 이미지 로드 및 전처리
        image_tensor = load_and_preprocess_image_python(image_path)
        if image_tensor is None:
            continue
        
        # 단계별 디버깅 추론 실행
        logits = debug_step_by_step_inference(model, image_tensor, image_path)
        
        # 결과 출력
        print(f"\n=== Python Final Results ===")
        for i, disease in enumerate(disease_names):
            print(f"  {disease}: {logits[i].item():.6f}")
        
        # 예측 클래스
        predicted_class = torch.argmax(logits).item()
        confidence = logits[predicted_class].item()
        
        print(f"\n🌱 PYTHON DIAGNOSIS RESULT 🌱")
        print(f"Predicted Disease: {disease_names[predicted_class]}")
        print(f"Confidence Score: {confidence:.6f}")

if __name__ == '__main__':
    test_python_inference() 