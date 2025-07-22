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

def test_python_inference():
    """
    Python으로 PlantVIT 추론 테스트
    """
    print("=== Python PlantVIT Inference Test ===")
    
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
            
        print(f"\n{'='*50}")
        print(f"Testing: {image_path}")
        print(f"{'='*50}")
        
        # 이미지 로드 및 전처리
        image_tensor = load_and_preprocess_image_python(image_path)
        if image_tensor is None:
            continue
        
        # 추론 실행
        with torch.no_grad():
            logits = model(image_tensor)
            logits = logits.squeeze(0)  # 배치 차원 제거
        
        # 결과 출력
        print("\n=== Python Inference Results ===")
        for i, disease in enumerate(disease_names):
            print(f"  {disease}: {logits[i].item():.6f}")
        
        # 예측 클래스
        predicted_class = torch.argmax(logits).item()
        confidence = logits[predicted_class].item()
        
        print(f"\n🌱 PYTHON DIAGNOSIS RESULT 🌱")
        print(f"Predicted Disease: {disease_names[predicted_class]}")
        print(f"Confidence Score: {confidence:.6f}")
        
        if predicted_class == 2:  # Healthy
            print("✅ Good news! The tomato plant appears to be healthy.")
        else:
            print("⚠️  Disease detected! Consider appropriate treatment measures.")

if __name__ == '__main__':
    test_python_inference() 