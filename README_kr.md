# PlantVIT-C: 순수 C언어로 구현된 토마토 질병 분류 Vision Transformer

![C](https://img.shields.io/badge/c-%2300599C.svg?style=for-the-badge&logo=c&logoColor=white)
![Makefile](https://img.shields.io/badge/Makefile-000000?style=for-the-badge&logo=makefile&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A8?style=for-the-badge&logo=python&logoColor=ffdd54)

## 🚀 프로젝트 소개

**PlantVIT-C**는 토마토 질병 분류를 위해 특별히 설계된 경량 Vision Transformer인 **PlantVIT** 모델을 순수 C언어로 처음부터 완전히 구현한 프로젝트입니다. 농업 IoT 디바이스, 임베디드 시스템, 그리고 스마트 팜 환경에서의 실시간 식물 질병 진단을 목적으로 설계되었으며, 복잡한 트랜스포머 아키텍처가 외부 의존성 없이 어떻게 구현될 수 있는지를 보여줍니다.

이 구현은 토마토의 9가지 질병 상태를 분류할 수 있으며, 표준 ViT보다 훨씬 가볍고 효율적인 아키텍처를 사용합니다. 기본적인 선형 대수 연산부터 멀티헤드 셀프어텐션 메커니즘까지, 모든 구성 요소가 표준 C 라이브러리만을 사용하여 처음부터 구현되었습니다.

## ✨ 핵심 특징

- **🚫 의존성 없음:** 순수 C99/C11과 표준 라이브러리(`math.h`, `stdlib.h`, `string.h`)만 사용
- **🧩 모듈화 설계:** 각 신경망 계층(Linear, Attention, MLP, LayerNorm)이 독립적이고 재사용 가능한 모듈로 구현
- **🌱 농업 특화:** 토마토 질병 분류에 최적화된 경량 PlantVIT 아키텍처
- **💻 이식성:** 표준 `gcc`와 `Makefile`을 사용하여 모든 주요 플랫폼에서 컴파일 가능
- **📖 상세한 문서화:** 모든 함수와 구조체에 포괄적인 Doxygen 스타일 문서 포함
- **⚡ 메모리 효율성:** 임베디드 배포에 적합한 최적화된 메모리 사용 패턴 (32차원 임베딩)
- **🏗️ 캐시 친화적:** 현대 CPU 캐시 계층에 최적화된 행 우선 행렬 연산

## 🌿 PlantVIT 모델 사양

PlantVIT는 농업 환경에서의 실시간 진단을 위해 설계된 경량 Vision Transformer입니다:

- **이미지 크기:** 256×256×3
- **패치 크기:** 32×32 (총 64개 패치)
- **임베딩 차원:** 32 (표준 ViT의 768에서 대폭 축소)
- **인코더 블록 수:** 3개 (표준 ViT의 12개에서 축소)
- **어텐션 헤드:** 3개
- **헤드 차원:** 32
- **MLP 차원:** 16
- **분류 클래스:** 9개 토마토 질병 상태

### 토마토 질병 분류 클래스

1. **Bacterial_spot** (세균성 반점병)
2. **Early_blight** (조기 마름병)
3. **Late_blight** (후기 마름병)
4. **Leaf_mold** (잎곰팡이병)
5. **Septoria_leaf_spot** (셉토리아 잎반점병)
6. **Spider_mites** (응애)
7. **Target_spot** (표적반점병)
8. **Yellow_leaf_curl_virus** (황화잎말림바이러스)
9. **Healthy** (건강함)

## 🛠️ 빌드 및 실행 방법

### 사전 준비물

다음 도구들이 설치되어 있는지 확인하세요:

- **C 컴파일러:** C99/C11을 지원하는 `gcc` 또는 `clang`
- **빌드 시스템:** `make` 유틸리티
- **Python 환경:** 다음 패키지가 설치된 `python3`
  ```bash
  pip install torch torchvision einops numpy
  ```

### 1단계: 가중치 파일 생성

PlantVIT 모델에서 가중치를 추출하여 바이너리 형식으로 변환합니다:

```bash
python3 utils/export_weights.py
```

이 명령은 PlantVIT 모델 구조를 생성하고 (학습된 가중치가 있다면 `plantvit_tomato.pth`에서 로드), C 구현에서 예상하는 정확한 순서로 모든 모델 매개변수를 포함하는 `vit_weights.bin` 파일을 생성합니다.

### 2단계: 프로젝트 컴파일

제공된 Makefile을 사용하여 C 프로젝트를 빌드합니다:

```bash
make all
```

이 명령은 최적화 플래그(`-O2`)와 함께 모든 소스 파일을 컴파일하고 `vit_inference` 실행 파일을 생성합니다.

### 3단계: 추론 데모 실행

컴파일된 프로그램을 실행합니다:

```bash
./vit_inference
```

**예상 출력:**

```
--- PlantVIT Pure C Inference ---
Image Size: 256x256, Patch Size: 32x32, Classes: 9
Embed Dim: 32, Blocks: 3, Heads: 3, MLP Dim: 16
Allocating model buffers...
Model buffers allocated successfully!
Loading PlantVIT weights from vit_weights.bin...
--- Loading Patch Embedding ---
  - Loaded patch_ln1_w: 3072 elements
  - Loaded patch_ln1_b: 3072 elements
  ...
Weight loading completed successfully!

Running PlantVIT forward pass...
Forward pass completed successfully!

Tomato Disease Classification Results:
  Bacterial_spot: 0.106876
  Early_blight: -0.151543
  Late_blight: -0.025076
  Leaf_mold: 0.136157
  Septoria_leaf_spot: 0.111755
  Spider_mites: -0.091206
  Target_spot: -0.002325
  Yellow_leaf_curl_virus: 0.002382
  Healthy: 0.013054

Predicted Disease: Leaf_mold (confidence: 0.136157)

PlantVIT inference completed successfully!
```

### 추가 빌드 명령어

```bash
make run    # 한 번에 컴파일하고 실행
make clean  # 모든 빌드 산출물 제거
```

## 🏗️ 아키텍처 개요

PlantVIT-C 구현은 경량화된 Vision Transformer 아키텍처를 따릅니다:

1. **패치 임베딩:** 입력 이미지(256×256×3)를 32×32 패치로 나누어 64개의 패치 토큰 생성
   - **특별한 구조:** LayerNorm → Linear → LayerNorm (표준 ViT와 다름)
2. **위치 임베디드:** 패치 임베디드에 학습 가능한 위치 인코딩 추가
3. **트랜스포머 인코더:** 3개의 동일한 블록, 각각 다음을 포함:
   - **Multi-Head Self-Attention (MHSA):** 32차원 헤드 공간을 가진 3개의 어텐션 헤드
   - **Layer Normalization:** 각 서브레이어 전에 적용 (Pre-LN 아키텍처)
   - **MLP 블록:** GELU 활성화를 가진 피드포워드 네트워크 (32→16→32)
   - **잔차 연결:** 각 서브레이어 주변의 스킵 연결
4. **분류 헤드:** 최종 레이어 정규화 + 9개 클래스(토마토 질병)로의 선형 투영

### 데이터 흐름

```
입력 이미지 (3×256×256)
    ↓ 패치 임베디드 (LayerNorm → Linear → LayerNorm)
패치 (64×32) + CLS 토큰 (1×32) + 위치 임베디드
    ↓ 3× 트랜스포머 인코더 블록
    ↓ [LayerNorm → MHSA → Add] → [LayerNorm → MLP → Add]
특징 표현 (65×32)
    ↓ CLS 토큰 추출 + 최종 LayerNorm
    ↓ 분류 헤드
로짓 (9개 토마토 질병 클래스)
```

## 📂 프로젝트 파일 구조 및 설명

프로젝트는 기능에 따라 `include` (헤더 파일), `src` (소스 파일), `utils` (유틸리티) 디렉토리로 구성되어 있습니다.

### `include/` - 헤더 파일

- **`vit_config.h`**: PlantVIT 모델의 모든 핵심 파라미터를 정의하는 설정 파일

  - 이미지 크기 (256×256), 패치 크기 (32×32), 임베딩 차원 (32) 등
  - 농업 환경에 최적화된 경량 아키텍처 설정

- **`linear.h`**: 선형 변환 함수의 인터페이스 정의

  - `y = Wx + b` 연산을 수행하는 `linear` 함수 선언
  - PlantVIT의 패치 임베딩 레이어들과 호환되는 가중치 형식 지원

- **`vit_math.h`**: 모델에 필요한 핵심 수학 함수들의 인터페이스

  - GELU 활성화 함수, Softmax, 벡터 덧셈, 행렬 곱셈, 전치 함수
  - 모든 함수는 수치적 안정성을 고려하여 설계

- **`layernorm.h`**: Layer Normalization 함수의 인터페이스

  - 평균과 분산을 계산하여 정규화하는 `layernorm` 함수 선언
  - PlantVIT의 패치 임베디드에서 사용되는 다중 LayerNorm 지원

- **`mlp.h`**: Feed-Forward Network의 구조체와 함수 인터페이스

  - MLP 블록의 가중치를 관리하는 `MLPWeights` 구조체
  - Linear → GELU → Linear 순서로 처리하는 `mlp_forward` 함수

- **`attention.h`**: Multi-Head Self-Attention의 핵심 인터페이스

  - QKV 프로젝션과 출력 프로젝션 가중치를 관리하는 `AttentionWeights` 구조체
  - PlantVIT의 통합 QKV 생성을 지원하는 `attention_forward` 함수

- **`vit.h`**: 전체 PlantVIT 모델의 최상위 인터페이스
  - 개별 인코더 블록을 정의하는 `EncoderBlock` 구조체
  - PlantVIT의 패치 임베디드 구조를 반영한 `ViTWeights` 구조체
  - 가중치와 중간 버퍼를 포함하는 `ViTModel` 구조체
  - 전체 추론을 수행하는 `vit_forward` 함수

### `src/` - 소스 파일

- **`linear.c`**: 선형 변환의 실제 구현

  - 효율적인 행렬-벡터 곱셈 알고리즘
  - 편향(bias) 항의 선택적 처리 (NULL 체크)
  - 캐시 친화적인 행 우선 접근 패턴

- **`vit_math.c`**: 모든 수학 함수들의 구현체

  - **GELU**: `erf` 함수를 사용한 정확한 구현
  - **Softmax**: 오버플로우 방지를 위한 최댓값 정규화 기법
  - **행렬 곱셈**: 표준 3중 루프 알고리즘
  - **전치**: 효율적인 인덱스 교환

- **`layernorm.c`**: Layer Normalization의 구현

  - 3단계 처리: 평균 계산 → 분산 계산 → 정규화 및 변환
  - 수치적 안정성을 위한 epsilon 매개변수 (1e-6)
  - `sqrtf` 함수를 사용한 단정밀도 최적화

- **`mlp.c`**: MLP 블록의 순방향 전파 구현

  - 두 개의 선형 레이어를 GELU 활성화로 연결
  - 중간 버퍼를 통한 메모리 효율적 처리
  - 모듈 간 인터페이스의 깔끔한 활용

- **`attention.c`**: 프로젝트에서 가장 복잡한 구현

  - 통합 QKV 생성을 위한 단일 선형 변환 후 분리
  - 3개 헤드의 병렬적 어텐션 스코어 계산
  - 스케일링 팩터 (`1/sqrt(head_dim)`) 적용
  - 행별 Softmax 정규화
  - 어텐션 가중 값 계산 및 헤드 연결
  - 최종 출력 프로젝션

- **`vit.c`**: 전체 PlantVIT 모델의 조립 및 오케스트레이션

  - **완전 구현된 패치 임베디드:** LayerNorm → Linear → LayerNorm 구조
  - CLS 토큰 및 위치 임베디드 추가
  - 3개 인코더 블록의 순차적 처리
  - 헬퍼 함수 `encoder_block_forward`를 통한 각 블록 실행
  - 최종 정규화 및 토마토 질병 분류 헤드 적용

- **`main.c`**: 프로그램의 진입점
  - **완전 구현된 가중치 로딩:** PlantVIT 구조에 맞춘 메모리 할당 및 바이너리 파일 읽기
  - 모델 버퍼 할당 및 관리
  - 토마토 질병 분류 결과 출력 및 해석
  - 완전한 메모리 정리 및 에러 처리

### `utils/` - 유틸리티

- **`export_weights.py`**: PlantVIT 가중치 변환 스크립트
  - PlantVIT 모델 클래스 정의 (Attention, FeedForward, Transformer, PlantVIT)
  - 학습된 모델 가중치 로딩 (`plantvit_tomato.pth`)
  - 가중치를 C 구조체 순서에 맞춰 바이너리 파일로 저장
  - PlantVIT의 통합 QKV 가중치 및 패치 임베디드 구조 처리
  - IEEE 754 단정밀도 부동소수점 형식으로 출력

## 🔧 구현 세부사항

### 메모리 관리

- **정적 할당 선호:** 대부분의 버퍼가 임베디드 호환성을 위해 스택 할당 사용
- **완전한 동적 할당:** 모든 가중치와 버퍼에 대한 완전한 메모리 관리 구현
- **사전 할당 버퍼:** `ViTModel` 구조체에 추론 중 할당을 피하기 위한 중간 버퍼 포함

### 수치적 안정성

- **Softmax:** 오버플로우 방지를 위한 최댓값 차감 기법
- **Layer Normalization:** 0으로 나누기 방지를 위한 epsilon 매개변수 (1e-6)
- **GELU 활성화:** 수학적으로 정확한 근사를 위한 `erff()` 사용

### 최적화 특징

- **캐시 친화적 접근:** 순차적 메모리 접근을 가진 행 우선 행렬 연산
- **컴파일러 최적화:** 벡터화 및 루프 최적화를 가능하게 하는 `-O2` 플래그
- **모듈러 설계:** 각 구성 요소를 개별적으로 최적화하거나 교체 가능
- **경량 아키텍처:** 32차원 임베딩으로 메모리 사용량 대폭 감소

## 🚧 향후 개발 계획

### 성능 최적화

1. **SIMD 명령어:** ARM NEON 또는 x86 SSE를 사용한 행렬 연산 벡터화
2. **멀티스레딩:** OpenMP를 사용한 어텐션 헤드 병렬화
3. **양자화:** 더 빠른 임베디드 배포를 위한 INT8 추론 지원
4. **메모리 풀:** `malloc`을 사전 할당된 메모리 풀로 교체

### 농업 특화 기능

1. **실시간 이미지 처리:** 카메라 입력에서 직접 추론
2. **다양한 작물 지원:** 다른 식물 질병 분류 모델 추가
3. **환경 데이터 통합:** 온도, 습도 등 환경 데이터와 결합
4. **IoT 연동:** MQTT, LoRa 등 IoT 프로토콜 지원

### 모델 확장

1. **더 많은 질병 클래스:** 추가 토마토 질병 및 해충 분류
2. **다중 작물 모델:** 토마토 외 다른 작물 질병 진단
3. **심각도 평가:** 질병의 진행 단계 및 심각도 평가

## 🤝 기여하기

기여를 환영합니다! 다음과 같이 참여해 주세요:

- 성능 최적화 추가
- 새로운 농업 특화 기능 개발
- 다른 작물 지원 확장
- 문서 및 예제 개선
- 포괄적인 테스트 스위트 추가

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 라이선스됩니다. 자세한 내용은 LICENSE 파일을 참조하세요.

## 🙏 감사의 말

- **Attention Is All You Need:** Vaswani et al. (2017) - 원본 Transformer 논문
- **An Image is Worth 16x16 Words:** Dosovitskiy et al. (2020) - Vision Transformer 논문
- **농업 AI 커뮤니티:** 스마트 팜 기술 발전에 기여하는 커뮤니티
- **임베디드 AI 커뮤니티:** 배포 중심 ML 구현에 영감을 준 커뮤니티

---

**참고:** 이 구현은 농업 환경에서의 실용성과 교육적 가치를 우선시합니다. PlantVIT의 경량 아키텍처는 임베디드 시스템과 IoT 디바이스에서의 실시간 질병 진단을 위해 특별히 설계되었습니다.
