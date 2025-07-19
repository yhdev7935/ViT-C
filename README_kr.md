# ViT-C: 순수 C언어로 구현된 Vision Transformer

![C](https://img.shields.io/badge/c-%2300599C.svg?style=for-the-badge&logo=c&logoColor=white)
![Makefile](https://img.shields.io/badge/Makefile-000000?style=for-the-badge&logo=makefile&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A8?style=for-the-badge&logo=python&logoColor=ffdd54)

## 🚀 프로젝트 소개

**ViT-C**는 Vision Transformer(ViT) 모델을 순수 C언어로 처음부터 완전히 구현한 프로젝트입니다. 임베디드 시스템, AI 가속 보드, 그리고 교육 목적을 위해 설계되었으며, 복잡한 트랜스포머 아키텍처가 외부 의존성 없이 어떻게 구현될 수 있는지를 보여줍니다. 이는 리소스가 제한된 디바이스에서의 배포나 현대 신경망의 내부 작동 원리를 이해하는 데 이상적입니다.

이 구현은 PyTorch의 `timm` 라이브러리 가중치와 호환되어, 사전 학습된 모델을 추론에 직접 사용할 수 있습니다. 기본적인 선형 대수 연산부터 멀티헤드 셀프어텐션 메커니즘까지, 모든 구성 요소가 표준 C 라이브러리만을 사용하여 처음부터 구현되었습니다.

## ✨ 핵심 특징

- **🚫 의존성 없음:** 순수 C99/C11과 표준 라이브러리(`math.h`, `stdlib.h`, `string.h`)만 사용
- **🧩 모듈화 설계:** 각 신경망 계층(Linear, Attention, MLP, LayerNorm)이 독립적이고 재사용 가능한 모듈로 구현
- **🔗 PyTorch 호환성:** `timm`의 `vit_base_patch16_224` 모델 가중치 및 아키텍처와 완전 호환
- **💻 이식성:** 표준 `gcc`와 `Makefile`을 사용하여 모든 주요 플랫폼에서 컴파일 가능
- **📖 상세한 문서화:** 모든 함수와 구조체에 포괄적인 Doxygen 스타일 문서 포함
- **⚡ 메모리 효율성:** 임베디드 배포에 적합한 최적화된 메모리 사용 패턴
- **🏗️ 캐시 친화적:** 현대 CPU 캐시 계층에 최적화된 행 우선 행렬 연산

## 🛠️ 빌드 및 실행 방법

### 사전 준비물

다음 도구들이 설치되어 있는지 확인하세요:

- **C 컴파일러:** C99/C11을 지원하는 `gcc` 또는 `clang`
- **빌드 시스템:** `make` 유틸리티
- **Python 환경:** 다음 패키지가 설치된 `python3`
  ```bash
  pip install torch torchvision timm numpy
  ```

### 1단계: 가중치 파일 생성

PyTorch에서 사전 학습된 ViT 가중치를 추출하여 바이너리 형식으로 변환합니다:

```bash
python3 utils/export_weights.py
```

이 명령은 `timm`에서 `vit_base_patch16_224` 모델을 다운로드하고, C 구현에서 예상하는 정확한 순서로 모든 모델 매개변수를 포함하는 `vit_weights.bin` 파일을 생성합니다.

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
--- Pure C Vision Transformer Inference ---
Skipping weight loading for this demonstration.
The model structure is ready, but weights are not loaded.
Running forward pass with dummy data (expecting garbage output without weights)...
Forward pass completed (simulation).
NOTE: The actual vit_forward call is commented out because weights are not loaded.
To make this fully functional, you must implement the full load_weights function.
Example logits (first 10):
  logit[0] = 0.000000
  ...
Project structure is complete. Implement the weight loader to run real inference.
```

### 추가 빌드 명령어

```bash
make run    # 한 번에 컴파일하고 실행
make clean  # 모든 빌드 산출물 제거
```

## 🏗️ 아키텍처 개요

ViT-C 구현은 표준 Vision Transformer 아키텍처를 따릅니다:

1. **패치 임베디드:** 입력 이미지(224×224×3)를 16×16 패치로 나누어 196개의 패치 토큰 생성
2. **위치 임베디드:** 패치 임베디드에 학습 가능한 위치 인코딩 추가
3. **트랜스포머 인코더:** 12개의 동일한 블록, 각각 다음을 포함:
   - **Multi-Head Self-Attention (MHSA):** 64차원 헤드 공간을 가진 12개의 어텐션 헤드
   - **Layer Normalization:** 각 서브레이어 전에 적용 (Pre-LN 아키텍처)
   - **MLP 블록:** GELU 활성화를 가진 피드포워드 네트워크 (768→3072→768)
   - **잔차 연결:** 각 서브레이어 주변의 스킵 연결
4. **분류 헤드:** 최종 레이어 정규화 + 1000개 클래스(ImageNet)로의 선형 투영

### 데이터 흐름

```
입력 이미지 (3×224×224)
    ↓ 패치 임베디드
패치 (196×768) + CLS 토큰 (1×768) + 위치 임베디드
    ↓ 12× 트랜스포머 인코더 블록
    ↓ [LayerNorm → MHSA → Add] → [LayerNorm → MLP → Add]
특징 표현 (197×768)
    ↓ CLS 토큰 추출 + 최종 LayerNorm
    ↓ 분류 헤드
로짓 (1000개 클래스)
```

## 📂 프로젝트 파일 구조 및 설명

프로젝트는 기능에 따라 `include` (헤더 파일), `src` (소스 파일), `utils` (유틸리티) 디렉토리로 구성되어 있습니다.

### `include/` - 헤더 파일

- **`vit_config.h`**: ViT 모델의 모든 핵심 파라미터를 정의하는 설정 파일

  - 이미지 크기 (224×224), 패치 크기 (16×16), 임베디드 차원 (768) 등
  - 다른 ViT 변종으로 쉽게 확장할 수 있도록 설계된 중앙 집중식 설정

- **`linear.h`**: 선형 변환 함수의 인터페이스 정의

  - `y = Wx + b` 연산을 수행하는 `linear` 함수 선언
  - PyTorch의 Linear 레이어와 호환되는 가중치 형식 지원

- **`vit_math.h`**: 모델에 필요한 핵심 수학 함수들의 인터페이스

  - GELU 활성화 함수, Softmax, 벡터 덧셈, 행렬 곱셈, 전치 함수
  - 모든 함수는 수치적 안정성을 고려하여 설계

- **`layernorm.h`**: Layer Normalization 함수의 인터페이스

  - 평균과 분산을 계산하여 정규화하는 `layernorm` 함수 선언
  - 학습 가능한 스케일(γ)과 시프트(β) 매개변수 지원

- **`mlp.h`**: Feed-Forward Network의 구조체와 함수 인터페이스

  - MLP 블록의 가중치를 관리하는 `MLPWeights` 구조체
  - Linear → GELU → Linear 순서로 처리하는 `mlp_forward` 함수

- **`attention.h`**: Multi-Head Self-Attention의 핵심 인터페이스

  - QKV 프로젝션과 출력 프로젝션 가중치를 관리하는 `AttentionWeights` 구조체
  - 전체 MHSA 로직을 처리하는 `attention_forward` 함수

- **`vit.h`**: 전체 ViT 모델의 최상위 인터페이스
  - 개별 인코더 블록을 정의하는 `EncoderBlock` 구조체
  - 전체 모델 가중치를 관리하는 `ViTWeights` 구조체
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

  - QKV 생성을 위한 단일 선형 변환 후 분리
  - 12개 헤드의 병렬적 어텐션 스코어 계산
  - 스케일링 팩터 (`1/sqrt(head_dim)`) 적용
  - 행별 Softmax 정규화
  - 어텐션 가중 값 계산 및 헤드 연결
  - 최종 출력 프로젝션

- **`vit.c`**: 전체 모델의 조립 및 오케스트레이션

  - 패치 임베디드 처리 (현재 플레이스홀더)
  - CLS 토큰 및 위치 임베디드 추가
  - 12개 인코더 블록의 순차적 처리
  - 헬퍼 함수 `encoder_block_forward`를 통한 각 블록 실행
  - 최종 정규화 및 분류 헤드 적용

- **`main.c`**: 프로그램의 진입점
  - 가중치 로딩 함수 (현재 플레이스홀더)
  - 더미 입력 데이터 생성 및 메모리 할당
  - 추론 실행 및 결과 출력
  - 깔끔한 메모리 정리

### `utils/` - 유틸리티

- **`export_weights.py`**: PyTorch 가중치 변환 스크립트
  - `timm` 라이브러리에서 사전 학습된 ViT 모델 로딩
  - 가중치를 C 구조체 순서에 맞춰 바이너리 파일로 저장
  - QKV 가중치의 결합 및 적절한 텐서 형태 변환
  - IEEE 754 단정밀도 부동소수점 형식으로 출력

### 루트 디렉토리

- **`Makefile`**: 프로젝트 빌드 시스템

  - 자동 소스 파일 탐지 (`wildcard` 함수 사용)
  - 최적화 플래그 (`-O2`) 및 경고 플래그 (`-Wall -Wextra`)
  - 수학 라이브러리 링크 (`-lm`)
  - `all`, `run`, `clean` 타겟 제공

- **`run_tests.sh`**: 자동화된 무결성 테스트

  - Python 가중치 추출 스크립트 테스트
  - C 프로젝트 컴파일 테스트
  - 실행 파일 크래시 테스트
  - 의존성 체크 및 적절한 폴백 처리

- **`README.md`**: 영문 프로젝트 문서
- **`README_kr.md`**: 한국어 프로젝트 문서 (이 파일)

## 🔧 구현 세부사항

### 메모리 관리

- **정적 할당 선호:** 대부분의 버퍼가 임베디드 호환성을 위해 스택 할당 사용
- **최소 동적 할당:** 어텐션 모듈만 `malloc` 사용 (향후 최적화 대상으로 표시)
- **사전 할당 버퍼:** `ViTModel` 구조체에 추론 중 할당을 피하기 위한 중간 버퍼 포함

### 수치적 안정성

- **Softmax:** 오버플로우 방지를 위한 최댓값 차감 기법
- **Layer Normalization:** 0으로 나누기 방지를 위한 epsilon 매개변수 (1e-6)
- **GELU 활성화:** 수학적으로 정확한 근사를 위한 `erff()` 사용

### 최적화 특징

- **캐시 친화적 접근:** 순차적 메모리 접근을 가진 행 우선 행렬 연산
- **컴파일러 최적화:** 벡터화 및 루프 최적화를 가능하게 하는 `-O2` 플래그
- **모듈러 설계:** 각 구성 요소를 개별적으로 최적화하거나 교체 가능

## 🚧 현재 제한사항 및 향후 개발 계획

### 해야 할 일

1. **가중치 로딩 완성:**

   - `src/main.c`의 `load_weights()` 함수 완전 구현
   - 각 가중치 텐서에 대한 적절한 메모리 할당 추가
   - 바이너리 파일 읽기가 내보내기 순서와 일치하는지 확인

2. **이미지 전처리:**

   - 원시 이미지에서 실제 패치 추출 구현
   - 이미지 정규화 (ImageNet 평균/표준편차) 추가
   - 다양한 입력 형식 (RGB, BGR 등) 지원

3. **성능 최적화:**

   - **SIMD 명령어:** ARM NEON 또는 x86 SSE를 사용한 행렬 연산 벡터화
   - **멀티스레딩:** OpenMP를 사용한 어텐션 헤드 병렬화
   - **양자화:** 더 빠른 임베디드 배포를 위한 INT8 추론 지원
   - **메모리 풀:** `malloc`을 사전 할당된 메모리 풀로 교체

4. **확장된 모델 지원:**
   - ViT-Large (24블록, 1024차원)
   - ViT-Huge (32블록, 1280차원)
   - 증류 토큰을 가진 DeiT 변종

### 알려진 문제

- `load_weights()` 함수가 현재 플레이스홀더 상태
- 패치 추출 로직이 견고한 구현을 필요로 함
- 플레이스홀더 코드의 일부 컴파일러 경고

## 🤝 기여하기

기여를 환영합니다! 다음과 같이 참여해 주세요:

- 누락된 기능 구현 (가중치 로딩, 패치 추출)
- 성능 최적화 추가
- 다른 ViT 변종 지원 확장
- 문서 및 예제 개선
- 포괄적인 테스트 스위트 추가

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 라이선스됩니다. 자세한 내용은 LICENSE 파일을 참조하세요.

## 🙏 감사의 말

- **Attention Is All You Need:** Vaswani et al. (2017) - 원본 Transformer 논문
- **An Image is Worth 16x16 Words:** Dosovitskiy et al. (2020) - Vision Transformer 논문
- **timm 라이브러리:** Ross Wightman의 우수한 PyTorch Image Models 라이브러리
- **임베디드 AI 커뮤니티:** 배포 중심 ML 구현에 영감을 준 커뮤니티

---

**참고:** 이 구현은 원시 성능보다 교육적 가치와 임베디드 배포를 우선시합니다. 대규모 추론이 필요한 프로덕션 사용을 위해서는 PyTorch 또는 TensorRT 구현을 고려하세요.
