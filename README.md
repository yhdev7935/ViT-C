# PlantVIT-C: A Pure C Implementation of Tomato Disease Classification Vision Transformer

![C](https://img.shields.io/badge/c-%2300599C.svg?style=for-the-badge&logo=c&logoColor=white)
![Makefile](https://img.shields.io/badge/Makefile-000000?style=for-the-badge&logo=makefile&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A8?style=for-the-badge&logo=python&logoColor=ffdd54)

## Overview

**PlantVIT-C** is a complete, from-scratch implementation of **PlantVIT**, a lightweight Vision Transformer specifically designed for tomato disease classification, in pure C. This project is tailored for agricultural IoT devices, embedded systems, and smart farming environments where real-time plant disease diagnosis is crucial. It demonstrates how complex transformer architectures can be implemented without external dependencies, making it ideal for deployment on resource-constrained agricultural devices.

This implementation can classify 9 different tomato disease states using a much lighter and more efficient architecture compared to standard ViT models. Every component, from basic linear algebra operations to multi-head self-attention mechanisms, has been implemented from the ground up using only standard C libraries.

## Key Features

- **🚀 Zero Dependencies:** Uses only pure C99/C11 and standard libraries (`math.h`, `stdlib.h`, `string.h`)
- **🔧 Modular Design:** Each neural network layer (Linear, Attention, MLP, LayerNorm) is implemented as an independent, reusable module
- **🌱 Agriculture-Focused:** Lightweight PlantVIT architecture optimized for tomato disease classification
- **💻 Portable:** Compiles on all major platforms using standard `gcc` and `Makefile`
- **📖 Well-Documented:** Every function and structure includes comprehensive Doxygen-style documentation
- **⚡ Memory Efficient:** Optimized memory usage patterns with 32-dimensional embeddings, suitable for embedded deployment
- **🏗️ Cache-Friendly:** Row-major matrix operations optimized for modern CPU cache hierarchies

## 🌿 PlantVIT Model Specifications

PlantVIT is a lightweight Vision Transformer designed for real-time diagnosis in agricultural environments:

- **Image Size:** 256×256×3
- **Patch Size:** 32×32 (total of 64 patches)
- **Embedding Dimension:** 32 (drastically reduced from standard ViT's 768)
- **Encoder Blocks:** 3 (reduced from standard ViT's 12)
- **Attention Heads:** 3
- **Head Dimension:** 32
- **MLP Dimension:** 16
- **Classification Classes:** 9 tomato disease states

### Tomato Disease Classification Classes

1. **Bacterial_spot** - Bacterial spot disease
2. **Early_blight** - Early blight disease
3. **Late_blight** - Late blight disease
4. **Leaf_mold** - Leaf mold disease
5. **Septoria_leaf_spot** - Septoria leaf spot disease
6. **Spider_mites** - Spider mites infestation
7. **Target_spot** - Target spot disease
8. **Yellow_leaf_curl_virus** - Yellow leaf curl virus
9. **Healthy** - Healthy tomato plant

## Project Structure

```
PlantVIT-C/
├── include/                    # Header files for all modules
│   ├── attention.h             # Multi-Head Self-Attention declarations
│   ├── layernorm.h             # Layer Normalization declarations
│   ├── linear.h                # Linear transformation declarations
│   ├── mlp.h                   # Feed-Forward Network declarations
│   ├── vit_config.h            # PlantVIT model configuration constants
│   ├── vit_math.h              # Mathematical utility functions
│   └── vit.h                   # Complete PlantVIT model structure
├── src/                        # Source implementations
│   ├── attention.c             # MHSA implementation (3-head processing)
│   ├── layernorm.c             # Layer normalization with numerical stability
│   ├── linear.c                # Optimized matrix-vector multiplication
│   ├── main.c                  # Main executable with complete weight loading
│   ├── mlp.c                   # MLP block with GELU activation
│   ├── vit_math.c              # Core mathematical operations
│   └── vit.c                   # Complete PlantVIT forward pass with patch embedding
├── utils/                      # Utility scripts
│   └── export_weights.py       # PlantVIT weight extraction to binary format
├── Makefile                    # Build system with optimization flags
└── README.md                   # This documentation
```

## How to Build and Run

### Prerequisites

Ensure you have the following tools installed:

- **C Compiler:** `gcc` or `clang` with C99/C11 support
- **Build System:** `make` utility
- **Python Environment:** `python3` with the following packages:
  ```bash
  pip install torch torchvision einops numpy
  ```

### Step 1: Generate Weights

Extract PlantVIT model weights and convert them to binary format:

```bash
python3 utils/export_weights.py
```

This will create the PlantVIT model structure and (if available, load from `plantvit_tomato.pth`) create a `vit_weights.bin` file containing all model parameters in the exact order expected by the C implementation.

### Step 2: Compile the Project

Build the C project using the provided Makefile:

```bash
make all
```

This compiles all source files with optimization flags (`-O2`) and creates the `vit_inference` executable.

### Step 3: Run the Inference Demo

Execute the compiled program:

```bash
./vit_inference
```

**Expected Output:**

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

### Additional Build Commands

```bash
make run    # Compile and run in one step
make clean  # Remove all build artifacts
```

## Architecture Overview

The PlantVIT-C implementation follows a lightweight Vision Transformer architecture:

1. **Patch Embedding:** Divides input images (256×256×3) into 32×32 patches, creating 64 patch tokens
   - **Special Structure:** LayerNorm → Linear → LayerNorm (different from standard ViT)
2. **Position Embedding:** Adds learnable positional encodings to patch embeddings
3. **Transformer Encoder:** 3 identical blocks, each containing:
   - **Multi-Head Self-Attention (MHSA):** 3 attention heads with 32-dimensional head space
   - **Layer Normalization:** Applied before each sub-layer (Pre-LN architecture)
   - **MLP Block:** Feed-forward network with GELU activation (32→16→32)
   - **Residual Connections:** Skip connections around each sub-layer
4. **Classification Head:** Final layer normalization + linear projection to 9 classes (tomato diseases)

### Data Flow

```
Input Image (3×256×256)
    ↓ Patch Embedding (LayerNorm → Linear → LayerNorm)
Patches (64×32) + CLS Token (1×32) + Position Embeddings
    ↓ 3× Transformer Encoder Blocks
    ↓ [LayerNorm → MHSA → Add] → [LayerNorm → MLP → Add]
Feature Representation (65×32)
    ↓ Extract CLS Token + Final LayerNorm
    ↓ Classification Head
Logits (9 tomato disease classes)
```

## Implementation Details

### Memory Management

- **Static Allocation Preferred:** Most buffers use stack allocation for embedded compatibility
- **Complete Dynamic Allocation:** Full memory management implementation for all weights and buffers
- **Pre-allocated Buffers:** The `ViTModel` structure includes intermediate buffers to avoid allocation during inference

### Numerical Stability

- **Softmax:** Uses max-subtraction technique to prevent overflow
- **Layer Normalization:** Includes epsilon parameter (1e-6) for division stability
- **GELU Activation:** Uses `erff()` for mathematically accurate approximation

### Optimization Features

- **Cache-Friendly Access:** Row-major matrix operations with sequential memory access
- **Compiler Optimizations:** `-O2` flag enables vectorization and loop optimizations
- **Modular Design:** Each component can be individually optimized or replaced
- **Lightweight Architecture:** 32-dimensional embeddings significantly reduce memory usage

## Key Implementation Features

### Fully Implemented Components

1. **Complete Weight Loading:**

   - Full `load_weights()` function implementation in `src/main.c`
   - Proper memory allocation for each weight tensor
   - Binary file reading perfectly matched to export order

2. **Advanced Patch Embedding:**

   - LayerNorm → Linear → LayerNorm structure implementation
   - Complete patch extraction from raw images
   - Optimized for agricultural image processing

3. **Agricultural-Specific Features:**
   - Tomato disease classification with confidence scores
   - Disease name mapping and result interpretation
   - Optimized for real-time agricultural diagnostics

## Future Development Plans

### Performance Optimizations

1. **SIMD Instructions:** Vectorize matrix operations using ARM NEON or x86 SSE
2. **Multi-threading:** Parallelize attention heads using OpenMP
3. **Quantization:** Support INT8 inference for faster embedded deployment
4. **Memory Pool:** Replace `malloc` with pre-allocated memory pools

### Agriculture-Specific Features

1. **Real-time Image Processing:** Direct inference from camera inputs
2. **Multi-crop Support:** Extend to other plant disease classification models
3. **Environmental Data Integration:** Combine with temperature, humidity, and other environmental data
4. **IoT Integration:** Support for MQTT, LoRa, and other IoT protocols

### Model Extensions

1. **Extended Disease Classes:** Additional tomato diseases and pest classifications
2. **Multi-crop Models:** Disease diagnosis for crops other than tomatoes
3. **Severity Assessment:** Evaluate disease progression stages and severity levels
4. **Preventive Recommendations:** AI-driven treatment and prevention suggestions

## Contributing

Contributions are welcome! Please feel free to:

- Add performance optimizations
- Develop new agriculture-specific features
- Extend support for other crops
- Improve documentation and examples
- Add comprehensive test suites

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- **Attention Is All You Need:** Vaswani et al. (2017) - The original Transformer paper
- **An Image is Worth 16x16 Words:** Dosovitskiy et al. (2020) - The Vision Transformer paper
- **Agricultural AI Community:** For advancing smart farming technologies
- **Embedded AI Community:** For inspiring deployment-focused ML implementations

---

**Note:** This implementation prioritizes practical agricultural applications and educational value. PlantVIT's lightweight architecture is specifically designed for real-time disease diagnosis on embedded systems and IoT devices in farming environments.
