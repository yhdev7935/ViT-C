// include/vit_config.h

#ifndef VIT_CONFIG_H
#define VIT_CONFIG_H

// --- ViT-Base Configuration ---
// Reference: "An Image is Worth 16x16 Words" paper, ViT-Base model.
// These parameters must match the pre-trained model being used.

// Image and Patch Dimensions
#define IMAGE_SIZE 224
#define NUM_CHANNELS 3
#define PATCH_SIZE 16

// Derived Patch Configuration
#define NUM_PATCHES_PER_DIM (IMAGE_SIZE / PATCH_SIZE)
#define NUM_PATCHES (NUM_PATCHES_PER_DIM * NUM_PATCHES_PER_DIM) // 14 * 14 = 196

// Transformer Core Dimensions
#define EMBED_DIM 768         // The dimensionality of the patch embeddings (D).
#define NUM_ENCODER_BLOCKS 12 // Number of Transformer Encoder layers.
#define NUM_HEADS 12          // Number of attention heads.
#define HEAD_DIM (EMBED_DIM / NUM_HEADS) // Dimensionality of each attention head.

// MLP (Feed-Forward) Network Configuration
#define MLP_RATIO 4
#define MLP_DIM (EMBED_DIM * MLP_RATIO) // 768 * 4 = 3072

// Classification Head
#define NUM_CLASSES 1000 // For ImageNet-1K

// Other constants
#define CLS_TOKEN_SIZE EMBED_DIM
#define POS_EMBED_SIZE ((NUM_PATCHES + 1) * EMBED_DIM) // +1 for the [class] token

#endif // VIT_CONFIG_H 