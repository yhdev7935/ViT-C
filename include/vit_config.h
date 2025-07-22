// include/vit_config.h

#ifndef VIT_CONFIG_H
#define VIT_CONFIG_H

// --- PlantVIT Configuration ---
// Custom lightweight Vision Transformer for tomato disease classification.
// These parameters match the trained PlantVIT model.

// Image and Patch Dimensions
#define IMAGE_SIZE 256
#define NUM_CHANNELS 3
#define PATCH_SIZE 32

// Derived Patch Configuration
#define NUM_PATCHES_PER_DIM (IMAGE_SIZE / PATCH_SIZE)
#define NUM_PATCHES (NUM_PATCHES_PER_DIM * NUM_PATCHES_PER_DIM) // 8 * 8 = 64

// Transformer Core Dimensions
#define EMBED_DIM 32          // The dimensionality of the patch embeddings (D).
#define NUM_ENCODER_BLOCKS 3  // Number of Transformer Encoder layers.
#define NUM_HEADS 3           // Number of attention heads.
#define HEAD_DIM 32           // Dimensionality of each attention head (custom for PlantVIT).

// MLP (Feed-Forward) Network Configuration
#define MLP_DIM 16            // Custom MLP dimension for PlantVIT (not using ratio).

// Classification Head
#define NUM_CLASSES 10        // For tomato disease classification (updated to match saved model)

// Other constants
#define CLS_TOKEN_SIZE EMBED_DIM
#define POS_EMBED_SIZE ((NUM_PATCHES + 1) * EMBED_DIM) // +1 for the [class] token

#endif // VIT_CONFIG_H 