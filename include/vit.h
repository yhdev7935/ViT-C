// include/vit.h

#ifndef VIT_H
#define VIT_H

#include "vit_config.h"
#include "attention.h"
#include "mlp.h"

// Structure to hold weights for a single Transformer Encoder Block
typedef struct {
    AttentionWeights attention_weights;
    MLPWeights mlp_weights;
    float* ln1_weights; // Shape: (EMBED_DIM)
    float* ln1_bias;    // Shape: (EMBED_DIM)
    float* ln2_weights; // Shape: (EMBED_DIM)
    float* ln2_bias;    // Shape: (EMBED_DIM)
} EncoderBlock;

// Structure to hold all weights for the PlantVIT model
typedef struct {
    // Patch Embedding (LayerNorm -> Linear -> LayerNorm structure)
    float* patch_ln1_w;         // Shape: (NUM_CHANNELS * PATCH_SIZE * PATCH_SIZE)
    float* patch_ln1_b;         // Shape: (NUM_CHANNELS * PATCH_SIZE * PATCH_SIZE)
    float* patch_linear_w;      // Shape: (EMBED_DIM, NUM_CHANNELS * PATCH_SIZE * PATCH_SIZE)
    float* patch_linear_b;      // Shape: (EMBED_DIM)
    float* patch_ln2_w;         // Shape: (EMBED_DIM)
    float* patch_ln2_b;         // Shape: (EMBED_DIM)

    // Position Embedding and CLS token
    float* pos_embed;           // Shape: (NUM_PATCHES + 1, EMBED_DIM)
    float* cls_token;           // Shape: (1, EMBED_DIM)

    // Transformer Encoder Blocks
    EncoderBlock encoder_blocks[NUM_ENCODER_BLOCKS];

    // Final MLP Head
    float* final_norm_weight; // Shape: (EMBED_DIM)
    float* final_norm_bias;   // Shape: (EMBED_DIM)
    float* head_weights;      // Shape: (NUM_CLASSES, EMBED_DIM)
    float* head_bias;         // Shape: (NUM_CLASSES)
} ViTWeights;

// Main model structure containing weights and pre-allocated buffers
typedef struct {
    ViTWeights weights;
    
    // Intermediate buffers to avoid malloc in the forward pass
    float* patch_embedding_buffer; // Shape: (NUM_PATCHES, EMBED_DIM)
    float* token_buffer;           // Shape: (NUM_PATCHES + 1, EMBED_DIM)
    float* x1_buffer;              // Buffer for encoder input/output
    float* x2_buffer;              // Buffer for encoder intermediate steps
} ViTModel;

/**
 * @brief Performs the full forward pass of the PlantVIT model.
 * 
 * @param model A pointer to the ViTModel struct containing weights and buffers.
 * @param image Input image tensor of shape (NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).
 * @param logits Output logits tensor of shape (NUM_CLASSES).
 */
void vit_forward(ViTModel* model, const float* image, float* logits);

#endif // VIT_H 