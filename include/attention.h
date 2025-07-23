// include/attention.h

#ifndef ATTENTION_H
#define ATTENTION_H

#include "vit_config.h"

// Structure to hold the weights and biases for the Multi-Head Self-Attention block
typedef struct {
    // Internal LayerNorm (Pre-LN) - matches notebook structure
    float* norm_weights; // Shape: (EMBED_DIM)
    float* norm_bias;    // Shape: (EMBED_DIM)
    
    // For creating Q, K, V from input tokens
    float* qkv_weights; // Shape: (3 * EMBED_DIM, EMBED_DIM)
    float* qkv_bias;    // Shape: (3 * NUM_HEADS * HEAD_DIM) = 288

    // For projecting the concatenated head outputs
    float* proj_weights; // Shape: (EMBED_DIM, NUM_HEADS * HEAD_DIM) = (32, 96)
    float* proj_bias;    // Shape: (EMBED_DIM)
} AttentionWeights;

/**
 * @brief Performs the forward pass of the Multi-Head Self-Attention block.
 * 
 * @param x Input tensor of shape (NUM_PATCHES + 1, EMBED_DIM).
 * @param weights A pointer to the AttentionWeights structure.
 * @param y Output tensor of shape (NUM_PATCHES + 1, EMBED_DIM).
 * @param buffers A structure or pointer to pre-allocated buffers needed for intermediate calculations,
 *                such as QKV combined matrix, attention scores, etc., to avoid repeated mallocs.
 *                (For simplicity in this prompt, we'll declare buffers inside the function,
 *                 but a real-world implementation would pass them in).
 */
void attention_forward(const float* x, const AttentionWeights* weights, float* y);

#endif // ATTENTION_H 