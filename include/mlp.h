// include/mlp.h

#ifndef MLP_H
#define MLP_H

#include "vit_config.h"

// Structure to hold the weights and biases for the MLP block
typedef struct {
    // Internal LayerNorm - matches notebook structure  
    float* norm_weights; // Shape: (EMBED_DIM)
    float* norm_bias;    // Shape: (EMBED_DIM)
    
    float* fc1_weights; // Shape: (MLP_DIM, EMBED_DIM)
    float* fc1_bias;    // Shape: (MLP_DIM)
    float* fc2_weights; // Shape: (EMBED_DIM, MLP_DIM)
    float* fc2_bias;    // Shape: (EMBED_DIM)
} MLPWeights;

/**
 * @brief Performs the forward pass of the MLP (Feed-Forward Network) block.
 *        The sequence is: Linear -> GELU -> Linear.
 * 
 * @param x Input tensor of size EMBED_DIM.
 * @param weights A pointer to the MLPWeights structure.
 * @param y Output tensor of size EMBED_DIM.
 * @param intermediate_buffer A pre-allocated buffer of size MLP_DIM for the output of the first linear layer.
 */
void mlp_forward(const float* x, const MLPWeights* weights, float* y, float* intermediate_buffer);

#endif // MLP_H 