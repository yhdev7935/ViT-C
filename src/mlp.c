// src/mlp.c

#include "mlp.h"
#include "linear.h"
#include "layernorm.h"
#include "vit_math.h"
#include <stdlib.h>

void mlp_forward(const float* x, const MLPWeights* weights, float* y, float* intermediate_buffer) {
    // Allocate temporary buffer for normalized input
    float* norm_output = (float*)malloc(EMBED_DIM * sizeof(float));
    
    // 0. Apply LayerNorm first (matches notebook structure: net.0 = LayerNorm)
    layernorm(x, weights->norm_weights, weights->norm_bias, norm_output, EMBED_DIM, 1e-6f);
    
    // 1. First linear layer: normalized_input -> intermediate (net.1)
    // y = norm_x * W1^T + b1
    linear(norm_output, weights->fc1_weights, weights->fc1_bias, intermediate_buffer, EMBED_DIM, MLP_DIM);

    // 2. GELU activation function (in-place) (net.2)
    // y = GELU(y)
    gelu(intermediate_buffer, MLP_DIM);

    // 3. Second linear layer: intermediate -> output (net.4)
    // y = y * W2^T + b2
    linear(intermediate_buffer, weights->fc2_weights, weights->fc2_bias, y, MLP_DIM, EMBED_DIM);
    
    // Clean up
    free(norm_output);
} 