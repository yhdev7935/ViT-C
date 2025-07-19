// src/mlp.c

#include "mlp.h"
#include "linear.h"
#include "vit_math.h"

void mlp_forward(const float* x, const MLPWeights* weights, float* y, float* intermediate_buffer) {
    // 1. First linear layer: input -> intermediate
    // y = x * W1^T + b1
    linear(x, weights->fc1_weights, weights->fc1_bias, intermediate_buffer, EMBED_DIM, MLP_DIM);

    // 2. GELU activation function (in-place)
    // y = GELU(y)
    gelu(intermediate_buffer, MLP_DIM);

    // 3. Second linear layer: intermediate -> output
    // y = y * W2^T + b2
    linear(intermediate_buffer, weights->fc2_weights, weights->fc2_bias, y, MLP_DIM, EMBED_DIM);
} 