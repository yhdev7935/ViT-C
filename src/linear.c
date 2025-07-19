// src/linear.c

#include "linear.h"

void linear(const float* x, const float* W, const float* b,
            float* y, int in_dim, int out_dim) {
    // Iterate over each output element
    for (int o = 0; o < out_dim; ++o) {
        // Start with the bias term. If bias is NULL, start with 0.
        y[o] = b ? b[o] : 0.0f;
        
        // Pointer to the current row in the weight matrix.
        // This corresponds to the weights for the o-th output feature.
        const float* w_row = &W[o * in_dim];
        
        // Perform the dot product between the input vector 'x' and the weight row 'w_row'.
        for (int i = 0; i < in_dim; ++i) {
            y[o] += x[i] * w_row[i];
        }
    }
} 