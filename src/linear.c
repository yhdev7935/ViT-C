// src/linear.c

#include "linear.h"

void linear(const float* x, const float* W, const float* b,
            float* y, int in_dim, int out_dim) {
    // Python PyTorch Linear: y = x @ W.T + b
    // W is stored as (out_dim, in_dim) in row-major order
    // We need to compute y[o] = sum(x[i] * W[o][i]) + b[o]
    
    for (int o = 0; o < out_dim; ++o) {
        // Start with the bias term. If bias is NULL, start with 0.
        y[o] = b ? b[o] : 0.0f;
        
        // For PyTorch compatible layout: W[o * in_dim + i] gives W[o][i]
        for (int i = 0; i < in_dim; ++i) {
            y[o] += x[i] * W[o * in_dim + i];
        }
    }
} 