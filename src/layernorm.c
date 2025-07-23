// src/layernorm.c

#include "layernorm.h"
#include <math.h>
#include <stdio.h>

void layernorm(const float* x, const float* weight, const float* bias,
               float* y, int dim, float eps) {
    // 1. Calculate the mean
    float mean = 0.0f;
    for (int i = 0; i < dim; ++i) {
        mean += x[i];
    }
    mean /= dim;

    // 2. Calculate the variance
    float variance = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = x[i] - mean;
        variance += diff * diff;
    }
    variance /= dim;

    // Debug output for first call only (CLS token)
    static int call_count = 0;
    if (call_count == 0 && dim == 32) {
        printf("        LayerNorm debug: mean=%.6f, variance=%.6f, eps=%.8f\n", mean, variance, eps);
        printf("        sqrt(variance + eps)=%.6f, inv_stddev=%.6f\n", sqrtf(variance + eps), 1.0f / sqrtf(variance + eps));
        call_count++;
    }

    // 3. Normalize and apply scale/shift
    const float inv_stddev = 1.0f / sqrtf(variance + eps);
    for (int i = 0; i < dim; ++i) {
        y[i] = (x[i] - mean) * inv_stddev * weight[i] + bias[i];
    }
} 