// src/layernorm.c

#include "layernorm.h"
#include <math.h>

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

    // 3. Normalize and apply scale/shift
    const float inv_stddev = 1.0f / sqrtf(variance + eps);
    for (int i = 0; i < dim; ++i) {
        y[i] = (x[i] - mean) * inv_stddev * weight[i] + bias[i];
    }
} 