// src/vit_math.c

#include "vit_math.h"
#include <math.h>

void gelu(float* x, int size) {
    for (int i = 0; i < size; ++i) {
        // Approximation of the GELU activation function
        x[i] = 0.5f * x[i] * (1.0f + erff(x[i] / sqrtf(2.0f)));
    }
}

void softmax(float* x, int size) {
    // Find the maximum value in the vector for numerical stability
    float max_val = x[0];
    for (int i = 1; i < size; ++i) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    // Calculate the sum of exponentials
    float sum_exp = 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] = expf(x[i] - max_val);
        sum_exp += x[i];
    }

    // Normalize by the sum
    for (int i = 0; i < size; ++i) {
        x[i] /= sum_exp;
    }
}

void add(float* a, const float* b, int size) {
    for (int i = 0; i < size; ++i) {
        a[i] += b[i];
    }
}

void matmul(const float* a, const float* b, float* c, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            c[m * N + n] = 0.0f;
            for (int k = 0; k < K; ++k) {
                c[m * N + n] += a[m * K + k] * b[k * N + n];
            }
        }
    }
}

void transpose(const float* src, float* dst, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
} 