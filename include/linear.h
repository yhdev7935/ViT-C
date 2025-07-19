// include/linear.h

#ifndef LINEAR_H
#define LINEAR_H

/**
 * @brief Performs a linear transformation: y = x * W^T + b.
 *        PyTorch's linear layer stores weights as (out_features, in_features),
 *        so a standard row-major matrix multiplication implicitly handles the transpose.
 * 
 * @param x Input vector of size `in_dim`.
 * @param W Weight matrix of size (`out_dim` x `in_dim`). Must be in row-major order.
 * @param b Bias vector of size `out_dim`. Can be NULL if no bias is needed.
 * @param y Output vector of size `out_dim`. This buffer must be pre-allocated.
 * @param in_dim The input dimension.
 * @param out_dim The output dimension.
 */
void linear(const float* x, const float* W, const float* b,
            float* y, int in_dim, int out_dim);

#endif // LINEAR_H 