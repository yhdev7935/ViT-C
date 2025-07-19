// include/layernorm.h

#ifndef LAYERNORM_H
#define LAYERNORM_H

/**
 * @brief Applies Layer Normalization to an input vector.
 *        Formula: y = (x - mean(x)) / sqrt(variance(x) + eps) * weight + bias
 * 
 * @param x Input vector. The operation is performed on this data.
 * @param weight The learnable gain parameter (gamma), of size `dim`.
 * @param bias The learnable bias parameter (beta), of size `dim`.
 * @param y Output vector. Must be a pre-allocated buffer of size `dim`.
 * @param dim The dimension of the input vector.
 * @param eps A small value added to the denominator for numerical stability.
 */
void layernorm(const float* x, const float* weight, const float* bias,
               float* y, int dim, float eps);

#endif // LAYERNORM_H 