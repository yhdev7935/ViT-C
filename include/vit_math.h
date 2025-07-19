// include/vit_math.h

#ifndef VIT_MATH_H
#define VIT_MATH_H

/**
 * @brief Applies the Gaussian Error Linear Unit (GELU) activation function element-wise.
 *        This is an approximation using the error function (erf).
 *        Formula: y = 0.5 * x * (1 + erf(x / sqrt(2)))
 * @param x Input/Output vector. The operation is done in-place.
 * @param size The number of elements in the vector.
 */
void gelu(float* x, int size);

/**
 * @brief Applies the Softmax function to a vector.
 *        To ensure numerical stability, the maximum value is subtracted from each element
 *        before exponentiation.
 * @param x Input/Output vector. The operation is done in-place.
 * @param size The number of elements in the vector.
 */
void softmax(float* x, int size);

/**
 * @brief Performs element-wise addition of two vectors: a = a + b.
 * @param a First input vector and destination for the result.
 * @param b Second input vector.
 * @param size The number of elements in the vectors.
 */
void add(float* a, const float* b, int size);

/**
 * @brief Performs matrix multiplication: C = A * B.
 *        Matrices are assumed to be in row-major order.
 * @param a Input matrix A of size M x K.
 * @param b Input matrix B of size K x N.
 * @param c Output matrix C of size M x N. Must be pre-allocated.
 * @param M Number of rows in matrix A and C.
 * @param N Number of columns in matrix B and C.
 * @param K Number of columns in matrix A and rows in matrix B.
 */
void matmul(const float* a, const float* b, float* c, int M, int N, int K);

/**
 * @brief Transposes a matrix.
 * @param src Source matrix of size rows x cols.
 * @param dst Destination matrix of size cols x rows. Must be pre-allocated.
 * @param rows Number of rows in the source matrix.
 * @param cols Number of columns in the source matrix.
 */
void transpose(const float* src, float* dst, int rows, int cols);

#endif // VIT_MATH_H 