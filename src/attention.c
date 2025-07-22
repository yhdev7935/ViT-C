// src/attention.c

#include "attention.h"
#include "linear.h"
#include "layernorm.h"
#include "vit_math.h"
#include <stdlib.h> // For malloc/free. Note: In a real embedded system, use static buffers.
#include <string.h> // For memcpy
#include <math.h>   // For sqrtf

void attention_forward(const float* x, const AttentionWeights* weights, float* y) {
    const int num_tokens = NUM_PATCHES + 1;
    const int qkv_dim = 3 * EMBED_DIM;

    // --- Temporary Buffers ---
    // In a production-grade embedded system, these should be statically allocated
    // and passed as arguments to avoid dynamic memory allocation during inference.
    float* norm_output = (float*)malloc(num_tokens * EMBED_DIM * sizeof(float));
    float* qkv_buffer = (float*)malloc(num_tokens * qkv_dim * sizeof(float));
    float* q_buffer = (float*)malloc(num_tokens * EMBED_DIM * sizeof(float));
    float* k_buffer = (float*)malloc(num_tokens * EMBED_DIM * sizeof(float));
    float* v_buffer = (float*)malloc(num_tokens * EMBED_DIM * sizeof(float));
    float* attn_scores = (float*)malloc(NUM_HEADS * num_tokens * num_tokens * sizeof(float));
    float* attn_output_buffer = (float*)malloc(num_tokens * EMBED_DIM * sizeof(float));

    // 0. Apply LayerNorm first (Pre-LN structure, matches notebook)
    for (int i = 0; i < num_tokens; ++i) {
        layernorm(&x[i * EMBED_DIM], weights->norm_weights, weights->norm_bias, 
                  &norm_output[i * EMBED_DIM], EMBED_DIM, 1e-6f);
    }

    // 1. Calculate Q, K, V for all tokens in a single batch operation
    // For each token, apply the linear transformation to get Q, K, V
    for (int i = 0; i < num_tokens; ++i) {
        const float* token_input = &norm_output[i * EMBED_DIM];  // Use normalized input
        float* token_output = &qkv_buffer[i * qkv_dim];
        linear(token_input, weights->qkv_weights, weights->qkv_bias, token_output, EMBED_DIM, qkv_dim);
    }
    
    // 2. Split QKV into separate Q, K, V tensors
    // This involves rearranging the data from [tok, 3*dim] to 3x[tok, dim]
    for (int i = 0; i < num_tokens; ++i) {
        const float* src = &qkv_buffer[i * qkv_dim];
        float* q_dst = &q_buffer[i * EMBED_DIM];
        float* k_dst = &k_buffer[i * EMBED_DIM];
        float* v_dst = &v_buffer[i * EMBED_DIM];
        memcpy(q_dst, src, EMBED_DIM * sizeof(float));
        memcpy(k_dst, src + EMBED_DIM, EMBED_DIM * sizeof(float));
        memcpy(v_dst, src + 2 * EMBED_DIM, EMBED_DIM * sizeof(float));
    }

    // Process each head in parallel (conceptually)
    for (int h = 0; h < NUM_HEADS; ++h) {
        // Pointers to the current head's data within Q, K, V buffers
        const float* q_head = q_buffer + h * HEAD_DIM;
        const float* k_head = k_buffer + h * HEAD_DIM;
        const float* v_head = v_buffer + h * HEAD_DIM;

        // Pointer to the current head's attention scores
        float* attn_scores_head = attn_scores + h * num_tokens * num_tokens;

        // Temporary buffer for K^T
        float* k_head_transposed = (float*)malloc(num_tokens * HEAD_DIM * sizeof(float));

        // Create K^T for this head
        // Note: This is inefficient. A better matmul would handle transposing.
        float k_temp[num_tokens][HEAD_DIM];
        for(int i=0; i<num_tokens; ++i)
            for(int j=0; j<HEAD_DIM; ++j)
                k_temp[i][j] = k_head[i*EMBED_DIM + j];
        transpose((float*)k_temp, k_head_transposed, num_tokens, HEAD_DIM);
        
        // 3. Calculate Attention Scores: (Q * K^T) / sqrt(d_k)
        matmul((float*)q_head, k_head_transposed, attn_scores_head, num_tokens, num_tokens, HEAD_DIM);

        // Scale the scores
        const float scale = 1.0f / sqrtf((float)HEAD_DIM);
        for (int i = 0; i < num_tokens * num_tokens; ++i) {
            attn_scores_head[i] *= scale;
        }

        // 4. Apply Softmax to scores for each token
        for (int i = 0; i < num_tokens; ++i) {
            softmax(&attn_scores_head[i * num_tokens], num_tokens);
        }

        // 5. Multiply scores by V: Attn(Q,K,V) = softmax(Q*K^T/sqrt(d_k)) * V
        float* attn_output_head = attn_output_buffer + h * HEAD_DIM;
        matmul(attn_scores_head, (float*)v_head, (float*)attn_output_head, num_tokens, HEAD_DIM, num_tokens);

        free(k_head_transposed);
    }
    
    // 6. Concatenate heads and apply final projection
    // The outputs from each head are already interleaved in attn_output_buffer.
    // We now apply the final linear layer to the entire result.
    for (int i = 0; i < num_tokens; ++i) {
        const float* token_input = &attn_output_buffer[i * EMBED_DIM];
        float* token_output = &y[i * EMBED_DIM];
        linear(token_input, weights->proj_weights, weights->proj_bias, token_output, EMBED_DIM, EMBED_DIM);
    }

    // --- Free Buffers ---
    free(norm_output);
    free(qkv_buffer);
    free(q_buffer);
    free(k_buffer);
    free(v_buffer);
    free(attn_scores);
    free(attn_output_buffer);
} 