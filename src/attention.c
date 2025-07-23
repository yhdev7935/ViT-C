// src/attention.c

#include "attention.h"
#include "linear.h"
#include "layernorm.h"
#include "vit_math.h"
#include <stdlib.h> // For malloc/free. Note: In a real embedded system, use static buffers.
#include <string.h> // For memcpy
#include <math.h>   // For sqrtf
#include <stdio.h>  // For printf

// Helper function to print first 10 values of a float array
void print_attention_debug(const char* name, const float* data, int print_count) {
    printf("%s first %d values: [", name, print_count);
    for (int i = 0; i < print_count; ++i) {
        printf("%.6f", data[i]);
        if (i < print_count - 1) printf(", ");
    }
    printf("]\n");
}

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

    printf("\n      === C ATTENTION DEBUG ===\n");
    printf("      Input CLS: ");
    print_attention_debug("", x, 10);
    
    printf("      QKV weight first 10: ");
    print_attention_debug("", weights->qkv_weights, 10);
    printf("      QKV bias first 10: ");
    print_attention_debug("", weights->qkv_bias, 10);
    // LayerNorm weights should now load correctly with fixed proj_weights size
    
    printf("      LayerNorm weight first 10: ");
    print_attention_debug("", weights->norm_weights, 10);
    printf("      LayerNorm bias first 10: ");
    print_attention_debug("", weights->norm_bias, 10);

    // 0. Apply LayerNorm first (Pre-LN structure, matches notebook)
    for (int i = 0; i < num_tokens; ++i) {
        layernorm(&x[i * EMBED_DIM], weights->norm_weights, weights->norm_bias, 
                  &norm_output[i * EMBED_DIM], EMBED_DIM, 1e-6f);
    }
    
    printf("      After LayerNorm CLS: ");
    print_attention_debug("", norm_output, 10);

    // 1. Calculate Q, K, V for all tokens in a single batch operation
    // For each token, apply the linear transformation to get Q, K, V
    for (int i = 0; i < num_tokens; ++i) {
        const float* token_input = &norm_output[i * EMBED_DIM];  // Use normalized input
        float* token_output = &qkv_buffer[i * qkv_dim];
        linear(token_input, weights->qkv_weights, weights->qkv_bias, token_output, EMBED_DIM, qkv_dim);
    }
    
    printf("      QKV CLS: ");
    print_attention_debug("", qkv_buffer, 10);
    printf("      DEBUG: qkv_dim=%d, EMBED_DIM=%d, expected_qkv_dim=%d\n", qkv_dim, EMBED_DIM, 3*EMBED_DIM);
    printf("      DEBUG: QKV[32:42] (should be K): ");
    print_attention_debug("", &qkv_buffer[32], 10);
    printf("      DEBUG: QKV[64:74] (should be V): ");
    print_attention_debug("", &qkv_buffer[64], 10);
    
    // 2. Split QKV into separate Q, K, V tensors  
    // Python uses chunk(3, dim=-1): simple consecutive split [0:32], [32:64], [64:96]
    for (int i = 0; i < num_tokens; ++i) {
        const float* src = &qkv_buffer[i * qkv_dim];
        float* q_dst = &q_buffer[i * EMBED_DIM];
        float* k_dst = &k_buffer[i * EMBED_DIM];
        float* v_dst = &v_buffer[i * EMBED_DIM];
        memcpy(q_dst, src, EMBED_DIM * sizeof(float));                 // Q: [0:32]
        memcpy(k_dst, src + EMBED_DIM, EMBED_DIM * sizeof(float));     // K: [32:64]
        memcpy(v_dst, src + 2 * EMBED_DIM, EMBED_DIM * sizeof(float)); // V: [64:96]
    }
    
    printf("      Q CLS: ");
    print_attention_debug("", q_buffer, 10);
    printf("      K CLS: ");
    print_attention_debug("", k_buffer, 10);
    printf("      V CLS: ");
    print_attention_debug("", v_buffer, 10);

    // Initialize attention output buffer to zero
    memset(attn_output_buffer, 0, num_tokens * EMBED_DIM * sizeof(float));

    // Process each head in parallel (conceptually)
    for (int h = 0; h < NUM_HEADS; ++h) {
        printf("      DEBUG: Processing head %d, accessing q_buffer[%d:%d]\n", h, 0 * EMBED_DIM + h * HEAD_DIM, 0 * EMBED_DIM + h * HEAD_DIM + HEAD_DIM - 1);
        printf("      Q head %d CLS: ", h);
        print_attention_debug("", &q_buffer[0 * EMBED_DIM + h * HEAD_DIM], 10);
        
        // For each head, extract the head-specific portion from each token
        // Head h uses dimensions [h * HEAD_DIM, (h+1) * HEAD_DIM)
        
        // Allocate temporary buffers for this head
        float* q_head = (float*)malloc(num_tokens * HEAD_DIM * sizeof(float));
        float* k_head = (float*)malloc(num_tokens * HEAD_DIM * sizeof(float));
        float* v_head = (float*)malloc(num_tokens * HEAD_DIM * sizeof(float));
        float* k_head_transposed = (float*)malloc(HEAD_DIM * num_tokens * sizeof(float));
        
        // Extract head-specific Q, K, V for all tokens
        for (int t = 0; t < num_tokens; ++t) {
            const float* q_token = &q_buffer[t * EMBED_DIM + h * HEAD_DIM];
            const float* k_token = &k_buffer[t * EMBED_DIM + h * HEAD_DIM];
            const float* v_token = &v_buffer[t * EMBED_DIM + h * HEAD_DIM];
            
            memcpy(&q_head[t * HEAD_DIM], q_token, HEAD_DIM * sizeof(float));
            memcpy(&k_head[t * HEAD_DIM], k_token, HEAD_DIM * sizeof(float));
            memcpy(&v_head[t * HEAD_DIM], v_token, HEAD_DIM * sizeof(float));
        }

        // Create K^T for this head: transpose from [num_tokens, HEAD_DIM] to [HEAD_DIM, num_tokens]
        transpose(k_head, k_head_transposed, num_tokens, HEAD_DIM);
        
        // Pointer to the current head's attention scores
        float* attn_scores_head = attn_scores + h * num_tokens * num_tokens;

        // 3. Calculate Attention Scores: Q * K^T / sqrt(d_k)
        // Q: [num_tokens, HEAD_DIM], K^T: [HEAD_DIM, num_tokens] -> Result: [num_tokens, num_tokens]
        matmul(q_head, k_head_transposed, attn_scores_head, num_tokens, num_tokens, HEAD_DIM);

        // Scale the scores
        const float scale = 1.0f / sqrtf((float)HEAD_DIM);
        for (int i = 0; i < num_tokens * num_tokens; ++i) {
            attn_scores_head[i] *= scale;
        }

        // 4. Apply Softmax to scores for each token (row-wise)
        for (int i = 0; i < num_tokens; ++i) {
            softmax(&attn_scores_head[i * num_tokens], num_tokens);
        }

        // 5. Multiply scores by V: Attn(Q,K,V) = softmax(Q*K^T/sqrt(d_k)) * V
        // attn_scores_head: [num_tokens, num_tokens], v_head: [num_tokens, HEAD_DIM] -> Result: [num_tokens, HEAD_DIM]
        float* head_output = (float*)malloc(num_tokens * HEAD_DIM * sizeof(float));
        matmul(attn_scores_head, v_head, head_output, num_tokens, HEAD_DIM, num_tokens);

        // 6. Copy head output to the corresponding position in the final attention output
        for (int t = 0; t < num_tokens; ++t) {
            float* output_token = &attn_output_buffer[t * EMBED_DIM + h * HEAD_DIM];
            const float* head_token = &head_output[t * HEAD_DIM];
            memcpy(output_token, head_token, HEAD_DIM * sizeof(float));
        }

        // Clean up head-specific buffers
        free(q_head);
        free(k_head);
        free(v_head);
        free(k_head_transposed);
        free(head_output);
    }
    
    printf("      Multi-head concat CLS: ");
    print_attention_debug("", attn_output_buffer, 10);
    
    // 6. Apply final projection
    // The outputs from each head are now concatenated in attn_output_buffer.
    // We now apply the final linear layer to the entire result.
    for (int i = 0; i < num_tokens; ++i) {
        const float* token_input = &attn_output_buffer[i * EMBED_DIM];
        float* token_output = &y[i * EMBED_DIM];
        linear(token_input, weights->proj_weights, weights->proj_bias, token_output, EMBED_DIM, EMBED_DIM);
    }
    
    printf("      Final projection CLS: ");
    print_attention_debug("", y, 10);

    // --- Free Buffers ---
    free(norm_output);
    free(qkv_buffer);
    free(q_buffer);
    free(k_buffer);
    free(v_buffer);
    free(attn_scores);
    free(attn_output_buffer);
} 