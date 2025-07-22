// src/vit.c

#include "vit.h"
#include "linear.h"
#include "layernorm.h"
#include "vit_math.h"
#include <string.h> // For memcpy

// Helper function for a single encoder block forward pass
static void encoder_block_forward(EncoderBlock* block, float* x) {
    const int num_tokens = NUM_PATCHES + 1;
    float residual[num_tokens * EMBED_DIM];
    memcpy(residual, x, num_tokens * EMBED_DIM * sizeof(float));

    // 1. Layer Norm 1 -> Multi-Head Attention
    float ln1_output[num_tokens * EMBED_DIM];
    for (int i = 0; i < num_tokens; ++i) {
        layernorm(&x[i * EMBED_DIM], block->ln1_weights, block->ln1_bias, &ln1_output[i * EMBED_DIM], EMBED_DIM, 1e-6f);
    }
    
    float attn_output[num_tokens * EMBED_DIM];
    // Note: The provided attention_forward uses malloc. For production, pass in buffers.
    attention_forward(ln1_output, &block->attention_weights, attn_output);

    // 2. Add & Norm (Residual Connection 1)
    add(x, attn_output, num_tokens * EMBED_DIM);
    memcpy(residual, x, num_tokens * EMBED_DIM * sizeof(float));

    // 3. Layer Norm 2 -> MLP
    float ln2_output[num_tokens * EMBED_DIM];
    for (int i = 0; i < num_tokens; ++i) {
        layernorm(&x[i * EMBED_DIM], block->ln2_weights, block->ln2_bias, &ln2_output[i * EMBED_DIM], EMBED_DIM, 1e-6f);
    }

    float mlp_output[num_tokens * EMBED_DIM];
    float mlp_intermediate_buffer[MLP_DIM];
    for (int i = 0; i < num_tokens; ++i) {
        mlp_forward(&ln2_output[i * EMBED_DIM], &block->mlp_weights, &mlp_output[i * EMBED_DIM], mlp_intermediate_buffer);
    }

    // 4. Add & Norm (Residual Connection 2)
    add(x, mlp_output, num_tokens * EMBED_DIM);
}

void vit_forward(ViTModel* model, const float* image, float* logits) {
    const int patch_dim = NUM_CHANNELS * PATCH_SIZE * PATCH_SIZE;
    
    // 1. Patch Embedding with PlantVIT's LayerNorm -> Linear -> LayerNorm structure
    for (int i = 0; i < NUM_PATCHES; ++i) {
        // Extract patch from image
        int row = i / NUM_PATCHES_PER_DIM;
        int col = i % NUM_PATCHES_PER_DIM;
        
        // Extract the current patch (simplified - assumes image is in HWC format)
        float patch_buffer[patch_dim];
        for (int c = 0; c < NUM_CHANNELS; ++c) {
            for (int h = 0; h < PATCH_SIZE; ++h) {
                for (int w = 0; w < PATCH_SIZE; ++w) {
                    int img_h = row * PATCH_SIZE + h;
                    int img_w = col * PATCH_SIZE + w;
                    int patch_idx = c * PATCH_SIZE * PATCH_SIZE + h * PATCH_SIZE + w;
                    int img_idx = c * IMAGE_SIZE * IMAGE_SIZE + img_h * IMAGE_SIZE + img_w;
                    patch_buffer[patch_idx] = image[img_idx];
                }
            }
        }
        
        // PlantVIT patch embedding: LayerNorm -> Linear -> LayerNorm
        float ln1_output[patch_dim];
        float linear_output[EMBED_DIM];
        
        // Step 1: First LayerNorm
        layernorm(patch_buffer, model->weights.patch_ln1_w, model->weights.patch_ln1_b, 
                  ln1_output, patch_dim, 1e-6f);
        
        // Step 2: Linear transformation
        linear(ln1_output, model->weights.patch_linear_w, model->weights.patch_linear_b,
               linear_output, patch_dim, EMBED_DIM);
        
        // Step 3: Second LayerNorm
        layernorm(linear_output, model->weights.patch_ln2_w, model->weights.patch_ln2_b,
                  &model->patch_embedding_buffer[i * EMBED_DIM], EMBED_DIM, 1e-6f);
    }

    // 2. Prepend CLS token and add Position Embeddings
    float* tokens = model->token_buffer;
    memcpy(tokens, model->weights.cls_token, EMBED_DIM * sizeof(float));
    memcpy(tokens + EMBED_DIM, model->patch_embedding_buffer, NUM_PATCHES * EMBED_DIM * sizeof(float));
    add(tokens, model->weights.pos_embed, (NUM_PATCHES + 1) * EMBED_DIM);

    // 3. Pass through Transformer Encoder blocks
    for (int i = 0; i < NUM_ENCODER_BLOCKS; ++i) {
        encoder_block_forward(&model->weights.encoder_blocks[i], tokens);
    }

    // 4. Final MLP Head for classification
    float final_norm_output[EMBED_DIM];
    layernorm(tokens, model->weights.final_norm_weight, model->weights.final_norm_bias, final_norm_output, EMBED_DIM, 1e-6f);
    
    linear(final_norm_output, model->weights.head_weights, model->weights.head_bias, logits, EMBED_DIM, NUM_CLASSES);
} 