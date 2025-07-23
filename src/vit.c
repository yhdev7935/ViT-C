// src/vit.c

#include "vit.h"
#include "linear.h"
#include "layernorm.h"
#include "vit_math.h"
#include <string.h> // For memcpy
#include <stdlib.h> // For malloc/free
#include <stdio.h>  // For printf
#include <math.h>   // For sqrtf

// Helper function to print first 10 values and stats of a float array
void print_debug_info(const char* name, const float* data, int size, int print_count) {
    if (print_count > size) print_count = size;
    
    printf("%s first %d values: [", name, print_count);
    for (int i = 0; i < print_count; ++i) {
        printf("%.6f", data[i]);
        if (i < print_count - 1) printf(", ");
    }
    printf("]\n");
    
    // Calculate mean and std
    float mean = 0.0f;
    for (int i = 0; i < size; ++i) {
        mean += data[i];
    }
    mean /= size;
    
    float variance = 0.0f;
    for (int i = 0; i < size; ++i) {
        float diff = data[i] - mean;
        variance += diff * diff;
    }
    variance /= size;
    float std = sqrtf(variance);
    
    printf("%s stats: mean=%.6f, std=%.6f\n", name, mean, std);
}

// Helper function for a single encoder block forward pass
static void encoder_block_forward(EncoderBlock* block, float* x, int block_idx) {
    (void)block_idx; // Suppress unused parameter warning
    const int num_tokens = NUM_PATCHES + 1;
    float residual[num_tokens * EMBED_DIM];
    
    // 1. Attention with internal LayerNorm (Pre-LN)
    // attention_forward now handles LayerNorm internally
    float attn_output[num_tokens * EMBED_DIM];
    attention_forward(x, &block->attention_weights, attn_output);

    // 2. Residual Connection 1
    add(x, attn_output, num_tokens * EMBED_DIM);
    printf("    After attention: ");
    print_debug_info("CLS", x, EMBED_DIM, 10);
    
    memcpy(residual, x, num_tokens * EMBED_DIM * sizeof(float));

    // 3. MLP with internal LayerNorm
    // mlp_forward now handles LayerNorm internally  
    float mlp_output[num_tokens * EMBED_DIM];
    float mlp_intermediate_buffer[MLP_DIM];
    for (int i = 0; i < num_tokens; ++i) {
        mlp_forward(&x[i * EMBED_DIM], &block->mlp_weights, &mlp_output[i * EMBED_DIM], mlp_intermediate_buffer);
    }

    // 4. Residual Connection 2
    add(x, mlp_output, num_tokens * EMBED_DIM);
    printf("    After feedforward: ");
    print_debug_info("CLS", x, EMBED_DIM, 10);
}

void vit_forward(ViTModel* model, const float* image, float* logits) {
    const int patch_dim = NUM_CHANNELS * PATCH_SIZE * PATCH_SIZE;
    
    printf("\n=== 🔧 C STEP-BY-STEP DEBUG ===\n");
    
    // 1. Patch Embedding with pv_pytorch/vit.py to_patches style
    printf("\n--- STEP 1: Patch Extraction ---\n");
    // Python: transposed = torch.transpose(images, 2, 3)    # (n, c, w, h)
    // Python: transposed = torch.transpose(transposed, 1, 3) # (n, h, w, c)
    // Python: patches = torch.reshape(transposed, (n, h*w // self.patch_area, self.patch_area * c))
    
    // Step 1: Create transposed image (c, w, h) from (c, h, w)
    float* transposed_cwh = (float*)malloc(NUM_CHANNELS * IMAGE_SIZE * IMAGE_SIZE * sizeof(float));
    for (int c = 0; c < NUM_CHANNELS; ++c) {
        for (int h = 0; h < IMAGE_SIZE; ++h) {
            for (int w = 0; w < IMAGE_SIZE; ++w) {
                int original_idx = c * IMAGE_SIZE * IMAGE_SIZE + h * IMAGE_SIZE + w;  // (c, h, w)
                int transposed_idx = c * IMAGE_SIZE * IMAGE_SIZE + w * IMAGE_SIZE + h;  // (c, w, h) 
                transposed_cwh[transposed_idx] = image[original_idx];
            }
        }
    }
    
    // Step 2: Create final transposed image (h, w, c) from (c, w, h)
    float* transposed_hwc = (float*)malloc(NUM_CHANNELS * IMAGE_SIZE * IMAGE_SIZE * sizeof(float));
    for (int c = 0; c < NUM_CHANNELS; ++c) {
        for (int w = 0; w < IMAGE_SIZE; ++w) {
            for (int h = 0; h < IMAGE_SIZE; ++h) {
                int cwh_idx = c * IMAGE_SIZE * IMAGE_SIZE + w * IMAGE_SIZE + h;  // (c, w, h)
                int hwc_idx = h * IMAGE_SIZE * NUM_CHANNELS + w * NUM_CHANNELS + c;  // (h, w, c)
                transposed_hwc[hwc_idx] = transposed_cwh[cwh_idx];
            }
        }
    }
    
    // Step 3: Extract patches using Python's reshape logic
    // Python reshape takes memory-contiguous elements and divides them into patches
    // transposed_hwc is in (h, w, c) format with size (256, 256, 3) = 196608 elements
    // Each patch has patch_dim = 3072 elements
    // So we take elements [0:3072] for patch 0, [3072:6144] for patch 1, etc.
    
    float* all_patches = (float*)malloc(NUM_PATCHES * patch_dim * sizeof(float));
    
    for (int patch_idx = 0; patch_idx < NUM_PATCHES; ++patch_idx) {
        float* patch_buffer = &all_patches[patch_idx * patch_dim];
        int start_idx = patch_idx * patch_dim;
        
        // Copy patch_dim consecutive elements from transposed_hwc
        for (int i = 0; i < patch_dim; ++i) {
            patch_buffer[i] = transposed_hwc[start_idx + i];
        }
    }
    
    printf("Patches shape: [1, %d, %d]\n", NUM_PATCHES, patch_dim);
    print_debug_info("First patch", all_patches, patch_dim, 10);
    
    // === STEP 2: Patch embedding ===
    printf("\n--- STEP 2: Patch Embedding ---\n");
    
    for (int patch_idx = 0; patch_idx < NUM_PATCHES; ++patch_idx) {
        float* patch_buffer = &all_patches[patch_idx * patch_dim];
        
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
                  &model->patch_embedding_buffer[patch_idx * EMBED_DIM], EMBED_DIM, 1e-6f);
    }
    
    printf("Patch embedding shape: [1, %d, %d]\n", NUM_PATCHES, EMBED_DIM);
    printf("DEBUG: patch_linear_w first 10: ");
    print_debug_info("", model->weights.patch_linear_w, 10, 10);
    printf("DEBUG: patch_linear_b first 10: ");
    print_debug_info("", model->weights.patch_linear_b, 10, 10);
    print_debug_info("First token", model->patch_embedding_buffer, EMBED_DIM, 10);
    
    // Cleanup transposed arrays
    free(transposed_cwh);
    free(transposed_hwc);
    free(all_patches);

    // === STEP 3: Add CLS token ===
    printf("\n--- STEP 3: Add CLS Token ---\n");
    float* tokens = model->token_buffer;
    memcpy(tokens, model->weights.cls_token, EMBED_DIM * sizeof(float));
    memcpy(tokens + EMBED_DIM, model->patch_embedding_buffer, NUM_PATCHES * EMBED_DIM * sizeof(float));
    
    printf("After CLS token shape: [1, %d, %d]\n", NUM_PATCHES + 1, EMBED_DIM);
    print_debug_info("CLS token", tokens, EMBED_DIM, 10);
    
    // === STEP 4: Add position embeddings ===
    printf("\n--- STEP 4: Add Position Embedding ---\n");
    add(tokens, model->weights.pos_embed, (NUM_PATCHES + 1) * EMBED_DIM);
    
    printf("After position embedding shape: [1, %d, %d]\n", NUM_PATCHES + 1, EMBED_DIM);
    print_debug_info("CLS token after pos_embed", tokens, EMBED_DIM, 10);
    print_debug_info("First patch token after pos_embed", tokens + EMBED_DIM, EMBED_DIM, 10);

    // === STEP 5: Transformer Encoder blocks ===
    printf("\n--- STEP 5: Transformer Blocks ---\n");
    for (int i = 0; i < NUM_ENCODER_BLOCKS; ++i) {
        printf("\n  Block %d:\n", i);
        encoder_block_forward(&model->weights.encoder_blocks[i], tokens, i);
    }

    // === STEP 6: Final Layer Norm ===
    printf("\n--- STEP 6: Final Layer Norm ---\n");
    float final_norm_output[EMBED_DIM];
    layernorm(tokens, model->weights.final_norm_weight, model->weights.final_norm_bias, final_norm_output, EMBED_DIM, 1e-6f);
    
    print_debug_info("After final norm: CLS", final_norm_output, EMBED_DIM, 10);
    
    // === STEP 7: Final classification head ===
    printf("\n--- STEP 7: Classification Head ---\n");
    linear(final_norm_output, model->weights.head_weights, model->weights.head_bias, logits, EMBED_DIM, NUM_CLASSES);
    
    printf("Final logits: [");
    for (int i = 0; i < NUM_CLASSES; ++i) {
        printf("%.6f", logits[i]);
        if (i < NUM_CLASSES - 1) printf(", ");
    }
    printf("]\n");
    
    // Calculate logits stats
    float mean = 0.0f;
    for (int i = 0; i < NUM_CLASSES; ++i) {
        mean += logits[i];
    }
    mean /= NUM_CLASSES;
    
    float variance = 0.0f;
    for (int i = 0; i < NUM_CLASSES; ++i) {
        float diff = logits[i] - mean;
        variance += diff * diff;
    }
    variance /= NUM_CLASSES;
    float std = sqrtf(variance);
    
    printf("Logits stats: mean=%.6f, std=%.6f\n", mean, std);
} 