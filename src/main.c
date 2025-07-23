// src/main.c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vit.h"

// Define STB_IMAGE_IMPLEMENTATION before including stb_image.h
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Function to safely read tensor data from binary file
void read_tensor(FILE* fp, float** ptr, int size, const char* name) {
    *ptr = (float*)malloc(size * sizeof(float));
    if (!*ptr) {
        printf("Error: Failed to allocate memory for %s\n", name);
        exit(1);
    }
    
    size_t read_count = fread(*ptr, sizeof(float), size, fp);
    if (read_count != (size_t)size) {
        printf("Error: Failed to read %s (expected %d, got %zu)\n", name, size, read_count);
        exit(1);
    }
    printf("  - Loaded %s: %d elements\n", name, size);
}

// Simple bilinear resize function
void resize_image(unsigned char* input, int input_w, int input_h, int channels,
                  unsigned char* output, int output_w, int output_h) {
    float x_ratio = (float)input_w / output_w;
    float y_ratio = (float)input_h / output_h;
    
    for (int y = 0; y < output_h; y++) {
        for (int x = 0; x < output_w; x++) {
            int px = (int)(x_ratio * x);
            int py = (int)(y_ratio * y);
            
            // Clamp to image bounds
            if (px >= input_w) px = input_w - 1;
            if (py >= input_h) py = input_h - 1;
            
            for (int c = 0; c < channels; c++) {
                output[(y * output_w + x) * channels + c] = 
                    input[(py * input_w + px) * channels + c];
            }
        }
    }
}

// Convert HWC format to CHW format and normalize to [0,1]
// Also convert RGB to BGR to match OpenCV/notebook preprocessing
void convert_hwc_to_chw_and_normalize(unsigned char* hwc_data, float* chw_data, 
                                      int width, int height, int channels) {
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int hwc_idx = (h * width + w) * channels + c;
                // Convert RGB to BGR to match OpenCV (notebook uses cv2.imread which loads as BGR)
                int bgr_channel = c;
                if (c == 0) bgr_channel = 2; // R → B
                else if (c == 2) bgr_channel = 0; // B → R
                // G stays at 1
                
                int chw_idx = bgr_channel * width * height + h * width + w;
                chw_data[chw_idx] = (float)hwc_data[hwc_idx] / 255.0f;
            }
        }
    }
}

// Load and preprocess image for PlantVIT
float* load_and_preprocess_image(const char* image_path) {
    printf("Loading image: %s\n", image_path);
    
    int width, height, channels;
    unsigned char* image_data = stbi_load(image_path, &width, &height, &channels, 3); // Force RGB
    
    if (!image_data) {
        printf("Error: Failed to load image %s\n", image_path);
        printf("Reason: %s\n", stbi_failure_reason());
        return NULL;
    }
    
    printf("Original image: %dx%d with %d channels\n", width, height, 3);
    
    // Allocate memory for resized image
    unsigned char* resized_image = (unsigned char*)malloc(IMAGE_SIZE * IMAGE_SIZE * 3);
    if (!resized_image) {
        printf("Error: Failed to allocate memory for resized image\n");
        stbi_image_free(image_data);
        return NULL;
    }
    
    // Resize to 256x256
    printf("Resizing image to %dx%d\n", IMAGE_SIZE, IMAGE_SIZE);
    resize_image(image_data, width, height, 3, resized_image, IMAGE_SIZE, IMAGE_SIZE);
    
    // Convert to CHW format and normalize
    float* processed_image = (float*)malloc(NUM_CHANNELS * IMAGE_SIZE * IMAGE_SIZE * sizeof(float));
    if (!processed_image) {
        printf("Error: Failed to allocate memory for processed image\n");
        free(resized_image);
        stbi_image_free(image_data);
        return NULL;
    }
    
    printf("Converting to CHW format and normalizing to [0,1]\n");
    convert_hwc_to_chw_and_normalize(resized_image, processed_image, IMAGE_SIZE, IMAGE_SIZE, 3);
    
    // Clean up
    free(resized_image);
    stbi_image_free(image_data);
    
    return processed_image;
}

// Load weights from the binary file created by export_weights.py
void load_weights(ViTModel* model, const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error: Could not open weight file %s\n", filename);
        exit(1);
    }
    
    printf("Loading PlantVIT weights from %s...\n", filename);
    
    // Calculate sizes
    const int patch_dim = NUM_CHANNELS * PATCH_SIZE * PATCH_SIZE; // 3072
    const int num_tokens = NUM_PATCHES + 1; // 65
    
    // 1. Patch Embedding Layers
    printf("--- Loading Patch Embedding ---\n");
    read_tensor(fp, &model->weights.patch_ln1_w, patch_dim, "patch_ln1_w");
    read_tensor(fp, &model->weights.patch_ln1_b, patch_dim, "patch_ln1_b");
    read_tensor(fp, &model->weights.patch_linear_w, EMBED_DIM * patch_dim, "patch_linear_w");
    read_tensor(fp, &model->weights.patch_linear_b, EMBED_DIM, "patch_linear_b");
    read_tensor(fp, &model->weights.patch_ln2_w, EMBED_DIM, "patch_ln2_w");
    read_tensor(fp, &model->weights.patch_ln2_b, EMBED_DIM, "patch_ln2_b");
    
    // 2. Position Embedding and CLS token
    printf("--- Loading Position Embedding & CLS Token ---\n");
    read_tensor(fp, &model->weights.pos_embed, num_tokens * EMBED_DIM, "pos_embed");
    read_tensor(fp, &model->weights.cls_token, EMBED_DIM, "cls_token");
    
    // 3. Transformer Encoder Blocks
    printf("--- Loading Transformer Blocks ---\n");
    for (int i = 0; i < NUM_ENCODER_BLOCKS; ++i) {
        printf("  Block %d:\n", i);
        EncoderBlock* block = &model->weights.encoder_blocks[i];
        
        // Attention weights
        read_tensor(fp, &block->attention_weights.qkv_weights, 3 * NUM_HEADS * HEAD_DIM * EMBED_DIM, "qkv_weights");
        read_tensor(fp, &block->attention_weights.qkv_bias, 3 * EMBED_DIM, "qkv_bias");
        read_tensor(fp, &block->attention_weights.proj_weights, 32 * 96, "proj_weights");
        read_tensor(fp, &block->attention_weights.proj_bias, EMBED_DIM, "proj_bias");
        
        // MLP weights
        read_tensor(fp, &block->mlp_weights.fc1_weights, MLP_DIM * EMBED_DIM, "mlp_fc1_weights");
        read_tensor(fp, &block->mlp_weights.fc1_bias, MLP_DIM, "mlp_fc1_bias");
        read_tensor(fp, &block->mlp_weights.fc2_weights, EMBED_DIM * MLP_DIM, "mlp_fc2_weights");
        read_tensor(fp, &block->mlp_weights.fc2_bias, EMBED_DIM, "mlp_fc2_bias");
        
        // Layer Norms - now inside attention and MLP structures
        read_tensor(fp, &block->attention_weights.norm_weights, EMBED_DIM, "attention_norm_weights");
        read_tensor(fp, &block->attention_weights.norm_bias, EMBED_DIM, "attention_norm_bias");
        read_tensor(fp, &block->mlp_weights.norm_weights, EMBED_DIM, "mlp_norm_weights");
        read_tensor(fp, &block->mlp_weights.norm_bias, EMBED_DIM, "mlp_norm_bias");
    }
    
    // 4. Final Layer Norm and Classification Head
    printf("--- Loading Final Layers ---\n");
    read_tensor(fp, &model->weights.final_norm_weight, EMBED_DIM, "final_norm_weight");
    read_tensor(fp, &model->weights.final_norm_bias, EMBED_DIM, "final_norm_bias");
    read_tensor(fp, &model->weights.head_weights, NUM_CLASSES * EMBED_DIM, "head_weights");
    read_tensor(fp, &model->weights.head_bias, NUM_CLASSES, "head_bias");
    
    fclose(fp);
    printf("Weight loading completed successfully!\n");
}

// Allocate model buffers
void allocate_model_buffers(ViTModel* model) {
    printf("Allocating model buffers...\n");
    
    model->patch_embedding_buffer = (float*)malloc(NUM_PATCHES * EMBED_DIM * sizeof(float));
    model->token_buffer = (float*)malloc((NUM_PATCHES + 1) * EMBED_DIM * sizeof(float));
    model->x1_buffer = (float*)malloc((NUM_PATCHES + 1) * EMBED_DIM * sizeof(float));
    model->x2_buffer = (float*)malloc((NUM_PATCHES + 1) * EMBED_DIM * sizeof(float));
    
    if (!model->patch_embedding_buffer || !model->token_buffer || 
        !model->x1_buffer || !model->x2_buffer) {
        printf("Error: Failed to allocate model buffers\n");
        exit(1);
    }
    
    printf("Model buffers allocated successfully!\n");
}

// Free all allocated memory
void free_model(ViTModel* model) {
    // Free patch embedding weights
    free(model->weights.patch_ln1_w);
    free(model->weights.patch_ln1_b);
    free(model->weights.patch_linear_w);
    free(model->weights.patch_linear_b);
    free(model->weights.patch_ln2_w);
    free(model->weights.patch_ln2_b);
    
    // Free position embedding and CLS token
    free(model->weights.pos_embed);
    free(model->weights.cls_token);
    
    // Free encoder blocks
    for (int i = 0; i < NUM_ENCODER_BLOCKS; ++i) {
        EncoderBlock* block = &model->weights.encoder_blocks[i];
        free(block->attention_weights.qkv_weights);
        free(block->attention_weights.qkv_bias);
        free(block->attention_weights.proj_weights);
        free(block->attention_weights.proj_bias);
        free(block->mlp_weights.fc1_weights);
        free(block->mlp_weights.fc1_bias);
        free(block->mlp_weights.fc2_weights);
        free(block->mlp_weights.fc2_bias);
        free(block->attention_weights.norm_weights);
        free(block->attention_weights.norm_bias);
        free(block->mlp_weights.norm_weights);
        free(block->mlp_weights.norm_bias);
    }
    
    // Free final layers
    free(model->weights.final_norm_weight);
    free(model->weights.final_norm_bias);
    free(model->weights.head_weights);
    free(model->weights.head_bias);
    
    // Free buffers
    free(model->patch_embedding_buffer);
    free(model->token_buffer);
    free(model->x1_buffer);
    free(model->x2_buffer);
}

int main(int argc, char* argv[]) {
    printf("--- PlantVIT Pure C Inference ---\n");
    printf("Image Size: %dx%d, Patch Size: %dx%d, Classes: %d\n", 
           IMAGE_SIZE, IMAGE_SIZE, PATCH_SIZE, PATCH_SIZE, NUM_CLASSES);
    printf("Embed Dim: %d, Blocks: %d, Heads: %d, MLP Dim: %d\n",
           EMBED_DIM, NUM_ENCODER_BLOCKS, NUM_HEADS, MLP_DIM);

    // Check command line arguments
    if (argc != 2) {
        printf("\nUsage: %s <image_path>\n", argv[0]);
        printf("Example: %s tomato_leaf.jpg\n", argv[0]);
        printf("\nSupported formats: JPG, PNG, BMP, TGA\n");
        return 1;
    }

    ViTModel model;
    
    // Allocate buffers
    allocate_model_buffers(&model);
    
    // Load weights from binary file
    load_weights(&model, "vit_weights.bin");

    // Load and preprocess the input image
    printf("\n=== Image Processing ===\n");
    float* input_image = load_and_preprocess_image(argv[1]);
    if (!input_image) {
        printf("Failed to load and process image: %s\n", argv[1]);
        free_model(&model);
        return 1;
    }
    
    // Create an output buffer for logits
    float* logits = (float*)malloc(NUM_CLASSES * sizeof(float));
    if (!logits) {
        printf("Failed to allocate memory for logits.\n");
        free(input_image);
        free_model(&model);
        return 1;
    }

    printf("\n=== Running PlantVIT Inference ===\n");
    
    // Run the actual inference on the real image
    vit_forward(&model, input_image, logits);
    
    printf("Forward pass completed successfully!\n");
    
    // Print all logits for tomato disease classification
    printf("\n=== Tomato Disease Classification Results ===\n");
    const char* disease_names[] = {
        "Bacterial_spot",           // 0
        "Early_blight",             // 1
        "Healthy",                  // 2 - Fixed!
        "Late_blight",              // 3
        "Leaf_mold",                // 4 - Now in correct position!
        "Septoria_leaf_spot",       // 5
        "Spider_mites",             // 6
        "Target_spot",              // 7
        "Tomato_mosaic_virus",      // 8
        "Yellow_leaf_curl_virus"    // 9
    };
    
    for (int i = 0; i < NUM_CLASSES; ++i) {
        printf("  %s: %.6f\n", disease_names[i], logits[i]);
    }
    
    // Find predicted class
    int predicted_class = 0;
    float max_logit = logits[0];
    for (int i = 1; i < NUM_CLASSES; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            predicted_class = i;
        }
    }
    
    printf("\n🌱 DIAGNOSIS RESULT 🌱\n");
    printf("Predicted Disease: %s\n", disease_names[predicted_class]);
    printf("Confidence Score: %.6f\n", max_logit);
    
    // Provide interpretation
    if (predicted_class == 2) { // Healthy
        printf("✅ Good news! The tomato plant appears to be healthy.\n");
    } else {
        printf("⚠️  Disease detected! Consider appropriate treatment measures.\n");
    }

    // Cleanup
    free(input_image);
    free(logits);
    free_model(&model);

    printf("\nPlantVIT inference completed successfully!\n");
    
    return 0;
} 