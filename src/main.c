#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include "weights.h"
#include "image6.h"
#include "image3.h"
#include "image0.h"
#include "image7.h"
#include "image1.h"
#include "image2.h"
#include "image4.h"
#include "image5.h"
#include "image8.h"
#include "image9.h"
#include "arm_nnfunctions.h"

// Input dimensions
#define INPUT_BATCHES 1
#define INPUT_H 8
#define INPUT_W 8
#define INPUT_CH 1

// Convolution layer dimensions
#define CONV1_OUT_CH 8
#define CONV1_KERNEL_H 3
#define CONV1_KERNEL_W 3
#define CONV1_OUT_H 6
#define CONV1_OUT_W 6

//Max Pooling Layer Dimensions
#define MAX_POOL_OUT_H 3
#define MAX_POOL_OUT_W 3

#define INPUT_DIM INPUT_BATCHES*INPUT_H*INPUT_W*INPUT_CH
#define CONV_FILTER_DIMS 72
#define CONV_OUT_CH 8
#define HIDDEN_DIM_1 INPUT_BATCHES*CONV1_OUT_H*CONV1_OUT_W*CONV1_OUT_CH
#define HIDDEN_DIM_2 INPUT_BATCHES*MAX_POOL_OUT_H*MAX_POOL_OUT_W*CONV1_OUT_CH
#define OUTPUT_DIM 10

// Define the quantization parameters
#define INPUT_OFFSET 128
#define OUTPUT_OFFSET -60
#define INPUT_SCALE 0.00392157
#define INPUT_ZERO_POINT -128
#define OUTPUT_SCALE 0.11720886
#define OUTPUT_ZERO_POINT 60

// Define the extracted multipliers and shifts for each layer
#define FC1_MULTIPLIER 1170927744
#define FC1_SHIFT -8

// Quantization parameters for each layer
int32_t conv_multiplier[CONV1_OUT_CH] = {2079830784, 1937730560, 1889943040, 1970120960, 1132961024, 1886510720, 1789831296, 2009985152};
int32_t conv_shift[CONV1_OUT_CH] = {-9, -9, -9, -9, -8, -9, -9, -9};


void quantize_input(const float *input_data, int8_t *quantized_data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        quantized_data[i] = (int8_t)round(input_data[i] / INPUT_SCALE) + INPUT_ZERO_POINT;
    }
    // Print quantized input data
    printf("Quantized input data:\n");
    for (size_t i = 0; i < size; ++i) {
        printf("%d ", quantized_data[i]);
    }
    printf("\n");
}

void dequantize_output(const int8_t *quantized_data, float *output_data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        output_data[i] = (quantized_data[i] - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;
    }
}

void run_inference(const int8_t *input_data, int8_t *output_data) {
    printf("Entered run_inference function.\n");

    // Convolution Layer 1
    cmsis_nn_context ctx, fcctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_per_channel_quant_params conv_quant_params;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims, max_pool_filter_dims, filter_dims, bias_dims, output_dims;

    int8_t hidden_1[HIDDEN_DIM_1] = {0};
    int8_t hidden_2[HIDDEN_DIM_2] = {0};
    int8_t hidden_temp[HIDDEN_DIM_2] = {0};

    conv_params.input_offset = 128;
    conv_params.output_offset = -128;
    conv_params.stride.w = 1;
    conv_params.stride.h = 1;
    conv_params.padding.w = 0;
    conv_params.padding.h = 0;
    conv_params.dilation.w = 1;
    conv_params.dilation.h = 1;
    conv_params.activation.max = 127;
    conv_params.activation.min = -128;

    conv_quant_params.multiplier = (int32_t *)conv_multiplier;
    conv_quant_params.shift = (int32_t *)conv_shift;

    input_dims.n = INPUT_BATCHES;
    input_dims.h = INPUT_H;
    input_dims.w = INPUT_W;
    input_dims.c = INPUT_CH;

    filter_dims.n = CONV1_OUT_CH;
    filter_dims.h = CONV1_KERNEL_H;
    filter_dims.w = CONV1_KERNEL_W;
    filter_dims.c = INPUT_CH;

    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = CONV1_OUT_CH;

    output_dims.n = INPUT_BATCHES;
    output_dims.h = CONV1_OUT_H;
    output_dims.w = CONV1_OUT_W;
    output_dims.c = CONV1_OUT_CH;

    // Print the weights and biases for debugging
    // printf("Weights of convolution layer:\n");
    // for (int i = 0; i < CONV_FILTER_DIMS; i++) {
    //     printf("%d ", sequential_conv2d_Conv2D[i]);
    // }
    // printf("\n");

    // printf("Biases of convolution layer:\n");
    // for (int i = 0; i < CONV1_OUT_CH; i++) {
    //     printf("%d ", sequential_conv2d_BiasAdd_ReadVariableOp[i]);
    // }
    // printf("\n");

    // printf("Multipliers and shifts for convolution layer:\n");
    // for (int i = 0; i < CONV1_OUT_CH; i++) {
    //     printf("Multiplier[%d]: %d, Shift[%d]: %d\n", i, conv_multiplier[i], i, conv_shift[i]);
    // }
    // printf("\n");

    // Get the required buffer size
    int32_t buffer_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    printf("Buffer Size is %d\n", buffer_size);
    ctx.buf = malloc(buffer_size);
    ctx.size = 0;

    arm_cmsis_nn_status status;
    status = arm_convolve_s8(&ctx, &conv_params, &conv_quant_params,
                             &input_dims, input_data,
                             &filter_dims, sequential_conv2d_Conv2D,
                             &bias_dims, sequential_conv2d_BiasAdd_ReadVariableOp,
                             &output_dims, hidden_1);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buffer_size);
        free(ctx.buf);
    }
    if (status != ARM_CMSIS_NN_SUCCESS) {
        printf("Error in convolution layer: %d\n", status);
        return;
    }

    printf("Convolution Layer output:\n");
    for (int i = 0; i < HIDDEN_DIM_1; i++) {
        printf("%d ", hidden_1[i]);
    }
    printf("\n");

    // for (size_t i = 0; i < HIDDEN_DIM_1; ++i) {
    //     hidden_temp[i] = (hidden_1[i] - 0.019547035917639732) * 128;
    // }
    // printf("Dequantized Convolution output:\n");
    // for (int i=0; i<HIDDEN_DIM_1; i++){
    //     printf("%0.4f ", hidden_temp[i]);
    // }
    // printf("\n");

    // ReLU Activation 1
    // arm_relu_q7(hidden_1, HIDDEN_DIM_1);

    // printf("ReLU Layer Output:\n");
    // for (int i = 0; i < HIDDEN_DIM_1; i++) {
    //     printf("%d ", hidden_1[i]);
    // }
    // printf("\n");

    // MaxPooling2D
    pool_params.padding.h = 0;
    pool_params.padding.w = 0;
    pool_params.stride.h = 2;
    pool_params.stride.w = 2;
    pool_params.activation.max = 127;
    pool_params.activation.min = -128;

    cmsis_nn_dims pool_input_dims;
    pool_input_dims.n = 1;
    pool_input_dims.h = 6;
    pool_input_dims.w = 6;
    pool_input_dims.c = 8;
    cmsis_nn_dims pool_output_dims;
    pool_output_dims.h = 3;
    pool_output_dims.w = 3;
    pool_output_dims.c = 8;

    max_pool_filter_dims.h = 2;
    max_pool_filter_dims.w = 2;
    

    status = arm_max_pool_s8(NULL, &pool_params,
                             &pool_input_dims, hidden_1,
                             &max_pool_filter_dims,
                             &pool_output_dims, hidden_2);

    if (status != ARM_CMSIS_NN_SUCCESS) {
        printf("Error in Max Pooling Layer: %d\n", status);
        return;
    }
    printf("Max Pooling Layer output:\n");
    for (int i = 0; i < HIDDEN_DIM_2; i++) {
        printf("%d ", hidden_2[i]);
    }
    printf("\n");

    arm_reshape_s8(hidden_2, hidden_temp, 72);

    printf("Reshape Layer output:\n");
    for (int i = 0; i < HIDDEN_DIM_2; i++) {
        printf("%d ", hidden_temp[i]);
    }
    printf("\n");

    // Fully Connected Layer 1
    quant_params.multiplier = FC1_MULTIPLIER;
    quant_params.shift = FC1_SHIFT;

    fc_params.input_offset = 128;
    fc_params.filter_offset = 0;
    fc_params.output_offset = 60;
    fc_params.activation.max = 127;
    fc_params.activation.min = -128;

    cmsis_nn_dims fc1_input_dims;
    fc1_input_dims.n = 1;
    fc1_input_dims.h = 1;
    fc1_input_dims.w = 72;
    fc1_input_dims.c = 1;
    cmsis_nn_dims fc1_filter_dims;
    fc1_filter_dims.n = 72;
    fc1_filter_dims.c = 10;
    cmsis_nn_dims fc1_bias_dims;
    fc1_bias_dims.c = 10;
    cmsis_nn_dims fc1_output_dims;
    fc1_output_dims.n = 1;
    fc1_output_dims.c = 10;

    const int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&fc1_filter_dims);
    fcctx.buf = malloc(buf_size);
    fcctx.size = buf_size;

    status = arm_fully_connected_s8(&fcctx, &fc_params, &quant_params,
                                    &fc1_input_dims, hidden_temp,
                                    &fc1_filter_dims, sequential_dense_MatMul,
                                    &fc1_bias_dims, sequential_dense_BiasAdd_ReadVariableOp,
                                    &fc1_output_dims, output_data);

    if (status != ARM_CMSIS_NN_SUCCESS) {
        printf("Error in fully connected layer 1: %d\n", status);
        return;
    }

    if (fcctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(fcctx.buf, 0, buf_size);
        free(fcctx.buf);
    }

    printf("Fully connected layer 1 output:\n");
    for (int i = 0; i < OUTPUT_DIM; i++) {
        printf("%d ", output_data[i]);
    }
    printf("\n");

    // // Softmax Activation
    // arm_softmax_s8(output_data, 1, OUTPUT_DIM, 2120891136, 23, 248, output_data);

    // printf("Softmax activation output:\n");
    // for (int i = 0; i < OUTPUT_DIM; i++) {
    //     printf("%d ", output_data[i]);
    // }
    // printf("\n");
}

int main() {
    // Initialize input data (example)
    int8_t input_data[INPUT_DIM];
    int8_t output_data[OUTPUT_DIM] = {0};
    float prediction[OUTPUT_DIM];

    printf("Program started.\n");
    // Quantize input data
    quantize_input(image0, input_data, INPUT_DIM);

    printf("Starting Inference\n");
    // Perform inference
    run_inference(input_data, output_data);

    // Dequantize output data
    dequantize_output(output_data, prediction, OUTPUT_DIM);

    // Print the output data
    for (int i = 0; i < OUTPUT_DIM; i++) {
        printf("Pre-Quant Output[%d]: %d\n", i, output_data[i]);
    }

    // Print the output data
    for (int i = 0; i < OUTPUT_DIM; i++) {
        printf("Output[%d]: %f\n", i, prediction[i]);
    }

    return 0;
}
