import tensorflow as tf
import numpy as np
import math

def quantize_scale(scale):
    """
    Calculate the multiplier and shift values for quantization.
    """
    significand, shift = math.frexp(scale)
    significand_q31 = round(significand * (1 << 31))
    return significand_q31, shift

softmax_input_integer_bits = 5

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="models\quantized_CNN_sfmax_model.tflite")
interpreter.allocate_tensors()

# Get tensor details
tensor_details = interpreter.get_tensor_details()

for tensor in tensor_details:
    if tensor['name'] == 'sequential_2/dense_1/MatMul;sequential_2/dense_1/BiasAdd':
        input_scale = tensor['quantization_parameters']['scales']

input_real_multiplier = min(input_scale * (1 << (31 - softmax_input_integer_bits)), (1 << 31) - 1)
(input_multiplier, input_left_shift) = quantize_scale(input_real_multiplier)

diff_min = ((1 << softmax_input_integer_bits) - 1) * (1 << (31 - softmax_input_integer_bits)) / (1 << input_left_shift)
diff_min = math.floor(diff_min)

print(input_multiplier, input_left_shift, diff_min)