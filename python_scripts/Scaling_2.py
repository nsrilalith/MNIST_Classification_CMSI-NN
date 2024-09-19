import tensorflow as tf
import numpy as np
import math

def get_quantization_params(scale):
    """
    Calculate the multiplier and shift values for quantization.
    """
    significand, shift = math.frexp(scale)
    significand_q31 = round(significand * (1 << 31))
    return significand_q31, shift

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="models\quantized_CNN_model.tflite")
interpreter.allocate_tensors()

# Get tensor details
tensor_details = interpreter.get_tensor_details()

# input_details = interpreter.get_input_details()
# input_scale, input_zero = input_details[0]['quantization']

output_details = interpreter.get_output_details()
output_scale, output_zero = output_details[0]['quantization']

for tensor in tensor_details:
    if tensor['name'] == 'sequential/flatten/Reshape':
        input_scale = tensor['quantization_parameters']['scales']
    if tensor['name'] == 'StatefulPartitionedCall:0':
        output_scale = tensor['quantization_parameters']['scales']

# Extract quantization parameters for fully connected layers
fc_quant_params = {}

for detail in tensor_details:
    if detail['name'] == 'sequential/dense/MatMul':
        scale = detail['quantization_parameters']['scales'][0]
        zero_point = detail['quantization_parameters']['zero_points'][0]
        input_product_scale = input_scale * scale
        effective_scale = input_product_scale / output_scale
        multiplier, shift = get_quantization_params(effective_scale)
        
        fc_quant_params[detail['name']] = {
            'scale': scale,
            'zero_point': zero_point,
            'multiplier': multiplier,
            'shift': shift
        }

# Print quantization parameters for fully connected layers
for layer, params in fc_quant_params.items():
    print(f"Layer: {layer}")
    print(f"  Scale: {params['scale']}")
    print(f"  Zero Point: {params['zero_point']}")
    print(f"  Multiplier: {params['multiplier']}")
    print(f"  Shift: {params['shift']}")
