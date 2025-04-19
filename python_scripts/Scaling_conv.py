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

input_details = interpreter.get_input_details()
input_scale, input_zero = input_details[0]['quantization']
print(input_scale, input_zero)
# output_details = interpreter.get_output_details()
# output_scale, output_zero = output_details[0]['quantization']

# Extract quantization parameters for convolution layers
conv_quant_params = {}

for tensor in tensor_details:
    # if tensor['name'] == 'tfl.quantize':
    #     input_scale = tensor['quantization_parameters']['scales']

    if tensor['name'] == 'sequential/conv2d/Relu;sequential/conv2d/BiasAdd;sequential/conv2d/Conv2D;sequential/conv2d/BiasAdd/ReadVariableOp':
        output_scale = tensor['quantization_parameters']['scales']
    # pass

for detail in tensor_details: #Iterating through every tensor
    if detail['name'] == 'sequential/conv2d/Conv2D':  # Selecting Weight tensor
        scales = detail['quantization_parameters']['scales']
        zero_points = detail['quantization_parameters']['zero_points']
        multipliers = []
        shifts = []
        for scale in scales: #Iterating through channel scales
            effective_scale =  input_scale * scale / output_scale 
            multiplier, shift = get_quantization_params(effective_scale)
            multipliers.append(multiplier) #List of per channel multipliers
            shifts.append(shift) #List of per channel shifts
        
        conv_quant_params[detail['name']] = {
            'scales': scales.tolist(),
            'zero_points': zero_points.tolist(),
            'multipliers': multipliers,
            'shifts': shifts
        }

# Print quantization parameters for convolution layers
for layer, params in conv_quant_params.items():
    print(f"Layer: {layer}")
    print(f"  Scales: {params['scales']}")
    print(f"  Zero Points: {params['zero_points']}")
    print(f"  Multipliers: {params['multipliers']}")
    print(f"  Shifts: {params['shifts']}")
