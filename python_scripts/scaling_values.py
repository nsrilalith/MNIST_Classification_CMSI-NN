import numpy as np
import tensorflow as tf

# Load your TFLite model
interpreter = tf.lite.Interpreter(model_path="models/quantized_CNN_model.tflite")
interpreter.allocate_tensors()

# Get input details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Extract the scale and zero point for the input
input_scale, input_zero_point = input_details[0]['quantization']

# Extract the scale and zero point for the output
output_scale, output_zero_point = output_details[0]['quantization']

print(f"Input scale: {input_scale}, Input zero point: {input_zero_point}")
print(f"Output scale: {output_scale}, Output zero point: {output_zero_point}")

# Function to get quantization parameters
def get_quantization_params(tensor_details):
    params = {}
    for tensor in tensor_details:
        params[tensor['name']] = {
            'scale': tensor['quantization'][0],
            'zero_point': tensor['quantization'][1]
        }
    return params

tensor_details = interpreter.get_tensor_details()

# Extract quantization parameters
quant_params = get_quantization_params(tensor_details)

# Print quantization parameters
for name, params in quant_params.items():
    print(f"Tensor: {name}, Scale: {params['scale']}, Zero Point: {params['zero_point']}")

# Refined function to calculate multiplier and shift
def calculate_multiplier_and_shift(scale):
    if scale == 0:
        return 0, 0
    multiplier, shift = np.frexp(scale)
    multiplier = int(round(multiplier * (1 << 31)))
    shift = 31 - shift
    
    return multiplier, shift

# Calculate and print multiplier and shift for each tensor
for name, params in quant_params.items():
    multiplier, shift = calculate_multiplier_and_shift(params['scale'])
    print(f"Tensor: {name}, Multiplier: {multiplier}, Shift: {shift}")

# Save the results to a text file
# with open('python_scripts\quantized_HP_model_scales.txt', 'w') as f:
#     f.write(f"Input scale: {input_scale}, Input zero point: {input_zero_point}\n")
#     f.write(f"Output scale: {output_scale}, Output zero point: {output_zero_point}\n")
#     for name, params in quant_params.items():
#         multiplier, shift = calculate_multiplier_and_shift(params['scale'])
#         f.write(f"Tensor: {name}, Scale: {params['scale']}, Zero Point: {params['zero_point']}, Multiplier: {multiplier}, Shift: {shift}\n")
