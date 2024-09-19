import tensorflow as tf
import numpy as np

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="models\quantized_CNN_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print input details
print("Input Details:")
for input_detail in input_details:
    print(input_detail)

# Print output details
print("Output Details:")
for output_detail in output_details:
    print(output_detail)

# Prepare float input data (replace with your actual float input data)
float_input_data = np.array([
    0.0, 0.0, 0.3125, 0.8125, 0.5625, 0.0625, 0.0, 0.0, 0.0, 0.0, 
    0.8125, 0.9375, 0.625, 0.9375, 0.3125, 0.0, 0.0, 0.1875, 0.9375, 0.125, 
    0.0, 0.6875, 0.5, 0.0, 0.0, 0.25, 0.75, 0.0, 0.0, 0.5, 
    0.5, 0.0, 0.0, 0.3125, 0.5, 0.0, 0.0, 0.5625, 0.5, 0.0, 
    0.0, 0.25, 0.6875, 0.0, 0.0625, 0.75, 0.4375, 0.0, 0.0, 0.125, 
    0.875, 0.3125, 0.625, 0.75, 0.0, 0.0, 0.0, 0.0, 0.375, 0.8125, 
    0.625, 0.0, 0.0, 0.0,
]).reshape(1, 8, 8, 1).astype(np.float32)

# Extract quantization parameters
input_scale, input_zero_point = input_details[0]['quantization']
print(f"Input scale: {input_scale}, Input zero point: {input_zero_point}")

# Quantize the float input data
quantized_input_data = np.round(float_input_data / input_scale) + input_zero_point
quantized_input_data = quantized_input_data.astype(np.int8)

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], quantized_input_data)

# Run the model (invoke the interpreter)
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

# Print the quantized output data
print("Quantized Output Data:")
print(output_data)

# Extract output quantization parameters
output_scale, output_zero_point = output_details[0]['quantization']
print(f"Output scale: {output_scale}, Output zero point: {output_zero_point}")

# Dequantize the output data
dequantized_output_data = (output_data - output_zero_point) * output_scale

# Print the dequantized output data
print("Dequantized Output Data:")
print(dequantized_output_data)

# Optional: Compare with the original TensorFlow model
# Load the original TensorFlow model
original_model = tf.keras.models.load_model('models\model.h5')

# Predict using the original model
original_output = original_model.predict(float_input_data)

# Print the original model output
print("Original Model Output:")
print(original_output)
