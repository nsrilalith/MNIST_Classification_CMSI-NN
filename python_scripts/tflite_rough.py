import tensorflow as tf
import numpy as np

def quantize_input(input_data, scale, zero_point):
    return np.round(input_data / scale + zero_point).astype(np.int8)

def dequantize_output(output_data, scale, zero_point):
    return scale * (output_data.astype(np.float32) - zero_point)

def load_and_run_model(model_path, input_data):
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    
    # Allocate tensors
    interpreter.allocate_tensors()

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Check input type and set the input data accordingly
    input_scale, input_zero_point = input_details[0]['quantization']
    if input_details[0]['dtype'] == np.float32:
        # If the model expects float32 input, directly set the input data
        interpreter.set_tensor(input_details[0]['index'], input_data)
    else:
        # Quantize the input data if the model expects int8
        quantized_input_data = np.round(input_data / input_scale + input_zero_point).astype(np.int8)
        interpreter.set_tensor(input_details[0]['index'], quantized_input_data)

    # Run the model
    interpreter.invoke()

    # Get the output data
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Dequantize the output data if necessary
    output_scale, output_zero_point = output_details[0]['quantization']
    if output_details[0]['dtype'] == np.int8:
        output_data = output_scale * (output_data.astype(np.float32) - output_zero_point)

    return output_data

# Example usage
model_path = 'models\quantized_CNN_model.tflite'
input_data = np.array([
    0.0, 0.0, 0.3125, 0.8125, 0.5625, 0.0625, 0.0, 0.0, 0.0, 0.0, 
    0.8125, 0.9375, 0.625, 0.9375, 0.3125, 0.0, 0.0, 0.1875, 0.9375, 0.125, 
    0.0, 0.6875, 0.5, 0.0, 0.0, 0.25, 0.75, 0.0, 0.0, 0.5, 
    0.5, 0.0, 0.0, 0.3125, 0.5, 0.0, 0.0, 0.5625, 0.5, 0.0, 
    0.0, 0.25, 0.6875, 0.0, 0.0625, 0.75, 0.4375, 0.0, 0.0, 0.125, 
    0.875, 0.3125, 0.625, 0.75, 0.0, 0.0, 0.0, 0.0, 0.375, 0.8125, 
    0.625, 0.0, 0.0, 0.0
]).reshape(1, 8, 8, 1).astype(np.float32)
output_data = load_and_run_model(model_path, input_data)
print("Output Data:", output_data)
