import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="models\quantized_CNN_model.tflite")
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print input details
print("Input Details:")
for detail in input_details:
    print(detail)

# Print output details
print("Output Details:")
for detail in output_details:
    print(detail)

# Print model layers and their details
def print_tensor_details(interpreter):
    tensor_details = interpreter.get_tensor_details()
    for tensor in tensor_details:
        print(f"Name: {tensor['name']}, Index: {tensor['index']}, Shape: {tensor['shape']}, Type: {tensor['dtype']}, Quantization Parameters: {tensor['quantization_parameters']}")

print_tensor_details(interpreter)
