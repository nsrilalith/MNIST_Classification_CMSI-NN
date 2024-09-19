import tensorflow as tf
import numpy as np
import os

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="models\quantized_CNN_model.tflite")
interpreter.allocate_tensors()

# Get the details of the tensors in the model
tensor_details = interpreter.get_tensor_details()

# Print tensor details for debugging
for tensor in tensor_details:
    print(f"Name: {tensor['name']}, Index: {tensor['index']}, Shape: {tensor['shape']}")

# Extract the weights from the tensors
weights = {}
for tensor in tensor_details:
    tensor_name = tensor['name']
    tensor_index = tensor['index']
    try:
        tensor_data = interpreter.get_tensor(tensor_index)
        weights[tensor_name] = tensor_data
    except ValueError as e:
        print(f"Could not get tensor data for {tensor_name} (index: {tensor_index}): {e}")

# Print or save the weights
for name, data in weights.items():
    print(f"Name: {name}")
    print(f"Shape: {data.shape}")
    print(f"Data: {data}\n")

os.makedirs('weights', exist_ok=True)
os.makedirs('include', exist_ok=True)

for name, data in weights.items():
    # Clean the tensor name to be a valid file name
    clean_name = name.replace('/', '_').replace(':', '_')
    # Save each tensor's data as a numpy array
    np.save(os.path.join('weights', f"{clean_name}.npy"), data)

# Open the header file for writing
with open(os.path.join('include', 'weights.h'), 'w') as header_file:
    header_file.write("#ifndef WEIGHTS_H\n")
    header_file.write("#define WEIGHTS_H\n")
    header_file.write("#include <stdint.h>\n\n")

    # Write each tensor's data as a C array
    for name, data in weights.items():
        # Clean the tensor name to be a valid C identifier
        clean_name = name.replace('/', '_').replace(':', '_')

        # Write the array definition
        header_file.write(f"// {name}\n")
        header_file.write(f"const int8_t {clean_name}[] = {{\n")

        # Write the array data
        flat_data = data.flatten()
        for i, value in enumerate(flat_data):
            if i % 10 == 0:
                header_file.write("\n    ")
            header_file.write(f"{value}, ")
        
        # Remove the trailing comma and space
        header_file.seek(header_file.tell() - 2, os.SEEK_SET)
        header_file.write("\n};\n\n")

    header_file.write("#endif // WEIGHTS_H\n")
