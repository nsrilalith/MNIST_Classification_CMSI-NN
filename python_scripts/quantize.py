import tensorflow as tf
import numpy as np
from sklearn.datasets import load_digits

# Load your trained Keras model
model = tf.keras.models.load_model('models\model_sfmax.h5')

def get_data():
    np.random.seed(1337)
    x_values, y_values = load_digits(return_X_y=True)
    x_values /= x_values.max()
    # reshape to (8 x 8 x 1)
    x_values = x_values.reshape((len(x_values), 8, 8, 1))
    # split into train, validation, test
    TRAIN_SPLIT = int(0.6 * len(x_values))
    TEST_SPLIT = int(0.2 * len(x_values) + TRAIN_SPLIT)
    x_train, x_test, x_validate = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
    y_train, y_test, y_validate = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT])
    return x_train, x_test, x_validate, y_train, y_test, y_validate


# Define a representative dataset generator for quantization
def representative_dataset():
    X_train, X_test, X_validate, y_train, y_test, y_validate = get_data()
    for i in range(len(X_train)):
        input_data = np.array([X_train[i]], dtype=np.float32)
        yield [input_data]
    
# Convert the model to TensorFlow Lite format with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # Quantize input
converter.inference_output_type = tf.int8  # Quantize output
tflite_quant_model = converter.convert()

# Save the quantized model
with open('models\quantized_CNN_sfmax_model.tflite', 'wb') as f:
    f.write(tflite_quant_model)
