import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import numpy as np
import tf2onnx
import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

# 1. Configuration
KERAS_MODEL_PATH = "./ascii_cam_model/model.keras"  # Update this
OUTPUT_DIR = "./ascii_cam_model"
FLOAT_ONNX_PATH = os.path.join(OUTPUT_DIR, "model.onnx")
QUANT_ONNX_PATH = os.path.join(OUTPUT_DIR, "model_quant.onnx")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Load Keras Model and Convert to Standard FLOAT32 ONNX
print("📦 Loading Keras model...")
keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH, compile=False)

print("⚡ Building model execution graph...")
# Create a single dummy frame matching your exact input shape (batch_size=1, height=16, width=8, channels=1)
dummy_input = np.zeros((1, 16, 8, 1), dtype=np.float32)
# Call the model once to force it to build its internal tensor structures
_ = keras_model(dummy_input)

print("🔄 Converting Keras model directly to standard ONNX format...")
keras_model.export(FLOAT_ONNX_PATH, format="onnx")

print(f"✅ Float32 ONNX saved to: {FLOAT_ONNX_PATH}")

# 3. Create a Calibration Data Reader
# Quantization maps floats to int8. It needs to see sample inputs to calibrate its ranges.
class ASCIIQuantCalibrationReader(CalibrationDataReader):
    def __init__(self, batch_count=20, batch_size=32):
        super().__init__()
        self.data_list = []
        self.batch_count = batch_count
        
        # Generate representative mock data matching your normalized (0.0 to 1.0) input
        # NOTE: For the absolute best results, replace this with a slice of your actual training images!
        for _ in range(batch_count):
            mock_batch = np.random.rand(batch_size, 16, 8, 1).astype(np.float32)
            self.data_list.append({"input_layer": mock_batch})
            
        self.enum_data = iter(self.data_list)

    def get_next(self):
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = iter(self.data_list)

# 4. Execute 8-Bit Static Quantization
print("⚡ Quantizing ONNX model to INT8...")
calibration_data_reader = ASCIIQuantCalibrationReader()

quantize_static(
    model_input=FLOAT_ONNX_PATH,
    model_output=QUANT_ONNX_PATH,
    calibration_data_reader=calibration_data_reader,
    activation_type=QuantType.QInt8,  # Quantize activations to signed int8
    weight_type=QuantType.QInt8,      # Quantize weights to signed int8
    extra_options={"ActivationSymmetric": True} # Matches standard normalization centering
)

print(f"🎉 Success! Quantized INT8 ONNX model saved to: {QUANT_ONNX_PATH}")