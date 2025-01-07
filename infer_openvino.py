import numpy as np
from openvino.runtime import Core

# Load OpenVINO model
model_path = "openvino_models/reid_model.xml"
core = Core()
model = core.read_model(model=model_path)
compiled_model = core.compile_model(model=model, device_name="CPU")

# Get input and output nodes
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Prepare input data
input_data = np.random.randn(1, 3, 256, 192).astype(np.float32)  # Replace with actual input

# Run inference
result = compiled_model([input_data])

# Process the output
print("Model output:", result[output_layer])
