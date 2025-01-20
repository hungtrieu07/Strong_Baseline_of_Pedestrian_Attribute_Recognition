#!/bin/bash

# Check if an input ONNX file is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_onnx_model>"
    exit 1
fi

input_model="$1"
output_dir="$(dirname "$input_model")/openvino_model"

# Check if input file exists
if [ ! -f "$input_model" ]; then
    echo "Error: Input ONNX model file does not exist!"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Convert ONNX model to OpenVINO IR format
echo "Converting $input_model to OpenVINO format..."
ovc "$input_model" \
    --output_model "$output_dir/model" \
    --compress_to_fp16 false

if [ $? -eq 0 ]; then
    echo "Conversion completed successfully!"
    echo "Output saved in: $output_dir"
else
    echo "Error: Conversion failed!"
    exit 1
fi