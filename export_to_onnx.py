import os
import argparse
import torch
import netron
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d
from torchinfo import summary

# Dictionary to map model names to their constructors
MODEL_MAP = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "resnext50_32x4d": resnext50_32x4d,
    "resnext101_32x8d": resnext101_32x8d,
}

def export_model_to_onnx(model_path):
    # Extract the model variant name from the file path
    model_name = model_path.split('/')[2]
    print(f"Detected model: {model_name}")

    if model_name not in MODEL_MAP:
        raise ValueError(f"Unsupported model variant: {model_name}")
    
    # Initialize the model
    model = MODEL_MAP[model_name]()
    checkpoint = torch.load(model_path)  # Load the checkpoint

    # Extract the state_dict
    if 'state_dicts' in checkpoint:  # Custom save_ckpt structure
        state_dict = checkpoint['state_dicts']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint  # Handle simple state_dict
    else:
        raise RuntimeError("Unexpected checkpoint structure")

    # Remove 'module.' prefix if trained with DataParallel
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Match ResNet keys for backbone or classifier
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}

    # Load the filtered state_dict
    model.load_state_dict(filtered_state_dict, strict=False)  # strict=False to ignore unmatched keys
    model.eval()  # Switch to evaluation mode

    # Define dummy input for the model (modify shape as needed)
    dummy_input = torch.randn(1, 3, 256, 192).cuda()  # Replace with the correct input shape

    # Print a model summary
    print("Model Summary:")
    summary(model, input_size=(1, 3, 256, 192))
    
    # Create a folder to save the ONNX file
    os.makedirs("onnx_models", exist_ok=True)
    onnx_path = os.path.join("onnx_models", f"{model_name}.onnx")

    # Export the model to ONNX format
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],  # Input tensor name
        output_names=["output"],  # Output tensor name
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # Enable dynamic batch size
        opset_version=11,  # ONNX opset version
    )

    print(f"ONNX model has been exported to {onnx_path}")

    # Open the ONNX file in Netron
    netron.start(onnx_path, address=8080, browse=False)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Export a ResNet model to ONNX format and visualize with Netron.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .pth model file.")
    args = parser.parse_args()

    # Export the model
    export_model_to_onnx(args.model_path)
