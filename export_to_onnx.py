import os
import torch
from models.base_block import FeatClassifier, BaseClassifier
from torchvision.models.resnet import resnet34
from torchinfo import summary

def export_reid_model_to_onnx(model_path, onnx_output_path, num_attributes=41):
    """
    Export the ReID model to ONNX format.

    Args:
        model_path (str): Path to the trained PyTorch model file.
        onnx_output_path (str): Path to save the ONNX model.
        num_attributes (int): Number of attributes for classification.
    """
    # Load backbone
    backbone = resnet34(weights=None)
    backbone = torch.nn.Sequential(*list(backbone.children())[:-2])  # Remove classification layers

    # Define classifier
    classifier = BaseClassifier(nattr=num_attributes, input_dim=512)  # Match backbone output channels

    # Combine backbone and classifier
    model = FeatClassifier(backbone, classifier)
    model = model.cuda()
    model.eval()

    # Load checkpoint
    checkpoint = torch.load(model_path)
    state_dict = checkpoint.get("state_dicts", checkpoint)  # Handle different state_dict structures
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}  # Remove 'module.' prefix if necessary
    model.load_state_dict(state_dict, strict=False)

    # Debug: Verify model and shapes
    dummy_input = torch.randn(1, 3, 256, 192).cuda()  # Adjust input size based on your model
    print("Testing the PyTorch model with dummy input...")
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Model output shape: {output.shape}")  # Should match [1, num_attributes]

    # Print model summary
    print("Model Summary:")
    summary(model, input_size=(1, 3, 256, 192))

    # Export to ONNX
    os.makedirs(os.path.dirname(onnx_output_path), exist_ok=True)
    print(f"Exporting the model to ONNX: {onnx_output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11
    )
    print(f"Model successfully exported to {onnx_output_path}")

    # Verify ONNX export
    import onnx
    print("Verifying ONNX model...")
    onnx_model = onnx.load(onnx_output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification completed successfully!")

if __name__ == "__main__":
    # Example usage
    model_path = "exp_result/resnet34/peta_feat_classifier.pth"  # Path to PyTorch model
    onnx_output_path = "onnx_models/reid_model.onnx"  # Path to save ONNX model
    num_attributes = 41  # Number of attributes for classification
    export_reid_model_to_onnx(model_path, onnx_output_path, num_attributes)
