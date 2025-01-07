import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def preprocess_image(image_path, input_size):
    """
    Preprocess the image for inference.

    Args:
        image_path (str): Path to the input image.
        input_size (tuple): Desired input size (height, width).

    Returns:
        np.ndarray: Preprocessed image tensor.
    """
    # Define the preprocessing transformations
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
    ])
    
    # Open the image
    image = Image.open(image_path).convert("RGB")
    
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Convert to numpy array
    return image_tensor.numpy()

def load_onnx_model(onnx_path):
    """
    Load the ONNX model.

    Args:
        onnx_path (str): Path to the ONNX model file.

    Returns:
        onnxruntime.InferenceSession: Loaded ONNX runtime session.
    """
    return ort.InferenceSession(onnx_path)

def run_inference(onnx_session, input_tensor):
    """
    Run inference on the ONNX model.

    Args:
        onnx_session (onnxruntime.InferenceSession): Loaded ONNX runtime session.
        input_tensor (np.ndarray): Preprocessed input tensor.

    Returns:
        np.ndarray: Model predictions.
    """
    input_name = onnx_session.get_inputs()[0].name  # Get input name
    output_name = onnx_session.get_outputs()[0].name  # Get output name
    
    # Perform inference
    result = onnx_session.run([output_name], {input_name: input_tensor})
    return result[0]

def main(image_path, onnx_path, input_size):
    # Preprocess the input image
    input_tensor = preprocess_image(image_path, input_size)

    # Load the ONNX model
    onnx_session = load_onnx_model(onnx_path)

    # Perform inference
    predictions = run_inference(onnx_session, input_tensor)
    
    # Print the predictions
    print("Predictions:", predictions)
    
    predicted_class = np.argmax(predictions)
    print(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    # Input arguments
    image_path = "test_image/8.webp"  # Replace with the path to your test image
    onnx_path = "onnx_models/resnet34.onnx"  # Replace with your ONNX model path
    input_size = (256, 192)  # Replace with the input size used during export

    # Run the main function
    main(image_path, onnx_path, input_size)
