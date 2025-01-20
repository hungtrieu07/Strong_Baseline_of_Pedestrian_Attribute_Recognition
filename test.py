import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import onnxruntime as ort
from dataset.AttrDataset import AttrDataset, get_transform
from tools.utils import set_seed
import torch


set_seed(605)

def preprocess_image(image, input_size=(256, 192)):
    """
    Preprocess the cropped person image to match ONNX model input.
    Args:
        image (PIL.Image): Cropped image of the person.
        input_size (tuple): Target size for the model (height, width).
    Returns:
        np.ndarray: Preprocessed image.
    """
    image = image.resize(input_size[::-1], Image.BILINEAR)  # Resize to (width, height)
    image = np.array(image).astype(np.float32) / 255.0     # Normalize to [0, 1]
    image = image.transpose(2, 0, 1)                       # HWC to CHW
    image = np.expand_dims(image, axis=0)                  # Add batch dimension
    return image


def process_frame(frame, yolo_model, onnx_session, attr_names):
    """
    Process a single video frame: detect persons, crop, and perform ReID using ONNX model.
    Args:
        frame (ndarray): Video frame.
        yolo_model (YOLO): YOLO detection model.
        onnx_session (ort.InferenceSession): ONNX Runtime inference session for ReID model.
        attr_names (list): List of attribute names.
    Returns:
        frame (ndarray): Annotated video frame.
    """
    # Perform YOLO detection
    results = yolo_model(frame)
    detections = results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, conf, class]
    
    for detection in detections:
        x1, y1, x2, y2, conf, cls = map(int, detection[:6])
        if cls != 0:  # Only process persons (class 0 in COCO)
            continue

        # Crop the person
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue

        # Preprocess the person image
        person_image = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
        input_data = preprocess_image(person_image)

        # Perform ReID inference using ONNX model
        input_name = onnx_session.get_inputs()[0].name
        output_name = onnx_session.get_outputs()[0].name
        reid_output = onnx_session.run([output_name], {input_name: input_data})[0][0]
        reid_output = torch.sigmoid(torch.tensor(reid_output)).numpy()

        # Get top 5 attributes
        top5_indices = np.argsort(reid_output)[-5:][::-1]
        top5_attrs = [(attr_names[i], reid_output[i]) for i in top5_indices]

        # Annotate the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for i, (attr, prob) in enumerate(top5_attrs):
            text = f"{attr}: {prob:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10 - i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return frame

def main(args):
    # Load YOLO model
    yolo_model = YOLO("yolov8n.pt")  # Replace with your YOLO model path if needed

    # Load ONNX ReID model
    onnx_model_path = "onnx_models/resnet34.onnx"  # Path to your ONNX model
    onnx_session = ort.InferenceSession(onnx_model_path)

    # Load attribute names
    _, valid_tsfm = get_transform(args)
    valid_set = AttrDataset(args=args, split=args.valid_split, transform=valid_tsfm)
    attr_names = valid_set.attr_id

    # Load video
    video_path = args.video_path
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        annotated_frame = process_frame(frame, yolo_model, onnx_session, attr_names)

        # Display the frame
        cv2.imshow("Video", annotated_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    from config import argument_parser
    parser = argument_parser()
    parser.add_argument('--video_path', type=str, default='test_video.avi', help='Path to the input video.')
    parser.add_argument('--dataset', type=str, default='PETA', help='Dataset name.')
    args = parser.parse_args()
    main(args)
