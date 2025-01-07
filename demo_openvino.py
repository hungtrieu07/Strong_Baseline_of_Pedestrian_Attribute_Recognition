import cv2
import numpy as np
from ultralytics import YOLO
from openvino.runtime import Core
from PIL import Image

# Load YOLOv8 model for person detection
yolo_model = YOLO("yolov8n.pt")  # Replace with your YOLOv8 model path

# Load OpenVINO ReID model
core = Core()
reid_model_path = "openvino_models/reid_model.xml"  # Path to your OpenVINO model
reid_model = core.read_model(model=reid_model_path)
compiled_reid_model = core.compile_model(model=reid_model, device_name="CPU")
input_layer = compiled_reid_model.input(0)
output_layer = compiled_reid_model.output(0)

# Attribute Names (Replace with your dataset's attribute list)
ATTR_NAMES = [
    "personalLess30", "personalLess45", "personalLess60", "personalLarger60",
    "carryingBackpack", "accessoryHat", "hairLong", "personalMale",
    "carryingMessengerBag", "lowerBodyShorts", "upperBodyShortSleeve",
    "lowerBodyShortSkirt", "upperBodyBlack", "upperBodyBlue",
    "upperBodyBrown", "upperBodyGreen", "upperBodyGrey", "upperBodyOrange",
    "upperBodyPink", "upperBodyPurple", "upperBodyRed", "upperBodyWhite",
    "upperBodyYellow", "lowerBodyBlack", "lowerBodyBlue", "lowerBodyBrown",
    "lowerBodyGreen", "lowerBodyGrey", "lowerBodyOrange", "lowerBodyPink",
    "lowerBodyPurple", "lowerBodyRed", "lowerBodyWhite", "lowerBodyYellow",
    "personalLess15", "personalFemale", "lowerBodyLongSkirt",
    "upperBodyLongSleeve", "carryingLuggageCase", "hairShort",
    "carryingSuitcase"
]

print(len(ATTR_NAMES))  # 40 attributes

# Preprocessing function for ReID model
def preprocess_image(image, input_size=(192, 256)):
    """
    Preprocesses the person image for the ReID model.
    Args:
        image (PIL.Image): Cropped person image.
        input_size (tuple): Target model input size (width, height).
    Returns:
        np.ndarray: Preprocessed image for the model.
    """
    image = image.resize(input_size, Image.BILINEAR)  # Resize image
    image = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
    image = image.transpose(2, 0, 1)  # HWC to CHW
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def sigmoid(x):
    """Applies sigmoid function to logits."""
    return 1 / (1 + np.exp(-x))

# Process frame
def process_frame(frame, attr_names=ATTR_NAMES):
    """
    Processes a video frame: detects persons, crops, and runs ReID inference.
    Args:
        frame (np.ndarray): Video frame.
        attr_names (list): List of attribute names.
    Returns:
        np.ndarray: Annotated frame.
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
        input_data = preprocess_image(person_image, (192, 256))

        # Perform ReID inference
        reid_output = compiled_reid_model([input_data])[output_layer]

        # Apply sigmoid to logits
        reid_output = sigmoid(reid_output[0])  # Output is batch [1, num_attrs]

        # Get top 5 attributes
        top5_indices = np.argsort(reid_output)[-5:][::-1]
        top5_attrs = [(attr_names[i], reid_output[i]) for i in top5_indices]

        # Annotate the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for i, (attr, prob) in enumerate(top5_attrs):
            text = f"{attr}: {prob:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10 - i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return frame

# Main function
def main(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        annotated_frame = process_frame(frame)

        # Display the frame
        cv2.imshow("Frame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("test_video.avi")  # Replace with the path to your video
