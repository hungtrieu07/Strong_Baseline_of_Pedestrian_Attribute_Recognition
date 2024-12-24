import os
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchvision.transforms import Compose

from dataset.AttrDataset import get_transform
from models.base_block import FeatClassifier, BaseClassifier
from models.resnet import resnet50
from tools.function import get_model_log_path
from tools.utils import set_seed

set_seed(605)


def process_frame(frame, yolo_model, reid_model, transform, attr_names):
    """
    Process a single video frame: detect persons, crop, and perform ReID.
    
    Args:
        frame (ndarray): Video frame.
        yolo_model (YOLO): YOLO detection model.
        reid_model (torch.nn.Module): Trained ReID model.
        transform (Compose): Transformations for the ReID model.
        attr_names (list): List of attribute names.
        
    Returns:
        frame (ndarray): Annotated video frame.
    """
    # Perform YOLO detection
    results = yolo_model(frame)
    detections = results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, conf, class]
    
    for detection in detections:
        x1, y1, x2, y2, conf, cls = map(int, detection[:6])
        if int(cls) != 0:  # Only process persons (class 0 in COCO)
            continue

        # Crop the person
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue
        
        # Convert to PIL Image and apply transforms
        person_image = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
        person_trans = transform(person_image).unsqueeze(0)  # Add batch dimension

        if torch.cuda.is_available():
            person_trans = person_trans.cuda()

        # Perform ReID inference
        reid_model.eval()
        with torch.no_grad():
            outputs = reid_model(person_trans)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]

        # Get top 5 attributes
        top5_indices = np.argsort(probs)[-5:][::-1]  # Indices of top 5 attributes
        top5_attrs = [(attr_names[i], probs[i]) for i in top5_indices]

        # Annotate frame with ReID results
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for i, (attr, prob) in enumerate(top5_attrs):
            text = f"{attr}: {prob:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10 - i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return frame


def main(args):
    # Load YOLO model
    yolo_model = YOLO("yolov8n.pt")  # Replace with your YOLO model path if needed

    # Load ReID model
    exp_dir = os.path.join('exp_result', args.dataset)
    model_dir, _ = get_model_log_path(exp_dir, args.dataset)
    save_model_path = os.path.join(model_dir, 'ckpt_max.pth')

    train_tsfm, valid_tsfm = get_transform(args)

    backbone = resnet50()
    classifier = BaseClassifier(nattr=args.attr_num)
    reid_model = FeatClassifier(backbone, classifier)

    if torch.cuda.is_available():
        reid_model = torch.nn.DataParallel(reid_model).cuda()

    # Load the trained ReID model
    checkpoint = torch.load(save_model_path)
    reid_model.load_state_dict(checkpoint['state_dicts'])

    # Attribute names (replace with actual attribute names from your dataset)
    attr_names = [f"Attr_{i}" for i in range(args.attr_num)]

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
        annotated_frame = process_frame(frame, yolo_model, reid_model, valid_tsfm, attr_names)

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
    parser.add_argument('--video_path', type=str, default='test_video.mp4', help='Path to the input video.')
    parser.add_argument('--dataset', type=str, default='your_dataset', help='Dataset name.')
    parser.add_argument('--attr_num', type=int, default=20, help='Number of attributes.')
    args = parser.parse_args()
    main(args)