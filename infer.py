import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from dataset.AttrDataset import AttrDataset, get_transform
from models.base_block import FeatClassifier, BaseClassifier
from models.resnet import resnet50
from tools.function import get_model_log_path, get_pedestrian_metrics
from tools.utils import set_seed

set_seed(605)


def single_image_inference(image_path, model, transform, attr_names, save_dir, log_file):
    """
    Perform inference on a single image, annotate the results, and log to a text file.
    
    Args:
        image_path (str): Path to the input image.
        model (torch.nn.Module): Trained model for inference.
        transform (torchvision.transforms.Compose): Image transformations.
        attr_names (list): List of attribute names.
        save_dir (str): Directory to save annotated results.
        log_file (str): Path to the log file for saving attributes and probabilities.
    """
    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB')
    img_trans = transform(img).unsqueeze(0)  # Add batch dimension

    if torch.cuda.is_available():
        img_trans = img_trans.cuda()

    # Perform inference
    model.eval()
    with torch.no_grad():
        outputs = model(img_trans)
        probs = torch.sigmoid(outputs).cpu().numpy()

    # Annotate the image
    # img = img.resize((256, 512), resample=Image.BILINEAR)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    positive_cnt = 0

    attributes_info = []

    for idx, attr in enumerate(attr_names):
        probability = probs[0, idx]
        attributes_info.append(f"{attr}: {probability:.2f}")
        if probability >= 0.5:  # Threshold for positive attributes
            text = f"{attr}: {probability:.2f}"
            draw.text((10, 10 + 20 * positive_cnt), text, (255, 0, 0), font=font)
            positive_cnt += 1

    # Save the annotated image
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(image_path))
    img.save(save_path)
    print(f"Annotated image saved to: {save_path}")

    # Write attributes and probabilities to the log file
    with open(os.path.join(save_dir, log_file), 'a') as f:
        f.write(f"Image: {os.path.basename(image_path)}\n")
        f.write("\n".join(attributes_info))
        f.write("\n\n")
    print(f"Attributes and probabilities logged to: {os.path.join(save_dir, log_file)}")


def main(args):
    visenv_name = args.dataset
    exp_dir = os.path.join('exp_result', args.dataset)
    model_dir, _ = get_model_log_path(exp_dir, visenv_name)
    save_model_path = os.path.join(model_dir, 'ckpt_max.pth')

    train_tsfm, valid_tsfm = get_transform(args)

    valid_set = AttrDataset(args=args, split=args.valid_split, transform=valid_tsfm)

    print(f'{args.valid_split} set: {len(valid_set)}, attr_num: {valid_set.attr_num}')

    backbone = resnet50()
    classifier = BaseClassifier(nattr=valid_set.attr_num)
    model = FeatClassifier(backbone, classifier)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    # Load the trained model
    checkpoint = torch.load(save_model_path)
    model.load_state_dict(checkpoint['state_dicts'])

    # Perform single-image inference
    demo_image_path = args.demo_image
    save_dir = "annotated_results"
    log_file = "attribute_results.txt"

    single_image_inference(demo_image_path, model, valid_tsfm, valid_set.attr_id, save_dir, log_file)


if __name__ == '__main__':
    from config import argument_parser
    parser = argument_parser()
    parser.add_argument('--demo_image', type=str, default='test_image/test.jpeg',
                        help='Path to the image for demo inference.')
    args = parser.parse_args()
    main(args)

