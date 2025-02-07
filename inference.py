import yaml
import cv2
import argparse
from models import EfficientUnet, ESNet
import torch
from typing import Dict
import numpy as np


def postprocess(output: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """Perform postprocess and draw resulting segmentation mask

    Parameters
    ----------
    output : np.ndarray
        Output tensor from model with shape (1, n_classes, H, W)
    colors : np.ndarray
        Colors for drawing mask with shape (n_classes - 1, 3)

    Returns
    -------
    np.ndarray
        Resulting mask with shape (H, W, 3)
    """
    output = (output.squeeze() > 0.5) * 1  # get one-hot encoded classes
    mask = np.zeros((*output.shape[-2:], 3), np.uint8)
    n_classes = output.shape[0]
    for i in range(n_classes - 1):
        mask = np.where(output[i + 1][..., None], colors[i], mask)
    return mask.astype(np.uint8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='effunet', 
                        help="""Segmentation model name. Available models: effunet, esnet""")
    parser.add_argument('--n-classes', type=int, default=1, 
                        help="""Number of classes in dataset""")
    parser.add_argument('--img', type=int, default=320, 
                        help='inference image size (pixels)')
    parser.add_argument('--data', type=str, default='data/data.yaml', 
                        help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default='weights/effunet/best.pt',
                        help='path to weight file to validate')
    parser.add_argument('--video', type=str, default='0',
                        help='path to video file for inference or 0 for webcam')
    
    args = parser.parse_args()

    with open(args.data, 'r') as f:
        data = yaml.full_load(f)

    ### GET MODEL ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # get cuda device
    if torch.cuda.is_available():
        print('[INFO] Using CUDA device')
    else:
        print('[INFO] CUDA is not detected. Using CPU')
    if args.model == 'effunet':
        model = EfficientUnet(args.n_classes).to(device)
    elif args.model == 'esnet':
        model = ESNet(args.n_classes).to(device)
    else:
        print(f'[INFO] Model {args.model} is not supported. Using default effunet')
        model = EfficientUnet(args.n_classes).to(device)
    
    model.load_state_dict(torch.load(args.weights))
    model.eval()

    # Load and process video
    cap = cv2.VideoCapture(args.video if not args.video.isdigit() else int(args.video))

    colors = np.random.randint(low=0, high=255, size=(args.n_classes - 1, 3))  # generate random colors

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or cv2.waitKey(1) == ord('q'):
            break
        frame = cv2.resize(frame, (args.img, args.img))
        x = frame[:, :, ::-1].transpose(2, 0, 1)[None, ...] / 255.0  # preprocess
        x = torch.from_numpy(x).to(device).float()
        with torch.no_grad():
            output = model(x).cpu().numpy()  # inference

        mask = postprocess(output, colors)

        frame = cv2.addWeighted(frame, 0.8, mask, 0.2)
        cv2.imshow('Test video', frame)
    
    cv2.destroyAllWindows()
    cap.release()
