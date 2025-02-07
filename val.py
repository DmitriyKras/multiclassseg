import glob
import os
import yaml
import json
import numpy as np
from utils import SegmentationDataset
import argparse
from torch.utils.data import DataLoader
import torch
from math import ceil
from tqdm import tqdm
import time
# import metrics
from utils import MeanIoUWrapper
# import models
from models import EfficientUnet, ESNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='effunet', 
                        help="""Segmentation model name. Available models: effunet, esnet""")
    parser.add_argument('--n-classes', type=int, default=1, 
                        help="""Number of classes in dataset""")
    parser.add_argument('--batch-size', type=int, default=20, 
                        help='batch size')
    parser.add_argument('--img', type=int, default=512, 
                        help='train, val image size (pixels)')
    parser.add_argument('--data', type=str, default='data/data.yaml', 
                        help='dataset.yaml path')
    parser.add_argument('--task', type=str, default='val', 
                        help='task of script usage: val or test')
    parser.add_argument('--weights', type=str, default='weights/effunet/best.pt',
                        help='path to weight file to validate')
    args = parser.parse_args()

    # create folders if not exist
    if not os.path.exists('weights'):
        os.makedirs('weights')
    if not os.path.exists(f'weights/{args.model}'):
        os.makedirs(f'weights/{args.model}')
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists(f'logs/{args.model}'):
        os.makedirs(f'logs/{args.model}')
    # open data config
    with open(args.data, 'r') as f:
        data = yaml.full_load(f)
    # get images and masks, configure datasets and dataloaders
    assert args.task == 'val' or args.task == 'test', f'Task {args.task} is not supported. Choose val or test'
    images = sorted(list(glob.glob(data['path'] + data[args.task] + '/*.jpg')))
    masks = sorted(list(glob.glob(data['path'] + data[args.task].replace('images', 'masks') + '/*.npy')))
    val_dataset = SegmentationDataset(images, masks, (args.img, args.img))
    val_dataloader = DataLoader(val_dataset, args.batch_size, num_workers=2)
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
    ### INIT METRICS ###
    history = {}  # result
    val_steps = ceil(len(val_dataset) / args.batch_size)
    miou = MeanIoUWrapper(data['hierarchy']).to(device)
    print(f'[INFO] Model {args.model} with weights {args.weights}. Starting to validate...')
    with torch.no_grad():
        model.eval()
        with tqdm(val_dataloader, unit='batch') as vepoch:
            for X_val, y_val in vepoch:
                vepoch.set_description('VALIDATING...')
                X_val, y_val = X_val.to(device), y_val.to(device)  # get data
                y_pred = model(X_val)  # get predictions 
                # get stats for metrics
                miou.update(y_pred, y_val.long())
                time.sleep(0.1)

    # save resultsprint(f"[INFO] Val mIoU: {} {} {}")
    print('[INFO] Validation completed. Saving logs...')
    logs = {'model': args.model, 'batch': args.batch_size, 'weights': args.weights, 'data': args.data}
    logs.update(history)
    logs['mIoU'] = np.array(miou.compute()).tolist()
    with open(f'logs/{args.model}/{args.task}_logs.json', 'w') as f:
        json.dump(logs, f)
    print(f"[INFO] Val mIoU: {logs['mIoU'][0]:.3f} {logs['mIoU'][1]:.3f} {logs['mIoU'][2]:.3f}")
