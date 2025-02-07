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
# import losses
from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch.losses import JaccardLoss as IoULoss
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
# import metrics
from utils import MeanIoUWrapper
# import models
from models import EfficientUnet, ESNet
# import callbacks
from utils import EarlyStopping, ModelCheckpoint
# to draw results
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='effunet', 
                        help="""Segmentation model name. Available models: effunet, esnet""")
    parser.add_argument('--n-classes', type=int, default=1, 
                        help="""Number of classes in dataset""")
    parser.add_argument('--loss', type=str, default='crossentropy', 
                        help="""Segmentation loss for training. Available loss crossentropy, dice, iou""")
    parser.add_argument('--epochs', type=int, default=30, 
                        help='maximum number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, 
                        help='batch size')
    parser.add_argument('--augment', type=bool, default=True, 
                        help='whether to use augmentation during training')
    parser.add_argument('--img', type=int, default=320, 
                        help='train, val image size (pixels)')
    parser.add_argument('--save-period', type=int, default=-1, 
                        help='Save checkpoint every x epochs')
    parser.add_argument('--patience', type=int, default=5, 
                        help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--data', type=str, default='data/data.yaml', 
                        help='dataset.yaml path')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='optimizer learning rate')
    parser.add_argument('--decay', type=float, default=0.9, 
                        help='optimizer learning rate exponential decay after each epoch')
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
    images = sorted(list(glob.glob(data['path'] + data['train'] + '/*.jpg')))
    masks = sorted(list(glob.glob(data['path'] + data['train'].replace('images', 'masks') + '/*.npy')))
    train_dataset = SegmentationDataset(images, masks, (args.img, args.img), args.augment)
    images = sorted(list(glob.glob(data['path'] + data['val'] + '/*.jpg')))
    masks = sorted(list(glob.glob(data['path'] + data['val'].replace('images', 'masks') + '/*.npy')))
    val_dataset = SegmentationDataset(images, masks, (args.img, args.img), False)
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4)
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

    ### CONFIGURE OPTIMIZERS ###
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=args.decay)
    ### GET LOSS ###
    if args.loss == 'crossentropy':
        train_loss = CrossEntropyLoss().to(device)
        val_loss = CrossEntropyLoss().to(device)
    elif args.loss == 'dice':
        train_loss = DiceLoss(mode='multiclass', smooth=1.0, from_logits=False).to(device)
        val_loss = DiceLoss(mode='multiclass', smooth=1.0, from_logits=False).to(device)
    elif args.loss == 'iou':
        train_loss = IoULoss(mode='multiclass', smooth=1.0, from_logits=False).to(device)
        val_loss = IoULoss(mode='multiclass', smooth=1.0, from_logits=False).to(device)
    else:
        print(f'[WARNING] Loss {args.loss} not implemented, using default CrossEntropyLoss')
        train_loss = CrossEntropyLoss().to(device)
        val_loss = CrossEntropyLoss().to(device)

    ### INIT METRICS ###
    miou = MeanIoUWrapper(data['hierarchy']).to(device)
    history = {'loss' : [], 'val_loss' : [], 'mIoU' : []}  # result
    train_steps = ceil(len(train_dataset) / args.batch_size)  # number of train and val steps
    val_steps = ceil(len(val_dataset) / args.batch_size)
    ### INIT CALLBACKS ###
    es = EarlyStopping(args.patience)
    mc = ModelCheckpoint(args.save_period)
    ### TRAIN LOOP ###
    best_val_loss = 100
    print(f'[INFO] Model {args.model} with loss {args.loss}. Starting to train...')
    for epoch in range(args.epochs):
        model.train()  # set model to training mode
        total_loss = 0  # init total losses and metrics
        total_val_loss = 0
        miou.reset()
        # iterate over batches
        with tqdm(train_dataloader, unit='batch') as tepoch:
            for X_train, y_train in tepoch:
                tepoch.set_description(f'EPOCH {epoch + 1}/{args.epochs} TRAINING')
                X_train, y_train = X_train.to(device), y_train.to(device)  # get data
                y_pred = model(X_train)  # get predictions    
                loss = train_loss(y_pred, y_train.long())  # compute loss
                optimizer.zero_grad()
                loss.backward()  # back propogation
                optimizer.step()  # optimizer's step
                total_loss += loss.item() # add to total loss
                tepoch.set_postfix(loss=loss.item())
                time.sleep(0.1)

        scheduler.step()  # apply lr decay
        history['loss'].append(float(total_loss / train_steps))  # write logs
        print('[INFO] Train loss: {:.4f}\n'.format(history['loss'][-1]))
        print('[INFO] Validating...')
        # perform validation
        with torch.no_grad():
            model.eval()
            with tqdm(val_dataloader, unit='batch') as vepoch:
                for X_val, y_val in vepoch:
                    vepoch.set_description(f'EPOCH {epoch + 1}/{args.epochs} VALIDATING')
                    X_val, y_val = X_val.to(device), y_val.to(device)  # get data
                    y_pred = model(X_val)  # get predictions
                    loss = val_loss(y_pred, y_val.long())
                    total_val_loss += loss.item()  # compute val loss
                    # get miou
                    miou.update(y_pred, y_val.long())
                    vepoch.set_postfix(loss=loss.item())
                    time.sleep(0.1)

        history['val_loss'].append(float(total_val_loss / val_steps))  # write logs
        history['mIoU'].append(miou.compute())
        print("[INFO] Val loss: {:.3f}\nVal mIoU: {:.3f} {:.3f} {:.3f}".format(
            history['val_loss'][-1], history['mIoU'][-1][0], history['mIoU'][-1][1], history['mIoU'][-1][2]))
        if history['val_loss'][-1] < best_val_loss:  # save best weights
            best_val_loss = history['val_loss'][-1]
            torch.save(model.state_dict(), f'weights/{args.model}/{args.model}_best.pt')
        if es.step(history['val_loss'][-1]):  # check early stopping
            print(f'[INFO] Activating early stopping callback at epoch {epoch}')
            break
        if mc.step():  # check model checkpoint
            print(f'[INFO] Activating model checkpoint callback at epoch {epoch}')
            torch.save(model.state_dict(), f'weights/{args.model}/{args.model}_epoch{epoch}.pt')

    print('[INFO] Training finished!')
    torch.save(model.state_dict(), f'weights/{args.model}/{args.model}_last.pt')
    print('[INFO] Saving training logs...')
    logs = {'model': args.model, 'loss_type': args.loss, 'batch': args.batch_size, 'epochs': epoch, 'data': args.data}
    logs.update(history)
    logs['mIoU'] = np.array(logs['mIoU']).tolist()
    with open(f'logs/{args.model}/train_logs.json', 'w') as f:
        json.dump(logs, f)

    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, logs['epochs'] + 2), logs['loss'], '-b', label='train loss')
    plt.plot(range(1, logs['epochs'] + 2), logs['val_loss'], '-r', label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss {args.loss} over epochs')
    plt.legend()
    plt.savefig(f'logs/{args.model}/loss.png', bbox_inches='tight')

    plt.figure(figsize=(10, 5))
    miou = np.array(logs['mIoU'])
    plt.plot(range(1, logs['epochs'] + 2), miou[:, 0], '-b', label='body_iou')
    plt.plot(range(1, logs['epochs'] + 2), miou[:, 1], '-r', label='body_side_miou')
    plt.plot(range(1, logs['epochs'] + 2), miou[:, 2], '-g', label='body_part_miou')
    plt.plot(range(1, logs['epochs'] + 2), miou.mean(axis=-1), '-g', label='total_miou')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.title('mIoU over epochs')
    plt.legend()
    plt.savefig(f'logs/{args.model}/metrics.png', bbox_inches='tight')
