import yaml
import os
import shutil
from tqdm import tqdm
import numpy as np

with open('data/data.yaml', 'r') as f:
    meta = yaml.full_load(f)

print(meta)


DS_DIR = 'Pascal-part'
DS_TARGET_DIR = meta['path']


# Get train and val ids
with open(f"{DS_DIR}/train_id.txt", 'r') as f:
    train_ids = f.readlines()

train_ids = [id.strip('\n') for id in train_ids]

with open(f"{DS_DIR}/val_id.txt", 'r') as f:
    val_ids = f.readlines()

val_ids = [id.strip('\n') for id in val_ids]

# Reorganize dataset dir
os.makedirs(f"{DS_TARGET_DIR}/train/images", exist_ok=True)
os.makedirs(f"{DS_TARGET_DIR}/train/masks", exist_ok=True)
os.makedirs(f"{DS_TARGET_DIR}/val/images", exist_ok=True)
os.makedirs(f"{DS_TARGET_DIR}/val/masks", exist_ok=True)

for id in tqdm(train_ids, total=len(train_ids)):
    shutil.move(f"{DS_DIR}/JPEGImages/{id}.jpg", f"{DS_TARGET_DIR}/train/images/{id}.jpg")
    shutil.move(f"{DS_DIR}/gt_masks/{id}.npy", f"{DS_TARGET_DIR}/train/masks/{id}.npy")

for id in tqdm(val_ids, total=len(val_ids)):
    shutil.move(f"{DS_DIR}/JPEGImages/{id}.jpg", f"{DS_TARGET_DIR}/val/images/{id}.jpg")
    shutil.move(f"{DS_DIR}/gt_masks/{id}.npy", f"{DS_TARGET_DIR}/val/masks/{id}.npy")
