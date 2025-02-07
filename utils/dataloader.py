import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
import numpy as np
from typing import List, Tuple


class SegmentationDataset(Dataset):
    """Класс для загрузки изображений и масок с аугментацией"""
    def __init__(self, images: List[str], masks: List[str], input_shape: Tuple[int, int], augment=True):
        """Class for loading images and detection labels
        and converting them to segmentation masks

        Parameters
        ----------
        images : list
            List of image pathes
        masks : list
            List of mask pathes
        input_shape : tuple
            Input shape of model
        augment : bool, optional
            Whether to use augmentation, by default True
        """
        self.images = images
        self.masks = masks
        self.augment = augment
        self.input_shape = input_shape
        affine_aug = A.OneOf([
              A.Rotate(60, p=0.25),
              A.Flip(p=0.25),
              A.Affine(scale=(0.5, 0.5), p=0.25),
              A.MaskDropout(5, p=0.25)
        ], p=0.5)
        self.transform = A.Compose([
              affine_aug,
              A.OneOf([
                    A.MotionBlur(p=0.125),
                    A.OpticalDistortion(p=0.125),
                    A.ISONoise(p=0.125),
                    A.GaussNoise(p=0.125),
                    A.RandomFog(0.2, 0.6, p=0.125),
                    A.RandomRain(blur_value=2, p=0.125),
                    A.RandomShadow(p=0.125),
                    A.RandomBrightnessContrast(p=0.125)
              ], p=0.5)
        ], p=0.5)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = cv2.cvtColor(cv2.imread(self.images[idx]), cv2.COLOR_BGR2RGB) # open image
        mask = np.load(self.masks[idx])  # open mask
        image = cv2.resize(image, self.input_shape)  # resize image
        mask = cv2.resize(mask, self.input_shape, interpolation=cv2.INTER_NEAREST)  # resize image
        if self.augment:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        mask = torch.from_numpy(mask).int()  # convert to tensor
        image = torch.from_numpy(image / 255.0).float()

        return torch.permute(image, (2, 0, 1)), mask