import torch
from torchmetrics.segmentation import MeanIoU
from typing import Dict, Tuple
from torch import Tensor
import numpy as np


class MeanIoUWrapper:
    def __init__(self, hierarchy: Dict):
        self.miou_body = MeanIoU(num_classes=2, include_background=False, input_format='index')  # binary iou for whole body
        self.miou_side = MeanIoU(num_classes=3, include_background=False, input_format='index')  # miou for body side
        self.miou_part = MeanIoU(num_classes=7, include_background=False, input_format='index')  # miou for body part
        self.hierarchy = hierarchy

    def to(self, device):
        self.miou_body.to(device)
        self.miou_side.to(device)
        self.miou_part.to(device)
        return self
    
    def __get_body(self, tensor: Tensor) -> Tensor:
        return (tensor != 0) * 1
    
    def __get_side(self, tensor: Tensor) -> Tensor:
        transformed = torch.where(torch.isin(tensor, torch.tensor(self.hierarchy['body']['upper_body']).to(tensor.device)), 1, 0)
        transformed = torch.where(torch.isin(tensor, torch.tensor(self.hierarchy['body']['lower_body']).to(tensor.device)), 2, transformed)
        return transformed

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds = preds.argmax(dim=1)  # get predicted class
        self.miou_body.update(self.__get_body(preds), self.__get_body(target))  # update body iou
        self.miou_side.update(self.__get_side(preds), self.__get_side(target))  # update body side miou
        self.miou_part.update(preds, target)  # update body part miou

    def reset(self) -> None:
        self.miou_body.reset()
        self.miou_side.reset()
        self.miou_part.reset()

    def compute(self) -> Tuple[np.ndarray]:
        return self.miou_body.compute().cpu().numpy(), self.miou_side.compute().cpu().numpy(),  self.miou_part.compute().cpu().numpy()
