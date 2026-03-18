"""
Script containing the Classifier model class
"""
import os
import logging
logging.basicConfig(level=logging.INFO)
# os.environ['MODEL_WEIGHTS'] = os.path.join('helper','model_weights.pt')

from typing import Iterable, List
from torchvision import transforms
from PIL import Image
import cv2
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2
from torchvision.models import efficientnet_b1

# DEVICE = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'

class CLFModel(nn.Module):
    """
    EfficientNet-B1 classifier for ship types
    """
    def __init__(self):
        super().__init__()

        self.classes = [
            'Barge', 'CGP', 'ContainerShip', 'Cruise', 'Dredger', 'Ferry',
            'LNG-LPG', 'RORO', 'Sampan', 'SupplyVessel',
            'Trawler-FishingVessel', 'Warship', 'Yacht'
        ]
        self.num_classes = len(self.classes)

        self.backbone = efficientnet_b1()
        self.backbone.classifier[1] = nn.Linear(1280, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def _preprocess(self):
        """Create preprocessing on demand (Fashion-style)."""
        return transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def predict(self, image_paths):
        """
        image_paths: Iterable[str] – paths to image files
        """
        preprocess = self._preprocess()

        tensors = [
            preprocess(Image.open(path).convert("RGB"))
            for path in image_paths
        ]

        batch = torch.stack(tensors)
        logits = self(batch)
        probs = F.softmax(logits, dim=1)

        return probs.argmax(dim=1).cpu().numpy()



# class CLFModel(nn.Module):
#     """
#     Classifier Model with EfficientNetB1 architecture
#     """
#     def __init__(self):
#         super().__init__() 
#         # f = open(os.environ["CLASSES_TXT"], "r")
#         # self.classes = f.read().split("\n")
#         self.classes = ['Barge', 'CGP', 'ContainerShip', 'Cruise', 'Dredger', 'Ferry', 'LNG-LPG', 'RORO', 'Sampan', 'SupplyVessel', 'Trawler-FishingVessel', 'Warship', 'Yacht']
#         self.num_classes = len(self.classes)
#         self.backbone = efficientnet_b1()
#         self.backbone.classifier[1] = nn.Linear(1280, self.num_classes)
        
#         self.input_w = 320
#         self.input_h = 240
        
#         self.transform_fn = Transform(
#             transform=A.Compose(
#                 [
#                     A.ToFloat(max_value=255),
#                     PadToAspectRatio(self.input_h, self.input_w),
#                     A.Resize(height=self.input_h, width=self.input_w),
#                     ToTensorV2(),
#                 ]
#             )
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.backbone(x)

#     def predict(self, images: List[np.ndarray]):
#         """
#         Do inference on list of images
#         """
#         data = [self.transform_fn(img) for img in images]
#         with torch.no_grad():
#             outputs = self.torch.stack(data)
#             outputs = outputs[:, : self.num_classes]
#             confs = F.softmax(outputs, dim=1)
#         raw_clfs = confs.cpu().detach().numpy()
#         preds = raw_clfs.argmax(axis=1)
#         return preds
    
        
# class Transform:
#     def __init__(self, transform):
#         self.transform = transform
        
#     def __call__(self, image):
#         return self.transform(image=image)["image"]
    
    
# class PadToAspectRatio(ImageOnlyTransform):
#     def __init__(self, h=240, w=320, mode="edge", always_apply=True, p=1.0, **kwargs):
#         super().__init__(always_apply, p)
#         self.h = h
#         self.w = w
#         self.mode = mode
        
#     def apply(self, image: np.ndarray, **kwargs):
#         h_1, w_1, _ = image.shape
        
#         resizing_ratio = min(self.h / h_1, self.w / w_1)
#         resized_h = int(h_1 * resizing_ratio)
#         resized_w = int(w_1 * resizing_ratio)
        
#         padding_size = [
#             int(np.floor((i-j)/resizing_ratio/2))
#             for i, j in zip((self.h, self.w), (resized_h, resized_w))
#         ]
#         hp = padding_size[0]
#         wp = padding_size[1]
#         image = np.pad(image, ((hp, hp), (wp, wp), (0, 0)), self.mode)
#         return image

        # self.model = self._init_model()
        # model.load_state_dict(torch.load(os.environ["MODEL_WEIGHTS"], weights_only=False))
        # self.model = model.to(DEVICE)
        # self.eval()

    # def _init_model(self):
    #     # model = efficientnet_b1(num_classes=self.num_classes)
    #     model = efficientnet_b1()
    #     model.classifier[1] = nn.Linear(1280, self.num_classes)
    #     return model 