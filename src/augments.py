import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from .config import Config
class Augments:
    # Contains Augmentation for train & valid
    train= A.Compose([
        A.Resize(Config['img_size'],Config['img_size'],cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomResizedCrop(Config['img_size'],Config['img_size'],p=0.5),
        A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ToTensorV2()
        ])

    valid= A.Compose([
        A.Resize(Config['img_size'],Config['img_size'],cv2.INTER_NEAREST),
        A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ToTensorV2()
        ])
