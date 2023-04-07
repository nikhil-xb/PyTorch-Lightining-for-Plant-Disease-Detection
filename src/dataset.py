from .config import Config
from .augments import Augments
import cv2
import numpy as np
from torch.utils.data import Dataset
import os

class Tomato(Dataset):
    def __init__(self, path, data, transform= None):
        super().__init__()
        self.classes= os.listdir(Config.config['train_path'])
        self.path= path
        self.transform= transform
        self.data= data.values

    def __getitem__(self, idx):
        img_id, label= self.data[idx]
        im_path= self.path+ label+ '/'+ img_id
        image= cv2.imread(im_path)
        image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform: 
            image= self.transform(image= image)['image']
        label_= np.zeros(len(self.classes), dtype=float)
        label_[self.classes.index(label)]= 1
        return image, label_
    def __len__(self):
        return len(self.data)

 
