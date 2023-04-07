import timm
import torch.nn as nn
from .config import Config
import torch
class VITModel(nn.Module):
    def __init__(self, num_classes= 11, model_name= Config.config['MODEL_NAME'], pretrained=False)
        super(VITModel, self).__init__()
        self.model= timm.create_model(model_name, pretrained= True)
        if pretrained:
            self.model.load_state_dict(torch.load(Config.config['MODEL_PATH']))
            
        self.model.head= nn.Linear(self.model.head.in_features, num_classes)
    def forward(self,x):
        x= self.model(x)
        return x


