import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from .config import Config
from .models import VITModel
class LightTomato(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model= VITModel()
        self.train_loss= nn.CrossEntropyLoss()
        self.valid_loss= nn.CrossEntropyLoss()

    def forward(self,x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        image, label= batch
        out= self.forward(image)
        train_loss= self.train_loss(out, label)
        self.log('train_loss', train_loss)
        return train_loss
    def validation_step(self, batch, batch_idx):
        image, label= batch
        out= self.forward(image)
        val_loss= self.valid_loss(out, label)
        return {"val_loss": val_loss}
    def validation_epoch_end(self,outputs):
        avg_loss= torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)
        print('Valid Loss= {}'.format(avg_loss))
        return {'val_loss': avg_loss}
    def configure_optimizers(self):
        optimizers= torch.optim.AdamW(self.model.parameters(), lr= 2e-4)
        scheduler= CosineAnnealingLR(optimizers, T_max=20, eta_min= 1e-6)
        return [optimizers], [scheduler]


