from src.helper import Helper
from src.config import Config
from src.augments import Augments
from src.models import VITModel
from src.lightning_tomato import LightTomato
from src.dataset import Tomato
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold 

def run_(wandb_logger):
    df= Helper.load_data(Config.config['train_path']) 
    kf= StratifiedKFold(n_splits= Config.config['NFOLDS'])
    for fold_, (train_idx, valid_idx) in enumerate(kf.split(X=df,y= df.label)):
        print(f"{'-'*20} Fold: {fold_} {'-'*20}")
        train_df= df.loc[train_idx]
        valid_df= df.loc[valid_idx]
        
        train_tf= Tomato(Config.config['train_path'],train_df,Augments.train)
        valid_tf= Tomato(Config.config['train_path'],valid_df,Augments.valid)
        

        train_load= DataLoader(train_tf,batch_size=Config.config['BATCH'], num_workers=Config.config['n_cpu'],shuffle= True,pin_memory=True)
        valid_load= DataLoader(valid_tf,batch_size=Config.config['BATCH'], num_workers=Config.config['n_cpu'],shuffle=False)
        
        checkpoint= ModelCheckpoint(monitor="val_loss",dirpath= './', filename=f"Fold={fold_}_Model={Config.config['MODEL_NAME']}", save_top_k=1, mode="min")

        LightModel= LightTomato()
        trainer= pl.Trainer(max_epochs= Config.config['EPOCH'], gpus= 1, callbacks= [checkpoint], logger= wandb_logger, fast_dev_run= False)

        trainer.fit(LightModel, train_load, valid_load)


if __name__=="__main__":
    wandb_logger= WandbLogger(project='Tomato Leaves', group='vision', job_type='train', anonymous= 'allow', config=Config.config)
    run_(wandb_logger)