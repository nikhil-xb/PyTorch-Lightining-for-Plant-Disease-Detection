import os
class Config:
    config= dict(
            train_path= './input/tomato/train/',
            test_path= './input/tomato/test/',
            img_size= 224,
            wd= 1e-6,
            BATCH= 32,
            NFOLDS= 5,
            EPOCH= 4,
            n_cpu= os.cpu_count(),
            M0DEL_NAME= "vit_base_patch16_224",
            MODEL_PATH= "",
            infr= "Kaggle",
            _wandb_kernel= "nikhil__xb"
            )


