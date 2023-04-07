import pandas as pd
from .config import Config
import numpy as np
class Helper:
    def load_data(path):
        X,Y= [], []
        classes= os.listdir(path)
        for class_ in classes:     
            class_dir= os.listdir(Config.config['train_path']+class_)
            for image_path in class_dir:
                X.append(image_path)
                Y.append(class_)
        return pd.DataFrame({'image_id':X, 'label':Y}, index= np.arange(0, len(X)))

