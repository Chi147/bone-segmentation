from torch.utils.data import Dataset
import cv2
from torch.utils.tensorboard.summary import image
import numpy as np

def read_xray(path):
    xray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    xray = xray.astype(np.float32)/255.0
    xray= xray.reshape((1,xray.shape[0], xray.shape[1]))
    return xray

def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    print()
    mask = (mask>0).astype(np.float32)
    mask= mask.reshape((1, mask.shape[0], mask.shape[1]))
    return mask

class Knee_dataset(Dataset):
    def __init__(self, df, is_test: bool=False):
        self.df = df
        self.is_test = is_test 
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self,index):
        image = read_xray(self.df['xrays'].iloc[index])

        if self.is_test or 'masks' not in self.df.columns:
            return image

        mask = read_mask(self.df['masks'].iloc[index])
        return image, mask