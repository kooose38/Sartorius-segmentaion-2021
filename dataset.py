import pandas as pd 
import numpy as np
import cv2 
import os  
import albumentations as A 
from albumentations.pytorch import ToTensorV2
import torch 
from torch.utils.data import Dataset 

params = {
   "img_size": 224, 
   "mean": (0.485, 0.456, 0.406), 
   "std": (0.229, 0.224, 0.225),
}

class SartoriusTrainDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.df = df
        self.is_train = is_train 
        self.id = df.id.unique()
        self.root = "/kaggle/input/sartorius-cell-instance-segmentation/train"
        self.transform_train = A.Compose([
            A.Resize(params["img_size"], params["img_size"]), 
            A.Normalize(mean=params["mean"], std=params["std"]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2()
        ])
        self.transform_val = A.Compose([
            A.Resize(params["img_size"], params["img_size"]),
            A.Normalize(mean=params["mean"], std=params["std"]),
            ToTensorV2()
        ])
        
    def __getitem__(self, idx):
        id = self.id[idx]
        w, h = self.df.loc[self.df["id"] == id, "width"].tolist()[0], self.df.loc[self.df["id"] == id, "height"].tolist()[0]
        # get input image 
        root_path = os.path.join(self.root, id + ".png")
        img = cv2.imread(root_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # get labels 
        mask = build_masks(self.df, id, input_shape=(h, w))
        mask = (mask >= 1).astype("float32")
        # transform tensor 
        if self.is_train:
            augment = self.transform_train(image=img, mask=mask)
        else:
            augment = self.transform_val(image=img, mask=mask)
        img, mask = augment["image"], augment["mask"]
        return {"image": img, "mask": mask.unsqueeze(0)} # (c, w, h)
        
    def __len__(self):
        return len(self.id)
        

class SartoriusTestDataset(Dataset):
    def __init__(self):
        self.test_path = "/kaggle/input/sartorius-cell-instance-segmentation/test"
        self.image_id = [f.split(".")[0] for f in os.listdir(self.test_path)]
        self.transform = A.Compose([
            A.Resize(params["img_size"], params["img_size"]),
            A.Normalize(mean=params["mean"], std=params["std"]),
            ToTensorV2()
        ])
        
    def __getitem__(self, idx):
        id = self.image_id[idx]
        image = cv2.imread(os.path.join(self.test_path, id + ".png"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]
        return {"image": image, "image_id": id}
        
    def __len__(self):
        return len(self.image_id)


def rle_decode(mask_rle, shape, color=1):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = color
    return img.reshape(shape)


def build_masks(df_train, image_id, input_shape):
    height, width = input_shape
    labels = df_train[df_train["id"] == image_id]["annotation"].tolist()
#     print(labels)
    mask = np.zeros((height, width))
    for label in labels:
        mask += rle_decode(label, shape=(height, width))
    mask = mask.clip(0, 1)
    return mask